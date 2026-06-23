# V1 输出、完成、取消与离线请求

> branch_name: 0.23.0
> source_baseline: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (`v0.23.0`)

本文基于当前 `0.23.0` 源码重新整理请求后半段：EngineCore 输出如何回到
AsyncLLM，OutputProcessor 如何 detokenize，取消/abort 如何传播，以及离线
`LLM.generate()` 与在线请求的主要差异。

## 1. 输出主线

```text
EngineCore.step()
  -> Scheduler.update_from_output()
  -> EngineCoreOutputs
  -> AsyncMPClient.get_output_async()
  -> AsyncLLM output_handler
  -> OutputProcessor.process_outputs()
  -> RequestOutputCollector.queue.put()
  -> AsyncLLM.generate() yield RequestOutput
  -> OpenAI Serving serializes response
```

输出不是在 GPU worker 中直接变成文本。worker 主要返回 token ids、logprobs、
pooling output、KV connector output 等；文本 detokenize 在前端进程的
`OutputProcessor` 中完成。

## 2. AsyncLLM output_handler

`AsyncLLM` 会启动后台 asyncio task 处理 EngineCore 输出。入口在
[`async_llm.py#L656`](../../../vllm/v1/engine/async_llm.py#L656)。

主流程：

- 从 EngineCore client 拉输出：
  [`async_llm.py#L658`](../../../vllm/v1/engine/async_llm.py#L658)
  到 [`async_llm.py#L661`](../../../vllm/v1/engine/async_llm.py#L661)。
- 按 chunk 切分，避免长时间阻塞 event loop：
  [`async_llm.py#L667`](../../../vllm/v1/engine/async_llm.py#L667)
  到 [`async_llm.py#L683`](../../../vllm/v1/engine/async_llm.py#L683)。
- 调用 `OutputProcessor.process_outputs()`：
  [`async_llm.py#L674`](../../../vllm/v1/engine/async_llm.py#L674)
  到 [`async_llm.py#L679`](../../../vllm/v1/engine/async_llm.py#L679)。
- 如果 detokenizer 检测到 stop string，需要反向 abort EngineCore：
  [`async_llm.py#L685`](../../../vllm/v1/engine/async_llm.py#L685)
  到 [`async_llm.py#L689`](../../../vllm/v1/engine/async_llm.py#L689)。
- 更新 scheduler stats 和 metrics：
  [`async_llm.py#L691`](../../../vllm/v1/engine/async_llm.py#L691)
  到 [`async_llm.py#L702`](../../../vllm/v1/engine/async_llm.py#L702)。

## 3. OutputProcessor

[`OutputProcessor`](../../../vllm/v1/engine/output_processor.py#L417)
维护前端侧 request state，包括 detokenizer、logprobs processor、parent/child
请求映射和 output queue。

`process_outputs()` 入口：

[`output_processor.py#L581`](../../../vllm/v1/engine/output_processor.py#L581)。

它在一个循环中处理每个 `EngineCoreOutput`：

- 找到 request state：
  [`output_processor.py#L606`](../../../vllm/v1/engine/output_processor.py#L606)
  到 [`output_processor.py#L611`](../../../vllm/v1/engine/output_processor.py#L611)。
- 更新 stats：
  [`output_processor.py#L613`](../../../vllm/v1/engine/output_processor.py#L613)
  到 [`output_processor.py#L616`](../../../vllm/v1/engine/output_processor.py#L616)。
- 取 token ids、finish reason、KV transfer params：
  [`output_processor.py#L618`](../../../vllm/v1/engine/output_processor.py#L618)
  到 [`output_processor.py#L622`](../../../vllm/v1/engine/output_processor.py#L622)。
- detokenize 并检查 stop string：
  [`output_processor.py#L635`](../../../vllm/v1/engine/output_processor.py#L635)
  到 [`output_processor.py#L645`](../../../vllm/v1/engine/output_processor.py#L645)。
- 更新 logprobs：
  [`output_processor.py#L646`](../../../vllm/v1/engine/output_processor.py#L646)
  到 [`output_processor.py#L648`](../../../vllm/v1/engine/output_processor.py#L648)。
- 构造并投递 `RequestOutput`：
  [`output_processor.py#L650`](../../../vllm/v1/engine/output_processor.py#L650)
  到 [`output_processor.py#L667`](../../../vllm/v1/engine/output_processor.py#L667)。
- finished 后清理前端 request state：
  [`output_processor.py#L668`](../../../vllm/v1/engine/output_processor.py#L668)
  到 [`output_processor.py#L688`](../../../vllm/v1/engine/output_processor.py#L688)。

## 4. AsyncLLM.generate() 如何 yield

`AsyncLLM.generate()` 在请求入队后，会循环从 `RequestOutputCollector` 里取输出：

[`async_llm.py#L573`](../../../vllm/v1/engine/async_llm.py#L573)
到 [`async_llm.py#L586`](../../../vllm/v1/engine/async_llm.py#L586)。

这里的 `finished = out.finished` 控制 generator 结束。Serving 层拿到的是一个
async generator，因此可以做 streaming，也可以收集完后一次性返回。

## 5. 正常完成的双侧清理

请求完成时有两侧状态需要清理：

- EngineCore/Scheduler 侧：释放或缓存 KV blocks，移除 running/waiting 状态。
- AsyncLLM/OutputProcessor 侧：移除 detokenizer、queue、external/internal id 映射。

Scheduler 输出更新入口：

[`Scheduler.update_from_output()`](../../../vllm/v1/core/sched/scheduler.py#L1329)。

其中请求停止时会调用 `_handle_stopped_request()` 和 `_free_request()`：

[`scheduler.py#L1520`](../../../vllm/v1/core/sched/scheduler.py#L1520)
到 [`scheduler.py#L1527`](../../../vllm/v1/core/sched/scheduler.py#L1527)。

前端状态清理：

[`OutputProcessor._finish_request()`](../../../vllm/v1/engine/output_processor.py#L695)
到 [`output_processor.py#L705`](../../../vllm/v1/engine/output_processor.py#L705)。

## 6. Stop token 与 stop string

stop token 通常可由 Scheduler/请求状态在 token 级别判断；stop string 需要
detokenize 后才能判断，所以发生在 OutputProcessor。

当 OutputProcessor 检测到 stop string，但 EngineCore 尚未认为请求 finished 时，
会把该 req id 加入 `reqs_to_abort`：

[`output_processor.py#L677`](../../../vllm/v1/engine/output_processor.py#L677)
到 [`output_processor.py#L681`](../../../vllm/v1/engine/output_processor.py#L681)。

随后 output handler 调用 EngineCore abort：

[`async_llm.py#L685`](../../../vllm/v1/engine/async_llm.py#L685)
到 [`async_llm.py#L689`](../../../vllm/v1/engine/async_llm.py#L689)。

## 7. 客户端断开和 abort

如果客户端断开，`AsyncLLM.generate()` 可能收到 `CancelledError` 或
`GeneratorExit`，此时会 abort：

[`async_llm.py#L588`](../../../vllm/v1/engine/async_llm.py#L588)
到 [`async_llm.py#L596`](../../../vllm/v1/engine/async_llm.py#L596)。

`AsyncLLM.abort()` 做双侧处理：

- OutputProcessor 先清理前端状态并生成最终 abort output：
  [`async_llm.py#L709`](../../../vllm/v1/engine/async_llm.py#L709)
  到 [`async_llm.py#L718`](../../../vllm/v1/engine/async_llm.py#L718)。
- EngineCore 收到 abort request ids：
  [`async_llm.py#L717`](../../../vllm/v1/engine/async_llm.py#L717)
  到 [`async_llm.py#L718`](../../../vllm/v1/engine/async_llm.py#L718)。

`OutputProcessor.abort_requests()` 负责 external/internal/parent-child id 映射：

[`output_processor.py#L450`](../../../vllm/v1/engine/output_processor.py#L450)
到 [`output_processor.py#L510`](../../../vllm/v1/engine/output_processor.py#L510)。

EngineCore 侧 abort：

[`EngineCore.abort_requests()`](../../../vllm/v1/engine/core.py#L378)
到 [`core.py#L384`](../../../vllm/v1/engine/core.py#L384)。

Scheduler 侧 finish：

[`Scheduler.finish_requests()`](../../../vllm/v1/core/sched/scheduler.py#L1825)。

## 8. Abort 与正在执行的 GPU batch

abort 不是“立即打断正在运行的 GPU kernel”。更准确地说：

- 请求可能已经在当前 `SchedulerOutput` 中被发给 worker。
- EngineCore 会在模型执行结束后处理 abort queue。
- Scheduler 更新时会忽略或清理已经 abort 的请求状态。

`EngineCore.step()` 中，模型执行结束后才处理 abort queue：

[`core.py#L465`](../../../vllm/v1/engine/core.py#L465)
到 [`core.py#L468`](../../../vllm/v1/engine/core.py#L468)。

## 9. 离线 LLM.generate()

离线入口类：

[`LLM`](../../../vllm/entrypoints/llm.py#L66)。

它同样使用 KV cache 和智能 batching，但调用模型时没有 OpenAI HTTP/Serving
层。典型路径是：

```text
LLM.generate()
  -> _add_completion_requests()
  -> _render_and_add_requests()
  -> LLMEngine.add_request()
  -> _run_engine()
  -> LLMEngine.step()
```

离线批量渲染和添加请求：

[`offline_utils.py#L497`](../../../vllm/entrypoints/offline_utils.py#L497)
到 [`offline_utils.py#L521`](../../../vllm/entrypoints/offline_utils.py#L521)。

逐个添加请求：

[`offline_utils.py#L523`](../../../vllm/entrypoints/offline_utils.py#L523)
到 [`offline_utils.py#L550`](../../../vllm/entrypoints/offline_utils.py#L550)。

`LLM.wait_for_completion()` 会驱动 engine 直到当前队列完成：

[`llm.py#L547`](../../../vllm/entrypoints/llm.py#L547)
到 [`llm.py#L569`](../../../vllm/entrypoints/llm.py#L569)。

## 10. 阶段验收

读完本专题后，应能回答：

- token id 到文本发生在哪个进程。
- stop string 为什么需要前端反向 abort EngineCore。
- abort 为什么不等于立即停止当前 GPU kernel。
- 在线 `AsyncLLM.generate()` 和离线 `LLM.generate()` 的主要差异是什么。
