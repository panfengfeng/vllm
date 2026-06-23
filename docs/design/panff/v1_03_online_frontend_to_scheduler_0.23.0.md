# V1 在线请求：Frontend 到 Scheduler

> branch_name: 0.23.0
> source_baseline: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (`v0.23.0`)

本文基于当前 `0.23.0` 源码重新整理在线请求前半段：HTTP 请求如何进入
OpenAI serving，如何变成 `EngineCoreRequest`，如何通过 ZMQ 进入
EngineCore，并最终进入 Scheduler 的 waiting 队列。

## 1. 最短主线

```text
POST /v1/chat/completions
  -> create_chat_completion()
  -> OpenAIServingChat.create_chat_completion()
  -> OpenAIServingChat._create_chat_completion()
  -> render_chat_request()
  -> request.to_sampling_params() / to_beam_search_params()
  -> engine_client.generate()
  -> AsyncLLM.generate()
  -> AsyncLLM.add_request()
  -> InputProcessor.process_inputs()
  -> OutputProcessor.add_request()
  -> EngineCoreClient.add_request_async()
  -> AsyncMPClient._send_input(ADD)
  -> EngineCore.add_request()
  -> Scheduler.add_request()
```

先把这条线走通，再看 Serving 层细节、beam search 分支、parallel sampling
子请求、流式输出。

## 2. HTTP Router

Chat Completions 的 router 入口是
[`create_chat_completion()`](../../../vllm/entrypoints/openai/chat_completion/api_router.py#L53)。

关键动作：

- 通过 `chat(raw_request)` 取当前模型对应的 serving handler：
  [`api_router.py#L57`](../../../vllm/entrypoints/openai/chat_completion/api_router.py#L57)。
- handler 不存在时返回不支持 Chat Completions：
  [`api_router.py#L58`](../../../vllm/entrypoints/openai/chat_completion/api_router.py#L58)
  到 [`api_router.py#L60`](../../../vllm/entrypoints/openai/chat_completion/api_router.py#L60)。
- 调用 handler：
  [`api_router.py#L61`](../../../vllm/entrypoints/openai/chat_completion/api_router.py#L61)。
- 按返回类型转换为 JSON response 或 streaming response：
  [`api_router.py#L63`](../../../vllm/entrypoints/openai/chat_completion/api_router.py#L63)
  到 [`api_router.py#L70`](../../../vllm/entrypoints/openai/chat_completion/api_router.py#L70)。

## 3. OpenAIServingChat.create_chat_completion()

[`OpenAIServingChat.create_chat_completion()`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L219)
只是外层包装。它通过 `_with_kv_transfer_rejection_cleanup()` 包住真正的
[`_create_chat_completion()`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L235)，
用于处理 KV transfer 相关拒绝清理。

真正的请求构造发生在 `_create_chat_completion()`。

## 4. render_chat_request()

`_create_chat_completion()` 一开始准备 tokenizer、chat template kwargs 和
reasoning parser：

- tokenizer：
  [`serving.py#L240`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L240)
  到 [`serving.py#L243`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L243)。
- reasoning parser：
  [`serving.py#L244`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L244)
  到 [`serving.py#L249`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L249)。
- 渲染 chat 请求：
  [`serving.py#L250`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L250)
  到 [`serving.py#L254`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L254)。

`render_chat_request()` 的职责是把 OpenAI chat messages 应用 chat template，
并生成一个或多个 `engine_inputs`。多模态内容、tokenization、prompt 组件解析也
是在这一阶段被整理成 engine 能理解的输入。

## 5. request_id、LoRA、DP rank

Serving 层会先创建外部 request id：

[`serving.py#L256`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L256)
到 [`serving.py#L263`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L263)。

然后解析 LoRA 和模型名：

[`serving.py#L264`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L264)
到 [`serving.py#L266`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L266)。

如果 router 注入了 data parallel rank，也会在这里读取：

[`serving.py#L268`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L268)
到 [`serving.py#L269`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L269)。

注意 request id 有两层语义：

- OpenAI 层 request id：用于响应、日志和客户端语义。
- EngineCore/Scheduler 层 request id：可能为了 `n>1` 或多 prompt 产生子请求。

## 6. SamplingParams 与 BeamSearchParams

对每个 `engine_input`，Serving 层会先计算本次请求允许生成的 `max_tokens`：

[`serving.py#L271`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L271)
到 [`serving.py#L292`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L292)。

随后根据 `request.use_beam_search` 选择参数类型：

- beam search：
  [`serving.py#L294`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L294)
  到 [`serving.py#L298`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L298)。
- 普通采样：
  [`serving.py#L299`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L299)
  到 [`serving.py#L303`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L303)。

默认走 `SamplingParams`。只有请求显式开启 `use_beam_search` 时，才走
`BeamSearchParams` 和 `beam_search()` 分支。

如何选择：

- 需要 OpenAI 常见的 temperature/top_p/top_k、流式输出、吞吐优先：用
  `SamplingParams`。
- 需要确定性更强的 beam search、多 beam 搜索候选：用 `BeamSearchParams`。

两者后续 generator 不同，是因为 beam search 需要在 Serving 层维护 beam 状态、
反复调用 engine 生成并扩展候选；普通 sampling 则把单个请求交给
`engine_client.generate()`，由 EngineCore/Scheduler 连续推进。

## 7. 调用 engine_client.generate()

非 beam search 分支进入
[`self.engine_client.generate()`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L341)。

传入内容包括：

- `engine_input`
- `sampling_params`
- `sub_request_id`
- `lora_request`
- trace headers
- priority
- data parallel rank
- reasoning parser 状态

对应代码：
[`serving.py#L341`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L341)
到 [`serving.py#L355`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L355)。

## 8. AsyncLLM.generate()

[`AsyncLLM.generate()`](../../../vllm/v1/engine/async_llm.py#L524)
返回一个 async generator。它本身不直接执行模型，而是：

1. 调用 `add_request()` 把请求加入引擎：
   [`async_llm.py#L557`](../../../vllm/v1/engine/async_llm.py#L557)
   到 [`async_llm.py#L571`](../../../vllm/v1/engine/async_llm.py#L571)。
2. 从 per-request queue 中不断取 `RequestOutput`：
   [`async_llm.py#L573`](../../../vllm/v1/engine/async_llm.py#L573)
   到 [`async_llm.py#L586`](../../../vllm/v1/engine/async_llm.py#L586)。
3. 如果客户端断开或 generator 被取消，调用 abort：
   [`async_llm.py#L588`](../../../vllm/v1/engine/async_llm.py#L588)
   到 [`async_llm.py#L596`](../../../vllm/v1/engine/async_llm.py#L596)。

## 9. AsyncLLM.add_request()

[`AsyncLLM.add_request()`](../../../vllm/v1/engine/async_llm.py#L280)
把 Serving 输入转换为 EngineCoreRequest。

关键步骤：

- 判断 pooling / generate：
  [`async_llm.py#L303`](../../../vllm/v1/engine/async_llm.py#L303)。
- 处理 streaming input 特殊分支：
  [`async_llm.py#L316`](../../../vllm/v1/engine/async_llm.py#L316)
  到 [`async_llm.py#L331`](../../../vllm/v1/engine/async_llm.py#L331)。
- 调用 `InputProcessor.process_inputs()`：
  [`async_llm.py#L348`](../../../vllm/v1/engine/async_llm.py#L348)
  到 [`async_llm.py#L360`](../../../vllm/v1/engine/async_llm.py#L360)。
- 分配内部 request id：
  [`async_llm.py#L368`](../../../vllm/v1/engine/async_llm.py#L368)。
- 创建 `RequestOutputCollector`：
  [`async_llm.py#L375`](../../../vllm/v1/engine/async_llm.py#L375)
  到 [`async_llm.py#L379`](../../../vllm/v1/engine/async_llm.py#L379)。
- `n == 1` 直接加入请求；`n > 1` 拆成 parent/children：
  [`async_llm.py#L381`](../../../vllm/v1/engine/async_llm.py#L381)
  到 [`async_llm.py#L397`](../../../vllm/v1/engine/async_llm.py#L397)。

## 10. OutputProcessor 与 EngineCore 同时登记

[`AsyncLLM._add_request()`](../../../vllm/v1/engine/async_llm.py#L400)
做两件事：

```text
OutputProcessor.add_request()
  记录前端侧请求状态、detokenizer、queue

EngineCoreClient.add_request_async()
  把 EngineCoreRequest 发送给后台 EngineCore
```

源码：

[`async_llm.py#L408`](../../../vllm/v1/engine/async_llm.py#L408)
到 [`async_llm.py#L412`](../../../vllm/v1/engine/async_llm.py#L412)。

这是理解 V1 的关键：前端和 EngineCore 各自维护请求状态。前端负责 detokenize、
streaming 和响应；EngineCore 负责调度、执行和 KV cache。

## 11. ZMQ 发送到 EngineCore

AsyncMPClient 的发送入口：

[`add_request_async()`](../../../vllm/v1/engine/core_client.py#L1106)
到 [`core_client.py#L1109`](../../../vllm/v1/engine/core_client.py#L1109)。

同步 MPClient 的同名语义：

[`add_request()`](../../../vllm/v1/engine/core_client.py#L871)
到 [`core_client.py#L874`](../../../vllm/v1/engine/core_client.py#L874)。

发送的是 `EngineCoreRequestType.ADD`，后台 EngineCore 的 input thread 会把这个
请求放入 EngineCore 的 input queue。

## 12. EngineCore.add_request()

后台进程中，
[`EngineCore.add_request()`](../../../vllm/v1/engine/core.py#L341)
负责最后的校验和 Scheduler 入队：

- 校验 request id 类型：
  [`core.py#L347`](../../../vllm/v1/engine/core.py#L347)
  到 [`core.py#L351`](../../../vllm/v1/engine/core.py#L351)。
- pooling task 校验：
  [`core.py#L353`](../../../vllm/v1/engine/core.py#L353)
  到 [`core.py#L362`](../../../vllm/v1/engine/core.py#L362)。
- KV transfer 参数兜底：
  [`core.py#L364`](../../../vllm/v1/engine/core.py#L364)
  到 [`core.py#L370`](../../../vllm/v1/engine/core.py#L370)。
- 进入 Scheduler：
  [`core.py#L372`](../../../vllm/v1/engine/core.py#L372)。

## 13. Scheduler.add_request()

[`Scheduler.add_request()`](../../../vllm/v1/core/sched/scheduler.py#L1801)
负责把请求加入 waiting 队列，并通知 KVConnector：

- streaming input 的重复 request id 会作为后续 chunk 处理：
  [`scheduler.py#L1802`](../../../vllm/v1/core/sched/scheduler.py#L1802)
  到 [`scheduler.py#L1815`](../../../vllm/v1/core/sched/scheduler.py#L1815)。
- 新请求进入 waiting queue：
  [`scheduler.py#L1816`](../../../vllm/v1/core/sched/scheduler.py#L1816)
  到 [`scheduler.py#L1819`](../../../vllm/v1/core/sched/scheduler.py#L1819)。
- 如果存在 KVConnector，调用 `on_new_request()`：
  [`scheduler.py#L1820`](../../../vllm/v1/core/sched/scheduler.py#L1820)
  到 [`scheduler.py#L1821`](../../../vllm/v1/core/sched/scheduler.py#L1821)。
- 开启 stats 时记录 queued event：
  [`scheduler.py#L1822`](../../../vllm/v1/core/sched/scheduler.py#L1822)
  到 [`scheduler.py#L1823`](../../../vllm/v1/core/sched/scheduler.py#L1823)。

到这里，请求已经完成“前端进入 Scheduler”的路径。下一步从
`Scheduler.schedule()` 开始进入动态 batch 和 KV block 分配。

## 14. 阶段验收

读完本专题后，应能回答：

- 为什么 Serving 层默认使用 `SamplingParams`。
- beam search 为什么不直接走普通 `engine_client.generate()` 主线。
- `AsyncLLM.generate()` 返回的 generator 实际从哪里拿输出。
- 为什么 `OutputProcessor` 和 `EngineCore` 都要登记同一个请求。
- 请求在 Scheduler 中初始进入的是 waiting 队列。
