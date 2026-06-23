# V1 Executor、Worker、ModelRunner 与 Attention

> branch_name: 0.23.0
> source_baseline: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (`v0.23.0`)

本文基于当前 `0.23.0` 源码重新整理 EngineCore step 之后的执行链路：Scheduler
产出的 `SchedulerOutput` 如何进入 Executor、Worker、GPUModelRunner，并最终
变成模型 forward 与 attention 对 KV cache 的读写。

## 1. 最短主线

```text
EngineCore.step()
  -> model_executor.execute_model(scheduler_output, non_block=True)
  -> MultiprocExecutor.execute_model()
  -> collective_rpc("execute_model")
  -> WorkerWrapperBase.execute_model()
  -> GPUWorker.execute_model()
  -> GPUModelRunner.execute_model()
  -> add_requests / update_requests / block table
  -> prepare_inputs()
  -> prepare_attn()
  -> set_forward_context()
  -> model(**model_inputs)
  -> sample_tokens()
```

如果是 `UniProcExecutor`，中间没有 worker 子进程和 MQ 广播，调用会更直接。

## 2. MultiprocExecutor.execute_model()

多进程执行入口：
[`MultiprocExecutor.execute_model()`](../../../vllm/v1/executor/multiproc_executor.py#L307)。

它本身不执行模型，而是调用：

[`collective_rpc("execute_model")`](../../../vllm/v1/executor/multiproc_executor.py#L310)
到 [`multiproc_executor.py#L317`](../../../vllm/v1/executor/multiproc_executor.py#L317)。

重要参数：

- `unique_reply_rank=self.output_rank`：通常只从输出 rank 收最终输出。
- `non_block=True`：EngineCore 可以先拿 Future，后面再 `future.result()`。
- `kv_output_aggregator`：有 KVConnector 时用于聚合 worker 侧 connector 输出。

`sample_tokens()` 也走相同 RPC 框架：

[`multiproc_executor.py#L319`](../../../vllm/v1/executor/multiproc_executor.py#L319)
到 [`multiproc_executor.py#L329`](../../../vllm/v1/executor/multiproc_executor.py#L329)。

## 3. collective_rpc()

[`MultiprocExecutor.collective_rpc()`](../../../vllm/v1/executor/multiproc_executor.py#L340)
是 executor 到 workers 的广播 RPC。

阅读重点：

- 检查 executor 是否已经失败：
  [`multiproc_executor.py#L352`](../../../vllm/v1/executor/multiproc_executor.py#L352)
  到 [`multiproc_executor.py#L356`](../../../vllm/v1/executor/multiproc_executor.py#L356)。
- 计算 timeout deadline：
  [`multiproc_executor.py#L358`](../../../vllm/v1/executor/multiproc_executor.py#L358)。
- 如果需要 KV output aggregator，则把多 worker 输出聚合成单个返回：
  [`multiproc_executor.py#L361`](../../../vllm/v1/executor/multiproc_executor.py#L361)
  到 [`multiproc_executor.py#L368`](../../../vllm/v1/executor/multiproc_executor.py#L368)。

这层的核心思想是：EngineCore 只和 executor 交互，executor 再把命令广播给
worker ranks。

## 4. WorkerProc 创建与加载模型

`MultiprocExecutor._init_executor()` 会调用
[`WorkerProc.make_worker_process()`](../../../vllm/v1/executor/multiproc_executor.py#L659)
创建 worker 子进程。

创建子进程的关键代码：

[`multiproc_executor.py#L669`](../../../vllm/v1/executor/multiproc_executor.py#L669)
到 [`multiproc_executor.py#L709`](../../../vllm/v1/executor/multiproc_executor.py#L709)。

worker 子进程入口：

[`WorkerProc.worker_main()`](../../../vllm/v1/executor/multiproc_executor.py#L807)。

它会：

- 设置 signal handler：
  [`multiproc_executor.py#L811`](../../../vllm/v1/executor/multiproc_executor.py#L811)
  到 [`multiproc_executor.py#L827`](../../../vllm/v1/executor/multiproc_executor.py#L827)。
- 创建 `WorkerProc`：
  [`multiproc_executor.py#L846`](../../../vllm/v1/executor/multiproc_executor.py#L846)
  到 [`multiproc_executor.py#L856`](../../../vllm/v1/executor/multiproc_executor.py#L856)。
- 发送 READY 和 response MQ handle 给父进程：
  [`multiproc_executor.py#L862`](../../../vllm/v1/executor/multiproc_executor.py#L862)
  到 [`multiproc_executor.py#L869`](../../../vllm/v1/executor/multiproc_executor.py#L869)。

`WorkerProc.__init__()` 里真正初始化 worker、device 并加载模型：

[`multiproc_executor.py#L593`](../../../vllm/v1/executor/multiproc_executor.py#L593)
到 [`multiproc_executor.py#L652`](../../../vllm/v1/executor/multiproc_executor.py#L652)。

其中：

- `wrapper.init_worker()` 解析 worker class 并创建 worker：
  [`multiproc_executor.py#L604`](../../../vllm/v1/executor/multiproc_executor.py#L604)
  到 [`multiproc_executor.py#L619`](../../../vllm/v1/executor/multiproc_executor.py#L619)。
- `init_device()`：
  [`multiproc_executor.py#L625`](../../../vllm/v1/executor/multiproc_executor.py#L625)
  到 [`multiproc_executor.py#L627`](../../../vllm/v1/executor/multiproc_executor.py#L627)。
- `load_model()`：
  [`multiproc_executor.py#L631`](../../../vllm/v1/executor/multiproc_executor.py#L631)
  到 [`multiproc_executor.py#L634`](../../../vllm/v1/executor/multiproc_executor.py#L634)。
- platform 修正 block size：
  [`multiproc_executor.py#L647`](../../../vllm/v1/executor/multiproc_executor.py#L647)
  到 [`multiproc_executor.py#L648`](../../../vllm/v1/executor/multiproc_executor.py#L648)。

## 5. WorkerWrapperBase

[`WorkerWrapperBase`](../../../vllm/v1/worker/worker_base.py#L187)
是 executor 和真实 worker 之间的外壳。

`init_worker()` 做的事情：

- 保存 `VllmConfig`：
  [`worker_base.py#L237`](../../../vllm/v1/worker/worker_base.py#L237)
  到 [`worker_base.py#L243`](../../../vllm/v1/worker/worker_base.py#L243)。
- 加载插件：
  [`worker_base.py#L245`](../../../vllm/v1/worker/worker_base.py#L245)
  到 [`worker_base.py#L247`](../../../vllm/v1/worker/worker_base.py#L247)。
- 解析 `parallel_config.worker_cls`：
  [`worker_base.py#L249`](../../../vllm/v1/worker/worker_base.py#L249)
  到 [`worker_base.py#L259`](../../../vllm/v1/worker/worker_base.py#L259)。
- 创建真实 worker：
  [`worker_base.py#L311`](../../../vllm/v1/worker/worker_base.py#L311)
  到 [`worker_base.py#L313`](../../../vllm/v1/worker/worker_base.py#L313)。

执行模型时，wrapper 会先应用多模态 receiver cache，再调用真实 worker：

[`worker_base.py#L340`](../../../vllm/v1/worker/worker_base.py#L340)
到 [`worker_base.py#L345`](../../../vllm/v1/worker/worker_base.py#L345)。

## 6. GPUWorker.initialize_from_config()

KV cache 初始化阶段，EngineCore 会通过 executor 调用 worker 的
[`GPUWorker.initialize_from_config()`](../../../vllm/v1/worker/gpu_worker.py#L563)。

关键步骤：

- 写回 profile 后的 `num_gpu_blocks`：
  [`gpu_worker.py#L566`](../../../vllm/v1/worker/gpu_worker.py#L566)
  到 [`gpu_worker.py#L569`](../../../vllm/v1/worker/gpu_worker.py#L569)。
- 初始化 KV transfer connector：
  [`gpu_worker.py#L570`](../../../vllm/v1/worker/gpu_worker.py#L570)
  到 [`gpu_worker.py#L575`](../../../vllm/v1/worker/gpu_worker.py#L575)。
- 调用 model runner 初始化 KV cache：
  [`gpu_worker.py#L577`](../../../vllm/v1/worker/gpu_worker.py#L577)
  到 [`gpu_worker.py#L579`](../../../vllm/v1/worker/gpu_worker.py#L579)。

显存 profile 入口：

[`GPUWorker.determine_available_memory()`](../../../vllm/v1/worker/gpu_worker.py#L372)。

其中会先执行 dummy forward：

[`gpu_worker.py#L404`](../../../vllm/v1/worker/gpu_worker.py#L404)
到 [`gpu_worker.py#L425`](../../../vllm/v1/worker/gpu_worker.py#L425)。

## 7. GPUModelRunner.initialize_kv_cache()

轻量 GPU runner 的 KV 初始化入口：

[`GPUModelRunner.initialize_kv_cache()`](../../../vllm/v1/worker/gpu/model_runner.py#L390)。

主要做：

- 根据 `max_model_len` 和 block size 计算 block table 容量：
  [`model_runner.py#L394`](../../../vllm/v1/worker/gpu/model_runner.py#L394)
  到 [`model_runner.py#L424`](../../../vllm/v1/worker/gpu/model_runner.py#L424)。
- 初始化 attention backend：
  [`model_runner.py#L426`](../../../vllm/v1/worker/gpu/model_runner.py#L426)
  到 [`model_runner.py#L428`](../../../vllm/v1/worker/gpu/model_runner.py#L428)。
- 创建 `BlockTables`：
  [`model_runner.py#L429`](../../../vllm/v1/worker/gpu/model_runner.py#L429)
  到 [`model_runner.py#L439`](../../../vllm/v1/worker/gpu/model_runner.py#L439)。
- 决定 CUDA graph mode：
  [`model_runner.py#L443`](../../../vllm/v1/worker/gpu/model_runner.py#L443)
  到 [`model_runner.py#L456`](../../../vllm/v1/worker/gpu/model_runner.py#L456)。
- 初始化 speculator 相关 attention：
  [`model_runner.py#L457`](../../../vllm/v1/worker/gpu/model_runner.py#L457)
  到 [`model_runner.py#L465`](../../../vllm/v1/worker/gpu/model_runner.py#L465)。
- 创建实际 KV cache tensors：
  [`model_runner.py#L467`](../../../vllm/v1/worker/gpu/model_runner.py#L467)
  到 [`model_runner.py#L470`](../../../vllm/v1/worker/gpu/model_runner.py#L470)。

## 8. GPUWorker.execute_model()

[`GPUWorker.execute_model()`](../../../vllm/v1/worker/gpu_worker.py#L806)
接收 `SchedulerOutput`。

入口阶段重点：

- 等待 pipeline parallel 发送完成：
  [`gpu_worker.py#L809`](../../../vllm/v1/worker/gpu_worker.py#L809)
  到 [`gpu_worker.py#L813`](../../../vllm/v1/worker/gpu_worker.py#L813)。
- 判断本轮是否有 forward：
  [`gpu_worker.py#L815`](../../../vllm/v1/worker/gpu_worker.py#L815)
  到 [`gpu_worker.py#L818`](../../../vllm/v1/worker/gpu_worker.py#L818)。
- PP/SP 特殊路径会先计算 batch descriptor：
  [`gpu_worker.py#L822`](../../../vllm/v1/worker/gpu_worker.py#L822)
  到 [`gpu_worker.py#L840`](../../../vllm/v1/worker/gpu_worker.py#L840)。

普通单 PP 场景可以继续往下看 `self.model_runner.execute_model()` 的调用点。

## 9. GPUModelRunner.add_requests() 和 update_requests()

[`GPUModelRunner.add_requests()`](../../../vllm/v1/worker/gpu/model_runner.py#L742)
消费 `scheduler_output.scheduled_new_reqs`：

- 添加 request state：
  [`model_runner.py#L753`](../../../vllm/v1/worker/gpu/model_runner.py#L753)
  到 [`model_runner.py#L762`](../../../vllm/v1/worker/gpu/model_runner.py#L762)。
- 写入 model state：
  [`model_runner.py#L767`](../../../vllm/v1/worker/gpu/model_runner.py#L767)。
- 把 Scheduler 分配的 `block_ids` 写入 block table：
  [`model_runner.py#L768`](../../../vllm/v1/worker/gpu/model_runner.py#L768)
  到 [`model_runner.py#L770`](../../../vllm/v1/worker/gpu/model_runner.py#L770)。
- last PP rank 上注册 sampler 和 prompt logprobs worker：
  [`model_runner.py#L773`](../../../vllm/v1/worker/gpu/model_runner.py#L773)
  到 [`model_runner.py#L781`](../../../vllm/v1/worker/gpu/model_runner.py#L781)。

[`update_requests()`](../../../vllm/v1/worker/gpu/model_runner.py#L789)
负责给已有请求追加新 block ids，并更新 computed tokens：

[`model_runner.py#L790`](../../../vllm/v1/worker/gpu/model_runner.py#L790)
到 [`model_runner.py#L815`](../../../vllm/v1/worker/gpu/model_runner.py#L815)。

这一步是 Scheduler 侧 block 分配与 Worker 侧 block table 连接起来的地方。

## 10. GPUModelRunner.execute_model()

[`GPUModelRunner.execute_model()`](../../../vllm/v1/worker/gpu/model_runner.py#L1082)
的主流程：

```text
finish/free/add/update request states
  -> block_tables.apply_staged_writes()
  -> dispatch_cg_and_sync_dp()
  -> prepare_inputs()
  -> prepare_attn()
  -> build_slot_mappings_by_layer()
  -> model_state.prepare_attn()
  -> model_state.prepare_inputs()
  -> kv_connector.pre_forward()
  -> cudagraph replay or model(**model_inputs)
  -> save execute_model_state
  -> sample_tokens()
```

关键代码：

- 更新请求状态：
  [`model_runner.py#L1090`](../../../vllm/v1/worker/gpu/model_runner.py#L1090)
  到 [`model_runner.py#L1097`](../../../vllm/v1/worker/gpu/model_runner.py#L1097)。
- 空 batch 走 KV connector no-forward：
  [`model_runner.py#L1098`](../../../vllm/v1/worker/gpu/model_runner.py#L1098)
  到 [`model_runner.py#L1101`](../../../vllm/v1/worker/gpu/model_runner.py#L1101)。
- 计算 batch descriptor：
  [`model_runner.py#L1103`](../../../vllm/v1/worker/gpu/model_runner.py#L1103)
  到 [`model_runner.py#L1124`](../../../vllm/v1/worker/gpu/model_runner.py#L1124)。
- 准备输入和 attention：
  [`model_runner.py#L1131`](../../../vllm/v1/worker/gpu/model_runner.py#L1131)
  到 [`model_runner.py#L1136`](../../../vllm/v1/worker/gpu/model_runner.py#L1136)。
- 构造 layer 级 slot mapping 和 attention metadata：
  [`model_runner.py#L1187`](../../../vllm/v1/worker/gpu/model_runner.py#L1187)
  到 [`model_runner.py#L1202`](../../../vllm/v1/worker/gpu/model_runner.py#L1202)。
- 准备 model inputs：
  [`model_runner.py#L1214`](../../../vllm/v1/worker/gpu/model_runner.py#L1214)
  到 [`model_runner.py#L1221`](../../../vllm/v1/worker/gpu/model_runner.py#L1221)。
- forward：
  [`model_runner.py#L1240`](../../../vllm/v1/worker/gpu/model_runner.py#L1240)
  到 [`model_runner.py#L1276`](../../../vllm/v1/worker/gpu/model_runner.py#L1276)。
- 保存 `execute_model_state`，给 `sample_tokens()` 使用：
  [`model_runner.py#L1293`](../../../vllm/v1/worker/gpu/model_runner.py#L1293)
  到 [`model_runner.py#L1306`](../../../vllm/v1/worker/gpu/model_runner.py#L1306)。

## 11. sample_tokens()

[`GPUModelRunner.sample_tokens()`](../../../vllm/v1/worker/gpu/model_runner.py#L1308)
在 last PP rank 上完成采样和后处理：

- 从 `execute_model_state` 取 hidden states 和 input batch：
  [`model_runner.py#L1313`](../../../vllm/v1/worker/gpu/model_runner.py#L1313)
  到 [`model_runner.py#L1323`](../../../vllm/v1/worker/gpu/model_runner.py#L1323)。
- 非 last PP rank 接收采样结果：
  [`model_runner.py#L1325`](../../../vllm/v1/worker/gpu/model_runner.py#L1325)
  到 [`model_runner.py#L1341`](../../../vllm/v1/worker/gpu/model_runner.py#L1341)。
- last rank 调用 sampler：
  [`model_runner.py#L1343`](../../../vllm/v1/worker/gpu/model_runner.py#L1343)
  到 [`model_runner.py#L1346`](../../../vllm/v1/worker/gpu/model_runner.py#L1346)。
- 构造 `ModelRunnerOutput` 和 async output copy：
  [`model_runner.py#L1367`](../../../vllm/v1/worker/gpu/model_runner.py#L1367)
  到 [`model_runner.py#L1383`](../../../vllm/v1/worker/gpu/model_runner.py#L1383)。
- 更新 request states：
  [`model_runner.py#L1400`](../../../vllm/v1/worker/gpu/model_runner.py#L1400)
  到 [`model_runner.py#L1411`](../../../vllm/v1/worker/gpu/model_runner.py#L1411)。
- speculative decoding proposer：
  [`model_runner.py#L1413`](../../../vllm/v1/worker/gpu/model_runner.py#L1413)
  到 [`model_runner.py#L1438`](../../../vllm/v1/worker/gpu/model_runner.py#L1438)。
- KV connector post-forward：
  [`model_runner.py#L1440`](../../../vllm/v1/worker/gpu/model_runner.py#L1440)
  到 [`model_runner.py#L1446`](../../../vllm/v1/worker/gpu/model_runner.py#L1446)。

## 12. block table 与 slot mapping

Worker 侧 block table 是 Scheduler 分配结果和 attention kernel 的桥：

```text
SchedulerOutput.block_ids
  -> GPUModelRunner.add_requests/update_requests
  -> BlockTables
  -> prepare_attn()
  -> slot_mappings_by_layer
  -> set_forward_context()
  -> attention backend
```

`slot_mapping` 告诉 attention 写入 KV cache 的物理 slot；`block_table` 告诉
attention 读取历史 KV 时 token/block 的映射关系。

在 `GPUModelRunner.execute_model()` 中，二者主要由
[`prepare_attn()`](../../../vllm/v1/worker/gpu/model_runner.py#L1135)
和后续 `build_slot_mappings_by_layer()` 使用。

## 13. 阶段验收

读完本专题后，应能回答：

- `Executor.execute_model()` 为什么通常不直接返回最终 token，而是先返回 Future。
- worker 子进程什么时候加载模型。
- Scheduler 侧 `block_ids` 在 Worker 侧如何进入 block table。
- `execute_model()` 和 `sample_tokens()` 为什么分成两段。
- attention 读写 KV cache 依赖哪些 metadata。
