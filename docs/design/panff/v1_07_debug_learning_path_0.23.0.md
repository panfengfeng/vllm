# V1 调试断点与源码学习路线

> branch_name: 0.23.0
> source_baseline: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (`v0.23.0`)

本文基于当前 `0.23.0` 源码重新整理调试路线。目标是让源码走读有顺序、有断点、
有变量记录，而不是在 EngineCore、Scheduler、Worker、attention backend 之间
来回迷路。

## 1. 第一阶段：只走 V1 主链

先不要读 kernel。第一轮只看对象和请求如何流动：

```text
API server
  -> AsyncLLM.generate()
  -> EngineCore.add_request()
  -> Scheduler.add_request()
  -> EngineCore.step()
  -> Scheduler.schedule()
  -> Executor.execute_model()
  -> Scheduler.update_from_output()
  -> OutputProcessor.process_outputs()
```

推荐断点：

- [`OpenAIServingChat._create_chat_completion()`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L235)
- [`AsyncLLM.generate()`](../../../vllm/v1/engine/async_llm.py#L524)
- [`AsyncLLM.add_request()`](../../../vllm/v1/engine/async_llm.py#L280)
- [`EngineCore.add_request()`](../../../vllm/v1/engine/core.py#L341)
- [`Scheduler.add_request()`](../../../vllm/v1/core/sched/scheduler.py#L1801)
- [`EngineCore.step()`](../../../vllm/v1/engine/core.py#L443)
- [`Scheduler.schedule()`](../../../vllm/v1/core/sched/scheduler.py#L340)
- [`Scheduler.update_from_output()`](../../../vllm/v1/core/sched/scheduler.py#L1329)
- [`OutputProcessor.process_outputs()`](../../../vllm/v1/engine/output_processor.py#L581)

每个断点记录：

```text
request_id
request.status
request.num_tokens
request.num_computed_tokens
len(waiting)
len(running)
num_scheduled_tokens
total_num_scheduled_tokens
```

## 2. 第二阶段：理解 KV Cache

KV 断点：

- [`EngineCore._initialize_kv_caches()`](../../../vllm/v1/engine/core.py#L236)
- [`Scheduler.__init__()`](../../../vllm/v1/core/sched/scheduler.py#L65)
- [`KVCacheManager.__init__()`](../../../vllm/v1/core/kv_cache_manager.py#L110)
- [`KVCacheManager.get_computed_blocks()`](../../../vllm/v1/core/kv_cache_manager.py#L196)
- [`KVCacheManager.allocate_slots()`](../../../vllm/v1/core/kv_cache_manager.py#L238)
- [`KVCacheManager.free()`](../../../vllm/v1/core/kv_cache_manager.py#L438)

记录变量：

```text
block_size
cache_config.num_gpu_blocks
num_new_tokens
num_new_computed_tokens
num_external_computed_tokens
num_lookahead_tokens
num_blocks_to_allocate
block_pool.get_num_free_blocks()
request.block_ids
```

阅读时优先看 Scheduler 侧 block 所有权，再看 Worker 侧 block table。

## 3. 第三阶段：理解 Worker 和 Attention

Worker 断点：

- [`MultiprocExecutor.execute_model()`](../../../vllm/v1/executor/multiproc_executor.py#L307)
- [`WorkerWrapperBase.execute_model()`](../../../vllm/v1/worker/worker_base.py#L340)
- [`GPUWorker.execute_model()`](../../../vllm/v1/worker/gpu_worker.py#L806)
- [`GPUModelRunner.add_requests()`](../../../vllm/v1/worker/gpu/model_runner.py#L742)
- [`GPUModelRunner.update_requests()`](../../../vllm/v1/worker/gpu/model_runner.py#L789)
- [`GPUModelRunner.execute_model()`](../../../vllm/v1/worker/gpu/model_runner.py#L1082)
- [`GPUModelRunner.sample_tokens()`](../../../vllm/v1/worker/gpu/model_runner.py#L1308)

记录变量：

```text
scheduler_output.scheduled_new_reqs
scheduler_output.scheduled_cached_reqs
scheduler_output.num_scheduled_tokens
new_req_data.block_ids
input_batch.req_ids
input_batch.positions
block_tables
slot_mappings
attn_metadata
```

## 4. 第四阶段：理解输出和 abort

输出断点：

- [`AsyncLLM output_handler`](../../../vllm/v1/engine/async_llm.py#L656)
- [`OutputProcessor.process_outputs()`](../../../vllm/v1/engine/output_processor.py#L581)
- [`AsyncLLM.abort()`](../../../vllm/v1/engine/async_llm.py#L709)
- [`OutputProcessor.abort_requests()`](../../../vllm/v1/engine/output_processor.py#L450)
- [`Scheduler.finish_requests()`](../../../vllm/v1/core/sched/scheduler.py#L1825)

记录变量：

```text
engine_core_output.request_id
engine_core_output.new_token_ids
finish_reason
stop_reason
request_output.finished
reqs_to_abort
```

## 5. 第五阶段：关键优化

按顺序读：

1. Continuous batching：
   [`Scheduler.schedule()`](../../../vllm/v1/core/sched/scheduler.py#L340)。
2. Chunked prefill：
   [`scheduler.py#L680`](../../../vllm/v1/core/sched/scheduler.py#L680)
   到 [`scheduler.py#L700`](../../../vllm/v1/core/sched/scheduler.py#L700)。
3. Prefix caching：
   [`KVCacheManager.get_computed_blocks()`](../../../vllm/v1/core/kv_cache_manager.py#L196)。
4. Speculative decoding：
   [`model_runner.py#L1413`](../../../vllm/v1/worker/gpu/model_runner.py#L1413)
   到 [`model_runner.py#L1438`](../../../vllm/v1/worker/gpu/model_runner.py#L1438)。
5. KV transfer：
   [`KVConnectorBase_V1`](../../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L171)。

## 6. 断点观察变量表

这张表可以直接作为走读 checklist。每到一个断点，先看输入对象，再看函数退出后
哪些字段发生变化。

| 断点 | 进入时看 | 退出后看 |
| --- | --- | --- |
| [`AsyncLLM.add_request()`](../../../vllm/v1/engine/async_llm.py#L280) | `request_id`, `prompt`, `params`, `priority`, `data_parallel_rank` | `request.request_id`, `request.params`, `queue.request_id` |
| [`EngineCore.add_request()`](../../../vllm/v1/engine/core.py#L341) | `request.request_id`, `request.kv_transfer_params`, `request.pooling_params` | `scheduler.requests`, `scheduler.waiting` |
| [`Scheduler.add_request()`](../../../vllm/v1/core/sched/scheduler.py#L1801) | `existing`, `request.resumable`, `self.connector` | `self.requests[request_id]`, waiting queue length |
| [`Scheduler.schedule()`](../../../vllm/v1/core/sched/scheduler.py#L340) | `token_budget`, `len(waiting)`, `len(running)` | `SchedulerOutput.num_scheduled_tokens`, `scheduled_new_reqs`, `scheduled_cached_reqs` |
| [`KVCacheManager.get_computed_blocks()`](../../../vllm/v1/core/kv_cache_manager.py#L196) | `request.block_hashes`, `request.num_tokens`, `enable_caching` | `computed_blocks`, `num_new_computed_tokens` |
| [`KVCacheManager.allocate_slots()`](../../../vllm/v1/core/kv_cache_manager.py#L238) | `num_new_tokens`, `num_new_computed_tokens`, `num_external_computed_tokens`, `num_lookahead_tokens` | `new_blocks`, `request.block_ids`, free block count |
| [`GPUModelRunner.add_requests()`](../../../vllm/v1/worker/gpu/model_runner.py#L742) | `scheduled_new_reqs`, `new_req_data.block_ids` | `req_states`, `block_tables.num_blocks` |
| [`GPUModelRunner.prepare_inputs()`](../../../vllm/v1/worker/gpu/model_runner.py#L816) | `num_scheduled_tokens`, `scheduled_spec_decode_tokens` | `input_ids`, `positions`, `query_start_loc`, `logits_indices` |
| [`GPUModelRunner.prepare_attn()`](../../../vllm/v1/worker/gpu/model_runner.py#L981) | `input_batch.idx_mapping`, `input_batch.positions` | `block_tables`, `slot_mappings` |
| [`Scheduler.update_from_output()`](../../../vllm/v1/core/sched/scheduler.py#L1329) | `model_runner_output`, `scheduler_output` | `request.num_computed_tokens`, `request.status`, `EngineCoreOutput` |
| [`OutputProcessor.process_outputs()`](../../../vllm/v1/engine/output_processor.py#L581) | `new_token_ids`, `finish_reason`, `stop_reason` | `RequestOutput`, `reqs_to_abort`, frontend request state |

建议第一次走读时只选一条非流式、非 beam search、非 speculative decoding 的普通
chat 请求。等主线跑通后，再打开 prefix cache、chunked prefill、spec decode 和
KV transfer。

## 7. 常见误区

- 把 API request、EngineCore request、Scheduler request 当成同一个对象。
- 以为启动阶段只是创建服务，没有加载模型和初始化 KV cache。
- 以为一次 `EngineCore.step()` 一定只生成一个 token。
- 以为 prefix cache 命中后会复制一份新的 KV。
- 以为 abort 会立即打断正在运行的 GPU kernel。
- 一开始就读 attention kernel，反而看不懂 block table 和 slot mapping 从哪里来。

## 8. 推荐阅读顺序

1. [v1_01_request_lifecycle_0.23.0.md](v1_01_request_lifecycle_0.23.0.md)
2. [v1_02_overview_startup_0.23.0.md](v1_02_overview_startup_0.23.0.md)
3. [v1_03_online_frontend_to_scheduler_0.23.0.md](v1_03_online_frontend_to_scheduler_0.23.0.md)
4. [v1_04_executor_worker_attention_0.23.0.md](v1_04_executor_worker_attention_0.23.0.md)
5. [v1_05_output_abort_offline_0.23.0.md](v1_05_output_abort_offline_0.23.0.md)
6. [v1_06_scheduler_kv_optimizations_0.23.0.md](v1_06_scheduler_kv_optimizations_0.23.0.md)
7. [v1_07_debug_learning_path_0.23.0.md](v1_07_debug_learning_path_0.23.0.md)

KV cache 细节可以在第 4 步前后补读
[kv_cache_study_manual_0.23.0.md](kv_cache_study_manual_0.23.0.md)。
