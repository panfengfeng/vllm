# vLLM V1 请求生命周期源码走读总览

> branch_name: 0.23.0
> source_baseline: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (`v0.23.0`)

本文是基于当前 `releases/v0.23.0` 源码重新整理的 V1 走读入口，不复制
`0.22.1` 文档内容。它的用途是先建立一张“从 API server 到 EngineCore、
Scheduler、Executor、Worker、KV Cache”的导航图，再把细节分流到专题文档。

## 文档地图

| 主题 | 文档 |
| --- | --- |
| V1 启动、进程和对象初始化 | [v1_02_overview_startup_0.23.0.md](v1_02_overview_startup_0.23.0.md) |
| 在线请求：Frontend 到 Scheduler | [v1_03_online_frontend_to_scheduler_0.23.0.md](v1_03_online_frontend_to_scheduler_0.23.0.md) |
| Executor、Worker、ModelRunner 与 Attention | [v1_04_executor_worker_attention_0.23.0.md](v1_04_executor_worker_attention_0.23.0.md) |
| 输出、完成、取消与离线请求 | [v1_05_output_abort_offline_0.23.0.md](v1_05_output_abort_offline_0.23.0.md) |
| VllmConfig 全局配置说明 | [vllm_config_0.23.0.md](vllm_config_0.23.0.md) |
| KV Cache 源码学习手册 | [kv_cache_study_manual_0.23.0.md](kv_cache_study_manual_0.23.0.md) |
| Scheduler、KV 与优化专题 | [v1_06_scheduler_kv_optimizations_0.23.0.md](v1_06_scheduler_kv_optimizations_0.23.0.md) |
| 调试断点与源码学习路线 | [v1_07_debug_learning_path_0.23.0.md](v1_07_debug_learning_path_0.23.0.md) |

## 一条最短主线

```text
OpenAI API server
  -> build_async_engine_client_from_engine_args()
  -> AsyncLLM.from_vllm_config()
  -> AsyncLLM.__init__()
  -> EngineCoreClient.make_async_mp_client()
  -> MPClient.__init__()
  -> launch_core_engines()
  -> EngineCoreProc.run_engine_core()
  -> EngineCoreProc.__init__()
  -> EngineCore.__init__()
  -> EngineCore._initialize_kv_caches()
  -> Scheduler(...)
  -> EngineCore.step()
  -> Scheduler.schedule()
  -> KVCacheManager.allocate_slots()
  -> Executor.execute_model()
  -> GPUWorker.execute_model()
  -> GPUModelRunner.execute_model()
  -> Scheduler.update_from_output()
```

这条线要优先走通。先不要急着读 CUDA kernel 或 attention backend，先看清楚
请求在 Python 侧如何被创建、排队、分配 KV block、执行模型、回收输出。

## 当前版本关键入口

- API server 构造 V1 engine client：
  [`build_async_engine_client_from_engine_args()`](../../../vllm/entrypoints/openai/api_server.py#L108)
  会先通过 `engine_args.create_engine_config()` 得到 `VllmConfig`，再调用
  `AsyncLLM.from_vllm_config()`。
- `AsyncLLM.from_vllm_config()`：
  [`async_llm.py#L203`](../../../vllm/v1/engine/async_llm.py#L203)
  通过 `Executor.get_class(vllm_config)` 选择 executor class。
- `AsyncLLM.__init__()`：
  [`async_llm.py#L73`](../../../vllm/v1/engine/async_llm.py#L73)
  创建 renderer、`InputProcessor`、`OutputProcessor`，并启动
  `EngineCoreClient`。
- 多进程客户端选择：
  [`EngineCoreClient.make_async_mp_client()`](../../../vllm/v1/engine/core_client.py#L108)
  根据 DP 配置选择 `AsyncMPClient`、`DPAsyncMPClient` 或
  `DPLBAsyncMPClient`。
- EngineCore 后台进程：
  [`EngineCoreProc.run_engine_core()`](../../../vllm/v1/engine/core.py#L1118)
  决定创建 `DPEngineCoreProc` 还是 `EngineCoreProc`。
- EngineCore 初始化：
  [`EngineCore.__init__()`](../../../vllm/v1/engine/core.py#L98)
  先创建 `model_executor`，再初始化 KV cache，最后创建 Scheduler。
- 单步推理主循环：
  [`EngineCore.step()`](../../../vllm/v1/engine/core.py#L443)
  串起 `schedule -> execute_model -> update_from_output`。

## 三条代码走读路线

### 路线 A：启动到 ready

适合回答“模型是否已加载、KV cache 是否已初始化、executor class 是什么”。

```text
build_async_engine_client_from_engine_args()
  -> AsyncLLM.from_vllm_config()
  -> AsyncLLM.__init__()
  -> EngineCoreClient.make_async_mp_client()
  -> MPClient.__init__()
  -> launch_core_engines()
  -> EngineCoreProc.run_engine_core()
  -> EngineCoreProc.__init__()
  -> EngineCore.__init__()
  -> _initialize_kv_caches()
  -> MPClient._apply_ready_response()
```

主文档：
[v1_02_overview_startup_0.23.0.md](v1_02_overview_startup_0.23.0.md)。

建议最少断点：

- [`AsyncLLM.from_vllm_config()`](../../../vllm/v1/engine/async_llm.py#L203)
- [`MPClient.__init__()`](../../../vllm/v1/engine/core_client.py#L477)
- [`EngineCore.__init__()`](../../../vllm/v1/engine/core.py#L98)
- [`EngineCore._initialize_kv_caches()`](../../../vllm/v1/engine/core.py#L236)
- [`MPClient._apply_ready_response()`](../../../vllm/v1/engine/core_client.py#L711)

### 路线 B：一个在线 chat 请求进入 Scheduler

适合回答“OpenAI 请求如何变成 Scheduler request”。

```text
create_chat_completion()
  -> OpenAIServingChat._create_chat_completion()
  -> engine_client.generate()
  -> AsyncLLM.generate()
  -> AsyncLLM.add_request()
  -> InputProcessor.process_inputs()
  -> EngineCoreClient.add_request_async()
  -> EngineCore.add_request()
  -> Scheduler.add_request()
```

主文档：
[v1_03_online_frontend_to_scheduler_0.23.0.md](v1_03_online_frontend_to_scheduler_0.23.0.md)。

建议最少断点：

- [`OpenAIServingChat._create_chat_completion()`](../../../vllm/entrypoints/openai/chat_completion/serving.py#L235)
- [`AsyncLLM.generate()`](../../../vllm/v1/engine/async_llm.py#L524)
- [`AsyncLLM.add_request()`](../../../vllm/v1/engine/async_llm.py#L280)
- [`EngineCore.add_request()`](../../../vllm/v1/engine/core.py#L341)
- [`Scheduler.add_request()`](../../../vllm/v1/core/sched/scheduler.py#L1801)

### 路线 C：一次 EngineCore step

适合回答“KV block 如何分配、worker 如何执行、输出如何回来”。

```text
EngineCore.run_busy_loop()
  -> EngineCore.step()
  -> Scheduler.schedule()
  -> KVCacheManager.get_computed_blocks()
  -> KVCacheManager.allocate_slots()
  -> MultiprocExecutor.execute_model()
  -> GPUWorker.execute_model()
  -> GPUModelRunner.execute_model()
  -> GPUModelRunner.sample_tokens()
  -> Scheduler.update_from_output()
  -> OutputProcessor.process_outputs()
```

主文档：
[kv_cache_study_manual_0.23.0.md](kv_cache_study_manual_0.23.0.md)、
[v1_04_executor_worker_attention_0.23.0.md](v1_04_executor_worker_attention_0.23.0.md)、
[v1_05_output_abort_offline_0.23.0.md](v1_05_output_abort_offline_0.23.0.md)。

建议最少断点：

- [`EngineCore.step()`](../../../vllm/v1/engine/core.py#L443)
- [`Scheduler.schedule()`](../../../vllm/v1/core/sched/scheduler.py#L340)
- [`KVCacheManager.allocate_slots()`](../../../vllm/v1/core/kv_cache_manager.py#L238)
- [`GPUModelRunner.execute_model()`](../../../vllm/v1/worker/gpu/model_runner.py#L1082)
- [`Scheduler.update_from_output()`](../../../vllm/v1/core/sched/scheduler.py#L1329)
- [`OutputProcessor.process_outputs()`](../../../vllm/v1/engine/output_processor.py#L581)

## 推荐走读顺序

如果只读 `v1_*.md` 文件，可以直接按文件名编号读：

1. [v1_01_request_lifecycle_0.23.0.md](v1_01_request_lifecycle_0.23.0.md)
2. [v1_02_overview_startup_0.23.0.md](v1_02_overview_startup_0.23.0.md)
3. [v1_03_online_frontend_to_scheduler_0.23.0.md](v1_03_online_frontend_to_scheduler_0.23.0.md)
4. [v1_04_executor_worker_attention_0.23.0.md](v1_04_executor_worker_attention_0.23.0.md)
5. [v1_05_output_abort_offline_0.23.0.md](v1_05_output_abort_offline_0.23.0.md)
6. [v1_06_scheduler_kv_optimizations_0.23.0.md](v1_06_scheduler_kv_optimizations_0.23.0.md)
7. [v1_07_debug_learning_path_0.23.0.md](v1_07_debug_learning_path_0.23.0.md)

如果要补齐 KV cache 和配置背景，再穿插阅读
[vllm_config_0.23.0.md](vllm_config_0.23.0.md)、
[kv_cache_study_manual_0.23.0.md](kv_cache_study_manual_0.23.0.md)。

## 版本提醒

`0.23.0` 与 `0.22.1` 的代码行号和部分调度细节已经不同。后续所有源码链接应以
当前分支为准，不再沿用 `*_0.22.1.md` 中的行号。
