# VllmConfig 全局配置说明

> branch_name: 0.23.0
> source_baseline: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (`v0.23.0`)

`VllmConfig` 是 V1 启动、调度、执行、KV cache、并行、观测和传输功能之间的
总配置对象。阅读它时不要只局限于启动链路，它是全局控制面：不同模块通过
同一个 `VllmConfig` 读取自己关心的子配置。

源码入口：
[`VllmConfig`](../../../vllm/config/vllm.py#L295)。

## 字段分组

[`VllmConfig`](../../../vllm/config/vllm.py#L295) 的主要字段如下：

| 字段 | 代码位置 | 用途 |
| --- | --- | --- |
| `model_config` | [`vllm.py#L302`](../../../vllm/config/vllm.py#L302) | 模型名称、架构、`max_model_len`、runner type、多模态等。 |
| `cache_config` | [`vllm.py#L304`](../../../vllm/config/vllm.py#L304) | KV cache block size、GPU blocks、prefix caching 等。 |
| `parallel_config` | [`vllm.py#L306`](../../../vllm/config/vllm.py#L306) | TP、PP、DP、DCP/PCP、Ray/local backend 等并行配置。 |
| `scheduler_config` | [`vllm.py#L308`](../../../vllm/config/vllm.py#L308) | batch token budget、max seqs、chunked prefill、调度策略。 |
| `device_config` | [`vllm.py#L312`](../../../vllm/config/vllm.py#L312) | CUDA/CPU/其他平台设备选择。 |
| `load_config` | [`vllm.py#L314`](../../../vllm/config/vllm.py#L314) | 权重加载来源和方式。 |
| `offload_config` | [`vllm.py#L316`](../../../vllm/config/vllm.py#L316) | 权重 offload 配置。 |
| `attention_config` | [`vllm.py#L318`](../../../vllm/config/vllm.py#L318) | attention backend 相关配置。 |
| `mamba_config` | [`vllm.py#L320`](../../../vllm/config/vllm.py#L320) | Mamba/SSM 模型相关配置。 |
| `kernel_config` | [`vllm.py#L322`](../../../vllm/config/vllm.py#L322) | kernel 选择和调优配置。 |
| `lora_config` | [`vllm.py#L324`](../../../vllm/config/vllm.py#L324) | LoRA 加载和并发限制。 |
| `speculative_config` | [`vllm.py#L326`](../../../vllm/config/vllm.py#L326) | speculative decoding 配置。 |
| `structured_outputs_config` | [`vllm.py#L328`](../../../vllm/config/vllm.py#L328) | JSON/schema/grammar 约束输出。 |
| `observability_config` | [`vllm.py#L332`](../../../vllm/config/vllm.py#L332) | metrics、tracing、iteration detail logging。 |
| `quant_config` | [`vllm.py#L336`](../../../vllm/config/vllm.py#L336) | 量化配置。 |
| `compilation_config` | [`vllm.py#L338`](../../../vllm/config/vllm.py#L338) | `torch.compile`、CUDA graph capture 等。 |
| `profiler_config` | [`vllm.py#L347`](../../../vllm/config/vllm.py#L347) | profiler 开关和输出目录。 |
| `kv_transfer_config` | [`vllm.py#L349`](../../../vllm/config/vllm.py#L349) | P/D 分离、remote KV、Mooncake/NIXL 等 KV 传输。 |
| `kv_events_config` | [`vllm.py#L351`](../../../vllm/config/vllm.py#L351) | KV cache event 发布。 |
| `ec_transfer_config` | [`vllm.py#L353`](../../../vllm/config/vllm.py#L353) | encoder cache transfer。 |
| `reasoning_config` | [`vllm.py#L355`](../../../vllm/config/vllm.py#L355) | reasoning model 相关配置。 |
| `additional_config` | [`vllm.py#L360`](../../../vllm/config/vllm.py#L360) | 平台扩展配置，要求可 hash。 |
| `optimization_level` | [`vllm.py#L366`](../../../vllm/config/vllm.py#L366) | 启动开销和运行性能之间的优化等级。 |
| `performance_mode` | [`vllm.py#L372`](../../../vllm/config/vllm.py#L372) | balanced/interactivity/throughput 运行偏好。 |
| `weight_transfer_config` | [`vllm.py#L379`](../../../vllm/config/vllm.py#L379) | RL 训练时权重传输。 |
| `shutdown_timeout` | [`vllm.py#L382`](../../../vllm/config/vllm.py#L382) | shutdown 时等待 in-flight 请求完成的时间。 |

## KV Cache 相关字段

KV Cache 走读时最常接触这些子配置：

- `model_config.max_model_len`：单条序列最多允许多少 token，包括 prompt 和
  output。它会影响 KV cache admission 和最大可服务长度。
- `cache_config.block_size`：一个 KV block 对应多少 token。`block_size = 16`
  表示每个 block 的逻辑容量是 16 个 token 的 K/V；实际物理大小还要乘以
  层数、KV heads、head size、dtype、cache group 等因素。
- `cache_config.num_gpu_blocks`：EngineCore profile 显存后写回的可用 GPU
  block 数。Scheduler 初始化要求它已经是正数。
- `scheduler_config.max_num_batched_tokens`：单步最多处理多少 token，影响
  continuous batching 和 chunked prefill。
- `scheduler_config.max_num_scheduled_tokens`：Scheduler 单步最多发出多少
  token；如果为空，默认等于 `max_num_batched_tokens`。
- `scheduler_config.max_num_seqs`：单步最多并发多少条序列。
- `scheduler_config.enable_chunked_prefill`：是否允许长 prompt 分块 prefill。
- `scheduler_config.async_scheduling`：是否启用 `AsyncScheduler`。
- `speculative_config`：存在时启用 speculative decoding，并带来 lookahead KV。
- `kv_transfer_config`：存在时创建 KVConnector，用于 remote KV load/save。

## SchedulerConfig 入口

`SchedulerConfig` 定义在
[`scheduler.py#L26`](../../../vllm/config/scheduler.py#L26)。

与 KV 最相关的字段：

- `max_num_batched_tokens`：
  [`scheduler.py#L49`](../../../vllm/config/scheduler.py#L49)。
- `max_num_scheduled_tokens`：
  [`scheduler.py#L56`](../../../vllm/config/scheduler.py#L56)。
- `max_num_seqs`：
  [`scheduler.py#L63`](../../../vllm/config/scheduler.py#L63)。
- `max_num_partial_prefills`：
  [`scheduler.py#L70`](../../../vllm/config/scheduler.py#L70)。
- `max_long_partial_prefills`：
  [`scheduler.py#L74`](../../../vllm/config/scheduler.py#L74)。
- `long_prefill_token_threshold`：
  [`scheduler.py#L80`](../../../vllm/config/scheduler.py#L80)。
- `enable_chunked_prefill`：
  [`scheduler.py#L84`](../../../vllm/config/scheduler.py#L84)。
- `policy`：
  [`scheduler.py#L109`](../../../vllm/config/scheduler.py#L109)。
- `scheduler_reserve_full_isl`：
  [`scheduler.py#L140`](../../../vllm/config/scheduler.py#L140)。
- `async_scheduling`：
  [`scheduler.py#L146`](../../../vllm/config/scheduler.py#L146)。

## Scheduler 类型如何选择

[`SchedulerConfig.get_scheduler_cls()`](../../../vllm/config/scheduler.py#L168)
的规则很直接：

- `scheduler_cls` 为空且 `async_scheduling=True`：返回 `AsyncScheduler`。
- `scheduler_cls` 为空且 `async_scheduling` 不是 True：返回普通 `Scheduler`。
- `scheduler_cls` 不为空：按自定义类或 qualname 解析。

代码：

- `AsyncScheduler` 分支：
  [`scheduler.py#L169`](../../../vllm/config/scheduler.py#L169)
  到 [`scheduler.py#L173`](../../../vllm/config/scheduler.py#L173)。
- 默认 `Scheduler` 分支：
  [`scheduler.py#L174`](../../../vllm/config/scheduler.py#L174)
  到 [`scheduler.py#L176`](../../../vllm/config/scheduler.py#L176)。

是否默认启用 async scheduling 取决于 engine args 和平台能力最终如何写入
`scheduler_config.async_scheduling`。源码走读时不要只看 dataclass 默认值，要在
实际启动日志或断点中看 `vllm_config.scheduler_config.async_scheduling`。

## compute_hash()

[`VllmConfig.compute_hash()`](../../../vllm/config/vllm.py#L388)
用于生成影响计算图结构的配置 hash。注意它不是所有运行时行为的 hash，而是
聚焦“从 input ids/embeddings 到 final hidden states”的图结构相关因素。

阅读重点：

- 版本号会进入 hash：
  [`vllm.py#L404`](../../../vllm/config/vllm.py#L404)
  到 [`vllm.py#L407`](../../../vllm/config/vllm.py#L407)。
- `model_config`、`cache_config`、`parallel_config`、`scheduler_config` 等会继续
  贡献子 hash：
  [`vllm.py#L407`](../../../vllm/config/vllm.py#L407)
  到 [`vllm.py#L426`](../../../vllm/config/vllm.py#L426)。

这也是为什么 `VllmConfig` 需要从全局角度理解：它不只是启动参数集合，还会影响
编译、图缓存、worker 初始化、调度和 KV cache 行为。
