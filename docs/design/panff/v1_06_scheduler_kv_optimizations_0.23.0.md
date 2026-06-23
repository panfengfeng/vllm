# V1 Scheduler、KV Cache 与关键优化

> branch_name: 0.23.0
> source_baseline: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (`v0.23.0`)

本文基于当前 `0.23.0` 源码重新整理 Scheduler 与 KV Cache 的优化路径，作为
[kv_cache_study_manual_0.23.0.md](kv_cache_study_manual_0.23.0.md) 的专题补充。

## 1. Scheduler 初始化

入口：
[`Scheduler.__init__()`](../../../vllm/v1/core/sched/scheduler.py#L65)。

初始化时重点看：

- 保存 config：
  [`scheduler.py#L77`](../../../vllm/v1/core/sched/scheduler.py#L77)
  到 [`scheduler.py#L85`](../../../vllm/v1/core/sched/scheduler.py#L85)。
- 计算调度约束：
  [`scheduler.py#L103`](../../../vllm/v1/core/sched/scheduler.py#L103)
  到 [`scheduler.py#L110`](../../../vllm/v1/core/sched/scheduler.py#L110)。
- 创建 KVConnector：
  [`scheduler.py#L116`](../../../vllm/v1/core/sched/scheduler.py#L116)
  到 [`scheduler.py#L136`](../../../vllm/v1/core/sched/scheduler.py#L136)。
- 检查 `num_gpu_blocks`：
  [`scheduler.py#L148`](../../../vllm/v1/core/sched/scheduler.py#L148)
  到 [`scheduler.py#L151`](../../../vllm/v1/core/sched/scheduler.py#L151)。
- 创建 waiting/running 队列：
  [`scheduler.py#L155`](../../../vllm/v1/core/sched/scheduler.py#L155)
  到 [`scheduler.py#L168`](../../../vllm/v1/core/sched/scheduler.py#L168)。

## 2. Continuous Batching

continuous batching 的核心是每个 engine step 都重新调度，把 running decode、
waiting prefill、被跳过的 waiting 请求合并考虑。

入口是 `Scheduler.schedule()`。代码较长，读法是先分 running 和 waiting：

- running 阶段：
  [`scheduler.py#L459`](../../../vllm/v1/core/sched/scheduler.py#L459)
  到 [`scheduler.py#L535`](../../../vllm/v1/core/sched/scheduler.py#L535)。
- waiting 阶段：
  [`scheduler.py#L562`](../../../vllm/v1/core/sched/scheduler.py#L562)
  到 [`scheduler.py#L700`](../../../vllm/v1/core/sched/scheduler.py#L700)。

理解重点：

- running 请求优先尝试继续推进。
- KV block 不够时，Scheduler 可以 preempt 低优先级请求。
- waiting 请求会受到 token budget、seq 数量、LoRA、encoder cache、remote KV
  等约束。

## 3. Chunked Prefill

chunked prefill 允许长 prompt 分多步 prefill，避免一个超长请求占满整个 batch。

关键配置：

- `enable_chunked_prefill`：
  [`scheduler.py#L84`](../../../vllm/config/scheduler.py#L84)。
- `max_num_batched_tokens`：
  [`scheduler.py#L49`](../../../vllm/config/scheduler.py#L49)。
- `max_num_scheduled_tokens`：
  [`scheduler.py#L56`](../../../vllm/config/scheduler.py#L56)。
- `long_prefill_token_threshold`：
  [`scheduler.py#L80`](../../../vllm/config/scheduler.py#L80)。

waiting 请求中裁剪 `num_new_tokens` 的位置：

[`scheduler.py#L680`](../../../vllm/v1/core/sched/scheduler.py#L680)
到 [`scheduler.py#L700`](../../../vllm/v1/core/sched/scheduler.py#L700)。

如果关闭 chunked prefill，且本次 prompt 超过 token budget，Scheduler 会停止
继续调度该请求：

[`scheduler.py#L689`](../../../vllm/v1/core/sched/scheduler.py#L689)
到 [`scheduler.py#L697`](../../../vllm/v1/core/sched/scheduler.py#L697)。

## 4. Prefix Caching

prefix caching 只复用完整 block。请求进入 Scheduler 后，会基于 block hash 查找
最长命中 prefix。

本地命中入口：

[`KVCacheManager.get_computed_blocks()`](../../../vllm/v1/core/kv_cache_manager.py#L196)。

Scheduler 调用位置：

[`scheduler.py#L608`](../../../vllm/v1/core/sched/scheduler.py#L608)
到 [`scheduler.py#L613`](../../../vllm/v1/core/sched/scheduler.py#L613)。

命中后，`allocate_slots()` 会把命中的 blocks 绑定到当前 request：

[`kv_cache_manager.py#L400`](../../../vllm/v1/core/kv_cache_manager.py#L400)
到 [`kv_cache_manager.py#L411`](../../../vllm/v1/core/kv_cache_manager.py#L411)。

真正把新计算完成且可提交的 tokens 放入 prefix cache：

[`kv_cache_manager.py#L420`](../../../vllm/v1/core/kv_cache_manager.py#L420)
到 [`kv_cache_manager.py#L434`](../../../vllm/v1/core/kv_cache_manager.py#L434)。

## 5. Speculative Decoding

speculative decoding 会增加两类 KV 压力：

- 已接受 token 的 KV。
- 尚未验证 draft token 的 lookahead KV。

Scheduler 分配 lookahead：

[`scheduler.py#L462`](../../../vllm/v1/core/sched/scheduler.py#L462)
到 [`scheduler.py#L466`](../../../vllm/v1/core/sched/scheduler.py#L466)。

waiting 阶段对 P/D + spec decode 的特殊限制：

[`scheduler.py#L731`](../../../vllm/v1/core/sched/scheduler.py#L731)
到 [`scheduler.py#L739`](../../../vllm/v1/core/sched/scheduler.py#L739)。

AsyncScheduler 的 placeholder 逻辑：

[`async_scheduler.py#L19`](../../../vllm/v1/core/sched/async_scheduler.py#L19)
到 [`async_scheduler.py#L37`](../../../vllm/v1/core/sched/async_scheduler.py#L37)。

## 6. KV Transfer 与 P/D 分离

KV transfer 把“本地是否已经算过 prefix”扩展为“远端是否已经有 KV”。

Scheduler 侧流程：

```text
local prefix cache hit
  -> connector.get_num_new_matched_tokens()
  -> num_external_computed_tokens
  -> maybe load_kv_async
  -> allocate_slots(..., num_external_computed_tokens, delay_cache_blocks)
```

remote KV 查询：

[`scheduler.py#L615`](../../../vllm/v1/core/sched/scheduler.py#L615)
到 [`scheduler.py#L637`](../../../vllm/v1/core/sched/scheduler.py#L637)。

async remote load 不计算新 token：

[`scheduler.py#L675`](../../../vllm/v1/core/sched/scheduler.py#L675)
到 [`scheduler.py#L679`](../../../vllm/v1/core/sched/scheduler.py#L679)。

`allocate_slots()` 的 external computed tokens 参数说明：

[`kv_cache_manager.py#L263`](../../../vllm/v1/core/kv_cache_manager.py#L263)
到 [`kv_cache_manager.py#L267`](../../../vllm/v1/core/kv_cache_manager.py#L267)。

## 7. Mooncake

Mooncake 是 KVConnector 的一种实现。它不替代 vLLM 本地 KV cache，而是负责跨
角色或跨实例传输 KV。

接口抽象：

- [`KVConnectorBase_V1`](../../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L171)。

注册位置：

- [`factory.py#L212`](../../../vllm/distributed/kv_transfer/kv_connector/factory.py#L212)
  到 [`factory.py#L219`](../../../vllm/distributed/kv_transfer/kv_connector/factory.py#L219)。

读 Mooncake 时建议先看接口方法的调用点，不要直接扎进传输实现：

1. Scheduler 侧如何判断 remote hit。
2. SchedulerOutput 如何携带 connector metadata。
3. Worker forward 前如何 load KV。
4. Worker forward 后如何 save KV。
5. Scheduler 如何接收 transfer finished 状态。

## 8. 走读验收

读完本专题后，应该能手动画出：

```text
waiting request
  -> local prefix cache lookup
  -> remote KV lookup
  -> token budget and chunked prefill
  -> allocate KV slots
  -> worker executes
  -> scheduler updates request
  -> cache/free/transfer states update
```

如果这条线已经清楚，再继续下钻 attention backend 和 kernel 才会比较稳。
