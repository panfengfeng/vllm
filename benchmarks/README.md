# Benchmarks

This directory contains kernel, distributed, and feature benchmarks retained for
the DeepSeek V4.1 specialized fork, along with shared benchmark utilities.
Selected benchmarks for unsupported models and platforms have been pruned.

## Contents

- **Kernel and distributed benchmarks**: Microbenchmarks for attention, MoE,
  quantization, and communication.
- **Feature benchmarks**: Tools for structured output, prefix caching, long
  document QA, request prioritization, and multi-turn serving.
- **Shared utilities**: Request clients and timing helpers used by these benchmarks.

## Usage

The primary serving, latency, and throughput benchmarks live in the vLLM CLI.
The deprecated `benchmark_serving.py`, `benchmark_latency.py`, and
`benchmark_throughput.py` wrappers have been removed. Use:

```bash
vllm bench serve --help
vllm bench latency --help
vllm bench throughput --help
```

Dataset loading and sampling are implemented in `vllm/benchmarks/datasets/`.
When a retained benchmark loads a model, select the supported DeepSeek V4.1
checkpoint explicitly; examples inherited from upstream may name other models.

For detailed usage instructions, examples, and dataset information, see the [Benchmark CLI documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/#benchmark-cli).

For full CLI reference see:

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>
