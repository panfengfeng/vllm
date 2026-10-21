# DeepSeek V4.1 GSM8K Accuracy Evaluation

The retained configuration is
[`configs/DeepSeek-V4.1-Flash-NVFP4-DSpark-TP4.yaml`](configs/DeepSeek-V4.1-Flash-NVFP4-DSpark-TP4.yaml).
It evaluates `nvidia/DeepSeek-V4.1-Flash-NVFP4` with TP4, five DSpark draft tokens,
1,319 questions and five few-shot examples. Its existing accuracy threshold is
0.88 with 0.03 tolerance (minimum passing accuracy 0.85), and its minimum mean
acceptance length is 2.5. These thresholds have not been validated on B300 for
this specialized branch.

## Run the retained evaluation

From the repository root, in an environment with the GPU evaluation dependencies:

```bash
.venv/bin/python -m pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py \
    --config-list-file=configs/models-deepseek-v41.txt
```

This list is also the pytest default. The CI job retains the existing optional
4×B200 allocation and 120-minute timeout; it is not a B300 acceptance result.
The test starts its own server and requires access to the configured checkpoint.

For another deployment, provide a config list explicitly. Each non-comment line
names a YAML relative to the list's directory. Copy the retained configuration
and set the checkpoint, server arguments and locally established quality
thresholds. Missing lists, missing YAML files and empty lists are errors.

## Evaluate an already running server

```bash
.venv/bin/python tests/evals/gsm8k/gsm8k_eval.py --port 8000
```

The standalone runner discovers the served model when `--model` is omitted.
Use `--use-chat-completions` when evaluating chat prompts, and set generation
parameters explicitly for comparisons. `--save-results` records the request
settings used. The server-based pytest path additionally checks that DSpark
acceptance counters show useful drafts; accuracy alone does not establish that.

## KV offloading coverage

`test_gsm8k_offloading.py` retains generic KV offload/reload regression logic.
Its legacy model fixtures have not been migrated to V4.1 and are excluded from
the specialized CI job. Running that suite is not part of this branch's current
V4.1 acceptance set.
