# Explicit LM Eval Baselines

The generic evaluator and baseline scripts remain available. Model-specific
YAMLs and CI lists have been removed from this specialized branch. They must be
provided explicitly; no V4.1 LM Eval Harness baseline is bundled or claimed as
validated on B300.

From this directory, using the repository environment:

```bash
: "${LM_EVAL_CONFIG_LIST:?Set the path to your evaluation config list}"
: "${TP:?Set the tensor parallel size for your hardware}"
../../.venv/bin/python -m pytest -s -v test_lm_eval_correctness.py \
    --config-list-file="$LM_EVAL_CONFIG_LIST" --tp-size="$TP"
```

Each non-comment line in the list names a YAML relative to that list. Missing
or empty lists and missing YAMLs are errors. Each YAML supplies `model_name`,
`num_fewshot`, `limit` and `tasks`; every task supplies its `name` and a `metrics`
list containing `name` and measured baseline `value`. Optional `rtol` controls
the relative tolerance against those values. Supply a baseline measured for
the intended V4.1 checkpoint and evaluation settings instead of borrowing one
from another model.

The retained `run-lm-eval-*-baseline.sh` scripts require explicit model and
sampling/workload arguments. Their usage messages list the applicable flags.
The V4.1 GSM8K/DSpark CI gate uses `tests/evals/gsm8k` and its retained configuration.
