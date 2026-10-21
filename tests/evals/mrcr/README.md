# MRCR Long-Context Accuracy Evaluation

The generic runner evaluates long-context retrieval with the public
[`openai/mrcr`](https://huggingface.co/datasets/openai/mrcr) dataset. Responses
must reproduce a selected earlier assistant turn after a random prefix.
Scoring checks the prefix, then measures text similarity against the reference.

The old Qwen configuration has been removed. No V4.1 quality threshold is
bundled: establish one on the intended checkpoint, context length and hardware
before using this evaluation as a pass/fail gate. B300 has not been validated.

## Measure an already running V4.1 server

```bash
.venv/bin/python tests/evals/mrcr/mrcr_eval.py --port 8000
```

The runner discovers the served model and context limit. `--max-prompt-tokens`
can cap context length; keep the prompt budget, sample count, generation budget
and concurrency fixed when comparing results. This standalone command reports
measurements without supplying an unvalidated quality threshold.

## Run a configured quality gate

Create an evaluation YAML and a text file listing its path relative to the list
file's directory, then pass the list explicitly:

```bash
: "${MRCR_CONFIG_LIST:?Set the path to your MRCR evaluation config list}"
.venv/bin/python -m pytest -s -v tests/evals/mrcr/test_mrcr_correctness.py \
    --config-list-file="$MRCR_CONFIG_LIST"
```

The YAML requires `model_name` and `match_ratio_threshold`. The latter can be a
scalar or a mapping from needle counts to thresholds; derive these values from
your measured baseline. Optional fields include `tolerance`, `server_args`,
`startup_max_wait_seconds`, `num_samples`, `needles`, `max_prompt_tokens`,
`max_tokens`, `concurrency`, `env` and `extra_body`. Specify the server options
needed by your V4.1 checkpoint and available hardware. The pytest runner starts
its own server. Missing lists, missing YAML files and empty lists are errors.

Samples stream from dataset shards and are filtered against the actual server's
tokenized chat length. Preserve the same dataset and prompt settings across
baseline and candidate runs.
