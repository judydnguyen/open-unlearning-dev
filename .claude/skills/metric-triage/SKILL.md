# Skill: Metric Triage for LLM Unlearning

Use this skill when analyzing results, comparing runs, or deciding whether an unlearning change is actually better.

## Goal

Judge experimental quality using the metric structure of this repository, not generic ML instincts.

## Repo Context

- TOFU metrics live in `configs/eval/tofu_metrics/`
- MUSE metrics live in `configs/eval/muse_metrics/`
- Metric implementations live in `src/evals/metrics/`
- Evaluation entrypoint is `src/eval.py`
- Output artifacts are usually written under `saves/eval/*` or `saves/unlearn/*/evals`

## TOFU Reading Rules

Read these together:

- `forget_quality`
- `model_utility`
- forget QA probability metrics
- forget QA ROUGE metrics
- truth-ratio metrics
- MIA, extraction, or memorization metrics when present

Interpretation rules:

- Prefer variants that improve forgetting without a large utility collapse.
- Low forget answer probability alone is not enough.
- If `forget_quality` improves but `model_utility` falls sharply, treat the result as unstable or over-forgetting.
- If forget metrics improve only because the model broadly refuses or degrades, flag that explicitly.
- Check whether the comparison used the same `retain_logs_path`. If not, avoid direct claims.

## MUSE Reading Rules

Read these together:

- `forget_knowmem_ROUGE`
- `forget_verbmem_ROUGE`
- `privleak`
- retain-side ROUGE or knowledge metrics
- any attached MIA metrics

Interpretation rules:

- Strong forgetting with weak retain performance is not a win.
- `privleak` is reference-dependent. Confirm which retain logs were used.
- Distinguish between genuine forgetting and generic output degradation.

## Run Comparison Template

When comparing two runs, report:

1. What changed
2. Which benchmark and split were evaluated
3. The main forgetting gains
4. The main utility or retain losses
5. Whether the new run is clearly better, clearly worse, or a tradeoff

## Red Flags

Flag these immediately:

- missing or mismatched `retain_logs_path`
- evaluating on a different split than the claimed baseline
- comparing a partial eval to a full benchmark run
- celebrating one metric while ignoring a severe collapse elsewhere
- treating leaderboard numbers as directly comparable to runs with different models or splits
