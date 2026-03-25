# Claude Guidelines for Open-Unlearning Research

This repository is an active research workspace for LLM unlearning. When working here, optimize for scientific signal, reproducibility, and safety over speed.

## Project Mental Model

- Main entrypoints:
  - `src/train.py` for finetuning and unlearning runs
  - `src/eval.py` for benchmark evaluation
- Main config roots:
  - `configs/experiment/*` for experiment presets
  - `configs/trainer/*` for method selection and method hyperparameters
  - `configs/eval/*` and `configs/eval/*_metrics/*` for benchmark and metric definitions
- Main implementation roots:
  - `src/trainer/unlearn/*` for unlearning methods
  - `src/evals/*` and `src/evals/metrics/*` for evaluators and metrics
- Main artifact location:
  - `saves/{mode}/{task_name}`

Hydra is the source of truth for how experiments are assembled. Before changing code, inspect the trainer config, experiment config, and launch script that currently define the run.

## Research Priorities

Treat this repo as a multi-objective optimization problem.

- Better forgetting is not enough if model utility collapses.
- Better utility is not enough if the model still memorizes or leaks.
- A result only counts if it is reproducible from code, config, and saved outputs.

When comparing variants, prefer Pareto-style reasoning across forgetting, retention, and privacy rather than chasing one scalar.

## Metric Reading Rules

For TOFU:

- Read `forget_quality` together with `model_utility`.
- Check supporting signals such as forget QA probability, ROUGE, truth-ratio-style metrics, and MIA metrics when available.
- Do not claim success from low forget probability alone if utility or retain behavior collapses.
- Remember that `forget_quality` depends on `retain_logs_path`, so comparisons are only meaningful when the reference logs are compatible.

For MUSE:

- Read forgetting metrics and retain metrics together.
- Interpret `privleak` only in context with the reference retain logs and the underlying MIA-style statistics.
- Do not claim success from strong forget ROUGE drops if retain knowledge quality degrades materially.

For WMDP or `lm_eval` style evaluation:

- Treat those as capability checks after unlearning, not direct forgetting scores.
- Use them to detect collateral damage and general capability drift.

## Allowed Changes

Usually safe to change:

- Trainer internals for the method currently under study
- Trainer hyperparameters in Hydra configs
- New experiment configs or launch scripts
- Logging, diagnostics, ablations, and targeted tests
- Small evaluation additions that do not silently redefine benchmark semantics

Higher risk and require extra caution:

- Metric formulas or benchmark aggregation logic
- Dataset wiring and split semantics
- Reference log usage and `retain_logs_path` assumptions
- Output schema changes that would break comparison with old runs
- Broad refactors that touch unrelated methods

When in doubt, preserve benchmark semantics and add a new config, metric, or trainer variant instead of mutating old behavior in place.

## Validation Expectations

Any method change should be paired with at least one of:

- a targeted unit test
- a smoke script
- a short eval on the relevant benchmark
- an apples-to-apples comparison against the current best-known variant

Before declaring a change promising, capture:

- exact command or script used
- task name
- checkpoint path
- eval output path
- the main metrics that improved
- the main metrics that regressed

## Best-Version and Rollback Rules

There are three separate things to track:

1. Best code state
2. Best config or launch recipe
3. Best checkpoint and eval artifacts

Do not assume they are all represented by the latest file edit.

When asked to revert to the best-known method:

- first identify the best-known run from scripts, comments, saved evals, and experiment outputs
- recover the exact command or Hydra overrides that produced it
- recover the matching checkpoint under `saves/`
- only then consider code rollback in Git if the implementation itself changed

Never delete newer experiment artifacts just to get back to an older baseline.

## Working Style

- Prefer small, reversible changes.
- Use descriptive `task_name`s for every run.
- Keep new variants in new configs or scripts when possible.
- Preserve comments in experiment scripts that explain why one version beat or lost to another.
- If a script or comment names a best variant, treat it as a hypothesis to verify from eval outputs, not as unquestioned truth.

## Local Skills

Use the project skills under `.claude/skills/` when the task is specifically about:

- interpreting unlearning metrics
- editing or launching experiments safely
- recovering the current best-known version of a method
