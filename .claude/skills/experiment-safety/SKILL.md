# Skill: Experiment Safety and Change Boundaries

Use this skill when editing trainers, configs, scripts, or evaluation wiring in this repository.

## Goal

Make research progress without losing reproducibility or silently breaking benchmark meaning.

## First Checks

Before changing code:

1. Identify the active method and benchmark.
2. Read the matching files in:
   - `configs/trainer/`
   - `configs/experiment/`
   - `src/trainer/unlearn/`
   - relevant scripts under `scripts/`
3. Check whether there is already a smoke script or test covering the behavior.

## Safe Default Strategy

Prefer this order:

1. hyperparameter override
2. new config variant
3. small trainer change
4. new script for a controlled ablation
5. benchmark or metric changes only if the research question truly requires it

## Usually Allowed

- tune method hyperparameters
- add logging or diagnostics
- add a focused unit test
- add a new trainer config
- add a new experiment script
- add a new optional reward or loss component behind a config flag

## Requires Extra Care

- changing metric formulas in `src/evals/metrics/`
- changing benchmark aggregation
- changing dataset split semantics
- changing output JSON structure
- reusing an old `task_name` for a new run
- overwriting a checkpoint that was previously treated as a baseline

## Naming and Artifact Rules

- Give each meaningful run a fresh `task_name`.
- Keep the command or script that produced a result.
- Save eval outputs inside the run directory when possible.
- If the run is meant to challenge the current best, say that explicitly in the script or notes.

## Validation Rules

After a code change, do at least one:

- run the relevant unit test
- run a smoke script
- run a short train + eval cycle on the intended benchmark

If validation is skipped, say so clearly and explain why.

## Non-Goals

Do not:

- silently redefine what a benchmark score means
- replace an existing baseline config without preserving a way to reproduce it
- claim a change is better from training loss alone
- assume the latest script is the best script
