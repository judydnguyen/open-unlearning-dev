# Skill: Research Iteration

Use this skill when improving an existing unlearning method through a careful, minimal experiment loop.

## Goal

Run one bounded research iteration:

1. inspect the current method and nearby baseline
2. propose a small change
3. criticize the proposal before acting
4. run the experiment with a fresh task prefix
5. wait for the run to finish
6. analyze the results against the baseline

This skill is for local improvement, not invention of a new method family.

## When To Use

Use this when:

- there is already a working script or launch command
- the method is already implemented
- the next step is to test one small idea
- the user wants iterative research help rather than a broad redesign

Do not use this skill to propose a completely new algorithm, benchmark, or metric.

## Core Rules

- Always prefer minimal changes.
- Change one important axis at a time.
- Reuse the existing script or command whenever possible.
- Never overwrite the current baseline checkpoint.
- Never claim success from training loss alone.
- Always compare against the closest valid baseline.

## Allowed Changes

Good default changes:

- one hyperparameter adjustment
- one reward-weight adjustment
- one retain-vs-forget balance adjustment
- one warmup or schedule adjustment
- one small local implementation fix within the same method
- one optional feature toggle already supported by the code

Avoid by default:

- new trainer families
- new benchmark semantics
- new metric definitions
- new dataset split semantics
- large refactors
- multi-change experiments that confound attribution

## Research Loop

### 1. Inspect

Before proposing anything:

- read the active script, config, and recent comments
- identify the current baseline task name
- identify which metrics matter for this benchmark
- identify the likely failure mode of the baseline

### 2. Propose

Generate 1 to 3 small candidate changes.

For each candidate, state:

- the exact change
- why it might help
- what could go wrong
- which metrics should improve if the idea is correct

Then select exactly one candidate to test.

### 3. Self-Critique

Before launching the run, criticize the selected idea.

Check for:

- utility-collapse risk
- over-forgetting by generic refusal
- mismatch with prior evidence in script comments
- confounding with another change
- invalid comparison because of model, split, or `retain_logs_path` mismatch

If the criticism is strong enough, reject the idea and choose a safer one.

### 4. Launch

Use the existing script whenever possible.

Run rules:

- preserve all unchanged arguments
- give the new run a fresh prefixed `task_name`
- do not reuse the prior baseline task name
- keep the run traceable to its parent baseline

Suggested prefixes:

- `r1_`
- `r2_`
- `iter01_`
- `crit1_`

Preferred pattern:

- `<prefix><baseline_task_name>`

### 5. Wait

If the run is expected to finish in a reasonable time, wait for completion.

Monitor for:

- crash or config errors
- obvious degeneration in logs
- completion marker in the script output
- existence of the expected checkpoint or eval output

If the run fails, report the failure mode clearly before proposing another iteration.

### 6. Analyze

Compare the new run only against the closest valid baseline.

Summarize:

1. what changed
2. which benchmark and split were used
3. forgetting gains
4. utility or retain losses
5. whether the result is better, worse, or a tradeoff

Also state whether the result is:

- actionable
- inconclusive
- not comparable

## Output Template

For each iteration, produce:

1. baseline being modified
2. selected minimal change
3. self-critique of the change
4. exact command or script delta
5. new task name
6. run status
7. result comparison
8. next recommendation

## Stop Conditions

Stop the loop and ask for direction if:

- every plausible next change is high-risk
- the baseline evidence is ambiguous
- the run artifacts needed for comparison are missing
- the proposed next step would require changing method family or benchmark semantics

## Repo-Specific Notes

- treat script comments as useful evidence, not final truth
- prefer comparing eval outputs under `saves/unlearn/{task_name}/evals`
- benchmark comparisons are only valid when model, split, and reference logs align
- if a script already documents version-to-version tradeoffs, continue that lineage instead of creating a disconnected naming scheme
