# Skill: Repo Context

Use this skill when the task is to understand, explain, or quickly orient to this repository before making changes.

## Goal

Build a concise, project-specific mental model of the codebase so later reasoning stays grounded in how this repo actually works.

## What This Repo Is

This repository is a research framework for LLM unlearning.

It supports:

- multiple benchmarks such as TOFU, MUSE, and WMDP
- multiple unlearning methods under a shared training interface
- multiple evaluation metrics for forgetting, utility, memorization, and privacy
- Hydra-based experiment composition for train, unlearn, and eval workflows

The project is not just an implementation of one method. It is an experimentation framework where reproducibility and comparability matter.

## Core Layout

- `src/train.py`
  - main entrypoint for finetuning and unlearning training
- `src/eval.py`
  - main entrypoint for benchmark evaluation
- `src/trainer/unlearn/`
  - unlearning method implementations
- `src/evals/`
  - benchmark evaluator implementations
- `src/evals/metrics/`
  - metric implementations
- `configs/trainer/`
  - method-level configs and hyperparameters
- `configs/experiment/`
  - benchmark- and workflow-specific experiment presets
- `configs/eval/`
  - evaluator and metric bundles
- `scripts/`
  - reproducible launch points, smoke tests, and experiment lineage
- `saves/`
  - checkpoints and evaluation artifacts

## How Experiments Work

Most work follows this path:

1. choose a Hydra experiment preset
2. choose a model config
3. choose a trainer or method
4. override a few hyperparameters or paths
5. run training or unlearning
6. run evaluation
7. compare results against prior runs

Output directories usually follow:

- `saves/train/{task_name}`
- `saves/unlearn/{task_name}`
- `saves/eval/{task_name}`

## Important Repo Habits

- benchmark semantics should stay stable unless explicitly changing the benchmark
- comparisons are only meaningful when model, split, and reference logs line up
- `retain_logs_path` matters for several unlearning metrics
- a result is not trustworthy unless it can be traced to code, config, command, and artifacts
- script comments often contain useful local research history

## Active Research Style

This repo often evolves by:

- adding or tuning a method under `src/trainer/unlearn/`
- creating a new config variant
- running a controlled script
- comparing eval outputs to a nearby baseline

Good work here is usually incremental, evidence-driven, and reversible.

## How To Respond When Using This Skill

When asked for repo context:

- summarize the repo in a few sentences first
- then point to the most relevant files or directories
- explain the workflow in terms of train, unlearn, and eval
- mention the benchmark and metric context if relevant
- avoid generic boilerplate that could describe any ML repo

## Good Questions For This Skill

- "what does this repo do?"
- "where should I look to change an unlearning method?"
- "how are experiments organized here?"
- "what files matter for evaluation?"
- "what is the overall workflow of this project?"
