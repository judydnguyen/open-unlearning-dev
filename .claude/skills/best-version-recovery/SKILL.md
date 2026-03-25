# Skill: Best-Version Recovery

Use this skill when asked questions like:

- "what is our current best version?"
- "how do we revert to the strongest method we had?"
- "which checkpoint/config should we go back to?"

## Goal

Recover the best-known method using evidence from code, scripts, comments, and saved artifacts.

## Evidence Order

Use this order of trust:

1. Saved eval outputs attached to a checkpoint
2. Script comments documenting why a variant won or lost
3. Named launch scripts under `scripts/`
4. Config defaults
5. Recent code edits without eval evidence

Do not equate "latest" with "best."

## Recovery Procedure

1. Identify candidate best runs.
   - inspect `scripts/` for named variants
   - inspect comments that compare versions
   - inspect saved eval directories under `saves/`

2. Verify what each candidate actually used.
   - trainer
   - model
   - splits
   - key Hydra overrides
   - `retain_logs_path`

3. Verify the evidence.
   - read the eval JSON
   - summarize the key forgetting and utility metrics
   - note any tradeoffs or missing pieces

4. Reconstruct the recoverable state.
   - code version in Git, if needed
   - launch script or full command
   - checkpoint path
   - eval path

## Rollback Rules

- Roll back code only if the implementation itself drifted away from the best-known method.
- If the best-known result came from a saved checkpoint, prefer reusing that checkpoint over trying to recreate it from memory.
- Do not delete newer runs during recovery.
- If evidence is ambiguous, present the top candidate runs and explain the uncertainty.

## Repo-Specific Hints

- `saves/unlearn/{task_name}` usually holds the checkpoint
- `saves/unlearn/{task_name}/evals` often holds the follow-up evaluation
- comments in scripts like `scripts/test_latent_rmu_nll_v5.sh` may encode why a variant was considered promising
- benchmark comparability depends on model, split, and reference logs matching
