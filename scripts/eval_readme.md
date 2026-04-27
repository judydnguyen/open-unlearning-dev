# Report Metrics: Memorization, Privacy, Utility

## Overview

`compute_report_metrics.py` computes three composite scores from existing
`TOFU_EVAL.json` files under `saves/unlearn/`. All metrics are normalized by
**checkpoint-0** (the init finetuned model before any unlearning), so scores
fall in [0, 1].

| Score | Formula | Higher = |
|-------|---------|----------|
| **Memorization Score** | HM(1−ES_norm, 1−EM_norm, 1−Para.Prob_norm, 1−TR_norm) | Better forgetting |
| **Utility** | model_utility / model_utility_init | Better retention |
| **Privacy Score** | `max(0, 1 − \|AUC−AUC_retain\| / max(AUC_retain, 1−AUC_retain))` = sMIA | Closer to retain AUC |

### Component Definitions

| Symbol | Source field | Direction before inversion |
|--------|-------------|---------------------------|
| ES | `extraction_strength` | Higher = more verbatim memorization |
| EM | `exact_memorization` | Higher = more exact memorization |
| Para.Prob | `forget_Q_A_PARA_Prob` | Higher = model assigns higher prob to forget set |
| TR | `forget_truth_ratio` (`closer_to_1_better`) | Lower = more confident = more memorized |

**Truth Ratio normalization note:** Because `forget_truth_ratio` uses the
`closer_to_1_better` aggregator (higher TR = more forgotten), its memorization
signal is `(1 − TR)`. The script normalizes this as `(1−TR) / (1−TR_init)` so
the direction aligns with the other components.

## Setup

Requires the `unlearning` conda environment (scipy, numpy):

```bash
conda activate unlearning
```

## Usage

```bash
# All experiments — pick best checkpoint per run by Memorization Score
python scripts/compute_report_metrics.py

# Show per-component breakdown
python scripts/compute_report_metrics.py -v

# Specific runs only
python scripts/compute_report_metrics.py \
    --runs saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_LatentRMU_v4.8

# Select best checkpoint by utility instead of mem_score
python scripts/compute_report_metrics.py --select_by utility

# List eval commands needed to add exact_memorization to existing checkpoints
python scripts/compute_report_metrics.py --missing_em_only
```

## Adding `exact_memorization` (EM)

EM is not computed in existing unlearn checkpoint evals (it was disabled to
save time). The Memorization Score falls back to a 3-component HM and is
marked with `*` in the output.

**Step 1** — `exact_memorization` is already enabled in `configs/eval/tofu.yaml`.

**Step 2** — Generate and run re-eval commands:

```bash
# Generate the shell script (works without the unlearning env)
python scripts/compute_report_metrics.py --missing_em_only > scripts/reeval_for_em.sh

# Review and run (long — ~5–10 min per checkpoint)
conda activate unlearning
bash scripts/reeval_for_em.sh
```

`reeval_for_em.sh` is pre-generated and covers all existing checkpoints under
`saves/unlearn/`. Each command passes `eval.tofu.overwrite=true` to update the
existing `TOFU_EVAL.json` in place.

## Output Example

```
Experiment                                                  Ckpt |  MemScore Comp |  Utility |  Privleak
--------------------------------------------------------------------------------------------------------
LatentRMU_v4.8 | forget01 | Llama-3.2-1B-Instruct            120 |    0.5427 *    3 |   0.9809 |     55.53
LatentRMU_v4.8 | forget05 | Llama-3.2-1B-Instruct            125 |    0.4276 *    3 |   0.9884 |     33.97

* = exact_memorization missing; Mem Score uses 3-component HM.
```

## Sanity Checks

At **checkpoint-0** (init model):
- All normalized components = 1.0 → all (1 − norm) = 0.0 → Memorization Score ≈ 0 ✓
- Utility = model_utility / model_utility_init = 1.0 ✓

As unlearning progresses, expect:
- Memorization Score to increase (more forgetting)
- Utility to stay near 1.0 (knowledge retained)
- Privleak to stay negative or near zero (no additional privacy leakage)
