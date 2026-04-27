# Skill: Collect Experiments / Paper Table Pipeline

## Purpose
Run all missing TOFU evals for paper methods and print the composite-score table.

## Baselines of interest (paper comparison set)

These are the eight methods that appear in the paper's experiment section.
All are normalized against **Baseline** (the full finetuned model).

| # | Label | Checkpoint type | Save location |
|---|---|---|---|
| 1 | **Baseline** | single (full model) | `saves/finetune/tofu_{model}_full/best` → eval in `saves/eval/tofu_{model}_full/evals_{split}/` |
| 2 | **Retrained** | single (retain model) | `saves/finetune/tofu_{model}_{retain_split}/best` → eval in `saves/eval/tofu_{model}_{retain_split}/` |
| 3 | **GradAscent** | single | `saves/unlearn/tofu_{model}_{split}_GradAscent/` |
| 4 | **GradDiff** | single | `saves/unlearn/tofu_{model}_{split}_GradDiff/` |
| 5 | **NPO** | single | `saves/unlearn/tofu_{model}_{split}_NPO/` |
| 6 | **RMU** | single | `saves/unlearn/tofu_{model}_{split}_RMU/` |
| 7 | **SimNPO** | single | `saves/unlearn/tofu_{model}_{split}_SimNPO/` |
| 8 | **Ours** | checkpoint-based | `saves/unlearn/tofu_{model}_{split}_LatentRMU_v4.8_sweep/checkpoint-*/` |

Splits evaluated: `forget01 / forget05 / forget10`.

## Scripts
- **`scripts/run_paper_evals.sh`** — checks every expected `TOFU_EVAL.json`, runs any that are missing, prints the console table, then patches `experiment.tex`.
- **`scripts/paper_table.py`** — reads all eval files, computes MemScore/Utility/Privacy, prints the paper table (console).
- **`scripts/update_paper_table.py`** — reads raw TOFU metrics (FQ, MU, PL, ES), generates LaTeX rows, and patches `../NeurIPS-26-LLM-Unlearning/secs/experiment.tex` in-place between sentinel comments.

## Metrics
See `scripts/eval_readme.md` for full definitions. Short form:
- **MemScore** = HM(1−ES_norm, 1−Para.Prob_norm, 1−TR_norm), normalized by Baseline. Higher = better forgetting.
- **Utility** = model_utility / model_utility_baseline. Higher = better retention.
- **Privacy** = sMIA score (1 = indistinguishable from retrained model).

## Usage

```bash
# Run missing evals, then print table
bash scripts/run_paper_evals.sh

# Print table only (evals already present)
/data/judy/conda/envs/unlearning/bin/python scripts/paper_table.py \
    --model Llama-3.2-1B-Instruct \
    --our_pattern "tofu_{model}_{split}_LatentRMU_v4.8_sweep"

# With per-component breakdown
python scripts/paper_table.py -v
```

## Config variables (edit in `run_paper_evals.sh`)
| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `Llama-3.2-1B-Instruct` | Model to run |
| `OUR_TASK_PATTERN` | `tofu_{MODEL}_{split}_LatentRMU_v4.8_sweep` | "Ours" task name template |
| `GPU` | `0` | CUDA device for eval |
| `PYTHON` | `/data/judy/conda/envs/unlearning/bin/python` | Python binary |

## Eval path conventions
| Method type | Eval JSON location |
|---|---|
| Baseline / Retrained (ref models) | `saves/eval/tofu_{model}_{split_or_full}/evals_{forget_split}/TOFU_EVAL.json` |
| Flat unlearn methods (GradAscent etc.) | `saves/unlearn/{task}/evals/TOFU_EVAL.json` |
| Checkpoint-based (LatentRMU = Ours) | `saves/unlearn/{task}/checkpoint-{N}/evals/TOFU_EVAL.json` |

For checkpoint-based runs, `paper_table.py` picks the best checkpoint by MemScore (excluding checkpoint-0).

## Paper update step

`update_paper_table.py` patches `../NeurIPS-26-LLM-Unlearning/secs/experiment.tex` using sentinel comments:

```latex
% AUTOGEN:tofu_main:begin — do not edit by hand; run scripts/update_paper_table.py
  Retain (upper)     & 1.0000 & 54.64 & 85.71 & 0.0291 & ...
  ...
% AUTOGEN:tofu_main:end
```

On first run the sentinels are inserted automatically by finding the `tab:tofu_main` label.
After that, re-runs replace only the region between the sentinels.

Metrics written to the paper (raw, not normalized):
| Column | JSON field | Format |
|---|---|---|
| FQ | `forget_quality` | 4 decimals |
| MU | `model_utility` | ×100, 2 decimals |
| PL | `privleak` | 2 decimals (already %) |
| ES | `extraction_strength` | 4 decimals |

Methods with no eval set (UNDIAL, APO, LUNAR) stay as `---`.

Dry-run (print rows without writing):
```bash
python scripts/update_paper_table.py --dry_run
```

## Notes
- Retrained eval is produced by `scripts/tofu_finetune.sh`; `run_paper_evals.sh` warns but does not re-run it.
- Baseline methods need checkpoint weights in their save dir (not just `evals/` or `logs/`); the script warns and skips if weights are absent.
- `forget_quality` from the raw TOFU eval is **not** used here; the paper uses the normalized MemScore defined above.
