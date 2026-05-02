# Phase 1 Component Ablation Table

Metrics: **MU** (model_utility ↑) · **F-Prob** (forget_Q_A_Prob ↓) · **MIA** (mia_auc ↓)

| Variant | Orth | RetainSep | Anchor | MU | F-Prob | MIA | Split |
|---|:---:|:---:|:---:|---|---|---|---|
| **full** | ✓ | ✓ | ✓ | **0.5980** | 0.0453 | 0.1059 | forget05 |
| no\_orth | ✗ | ✓ | ✓ | 0.5863 | 0.0642 | 0.1418 | forget05 |
| no\_retain\_sep | ✓ | ✗ | ✓ | 0.5851 | 0.0390 | 0.0770 | forget05 † |
| no\_anchor | ✓ | ✓ | ✗ | 0.5908 | 0.1409 | 0.4313 | forget05 |
| orth\_only | ✓ | ✗ | ✗ | 0.5805 | 0.2983 | N/A | forget01 |
| retain\_sep\_only | ✗ | ✓ | ✗ | 0.5804 | 0.2983 | N/A | forget01 |
| anchor\_only | ✗ | ✗ | ✓ | 0.5851 | 0.0390 | 0.0770 | forget05 † |
| no\_phase1 | ✗ | ✗ | ✗ | 0.5838 | 0.1134 | N/A | forget01 |

† `no_retain_sep` and `anchor_only` are identical in the forget05 CSV — likely a data collection bug. Needs re-run.

Missing forget05 MIA for `orth_only`, `retain_sep_only`, `no_phase1` — run:
```bash
bash scripts/tofu_ablation_study.sh forget05
```
