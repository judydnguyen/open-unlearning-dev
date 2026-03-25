# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import glob
import json
import os
import re
import logging
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

logger = logging.getLogger(__name__)

_PLOT_METRICS = [
    ("model_utility",      "Model Utility",     "tab:blue"),
    ("forget_truth_ratio", "Forget Truth Ratio", "tab:orange"),
    ("forget_Q_A_Prob",    "Forget Q/A Prob",    "tab:red"),
    # ("forget_quality",     "Forget Quality",     "tab:green"),
    ("extraction_strength", "Extraction Strength", "tab:purple"),
    ("privleak",          "Privacy Leakage",    "tab:brown"),
]


def _refresh_metrics_plot(run_dir: str) -> None:
    """Scan all checkpoint-*/evals/TOFU_SUMMARY.json and save metrics_plot.png.

    Metrics are normalized to 0-100 relative to checkpoint-0 (pre-unlearning baseline).
    """
    if not _MPL:
        return
    paths = glob.glob(os.path.join(run_dir, "checkpoint-*", "evals", "TOFU_SUMMARY.json"))
    if not paths:
        return

    records = []
    for p in paths:
        m = re.search(r"checkpoint-(\d+)", p)
        if not m:
            continue
        with open(p) as f:
            data = json.load(f)
        records.append((int(m.group(1)), data))
    records.sort(key=lambda x: x[0])

    # Use checkpoint-0 as the baseline for normalization (pre-unlearning values = 100)
    baseline = records[0][1] if records[0][0] == 0 else {}

    steps = [r[0] for r in records]
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, label, color in _PLOT_METRICS:
        vals = [r[1].get(key) for r in records]
        if not any(v is not None for v in vals):
            continue
        base = baseline.get(key)
        if base and base != 0:
            vals = [v / base * 100 if v is not None else None for v in vals]
        ax.plot(steps, vals, label=label, color=color, marker="o", markersize=3)

    ax.set_xlabel("Step")
    ax.set_ylabel("Score (% of pre-unlearning baseline)")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("TOFU Metrics Over Training (normalized to baseline)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "metrics_plot_by_step.png"), dpi=120)
    plt.close(fig)


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators:
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_metrics = {}
                    for _, evaluator in self.evaluators.items():
                        eval_args = {
                            "output_dir": output_dir,
                            "template_args": self.template_args,
                            "model": self.model,
                            "tokenizer": self.tokenizer,
                        }
                        eval_metrics.update(evaluator.evaluate(**eval_args))
                    self.log(eval_metrics)
                    _refresh_metrics_plot(run_dir)
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
