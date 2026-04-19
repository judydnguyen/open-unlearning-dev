# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import glob
import json
import os
import re
import shutil
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
        self._best_forget_quality = -float("inf")
        self._custom_eval_skipped_warned = False
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
            if self.accelerator.num_processes > 1:
                if not self._custom_eval_skipped_warned:
                    logger.warning(
                        "Custom evaluator requires a single accelerator process; "
                        "skipping in-training eval (num_processes=%d). "
                        "Run src/eval.py after training for full metrics.",
                        self.accelerator.num_processes,
                    )
                    self._custom_eval_skipped_warned = True
                return {}
            if self.accelerator.is_local_main_process:
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
                mu = eval_metrics.get("model_utility")
                ner = eval_metrics.get("forget_Q_A_NER")
                if mu is not None and ner is not None:
                    joint_score = (mu + (1.0 - ner)) / 2.0
                else:
                    joint_score = None
                if joint_score is not None and joint_score > self._best_forget_quality:
                    self._best_forget_quality = joint_score
                    best_dir = os.path.join(run_dir, "best")
                    self.save_model(best_dir)
                    with open(os.path.join(best_dir, "best_step.json"), "w") as _f:
                        json.dump({
                            "step": self.state.global_step,
                            "model_utility": mu,
                            "forget_Q_A_NER": ner,
                            "joint_score": joint_score,
                        }, _f, indent=2)
                    logger.info(
                        "New best joint_score=%.4f (model_utility=%.4f, forget_Q_A_NER=%.4f) at step %d → saved to %s",
                        joint_score, mu, ner, self.state.global_step, best_dir,
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        if self.accelerator.is_local_main_process:
            run_dir = self._get_output_dir(trial=None)
            last_dir = os.path.join(run_dir, "last")
            # Remove stale evals before saving so they can't outlive the model
            stale_evals = os.path.join(last_dir, "evals")
            if os.path.exists(stale_evals):
                shutil.rmtree(stale_evals)
            self.save_model(last_dir)
            with open(os.path.join(last_dir, "last_step.json"), "w") as _f:
                json.dump({"step": self.state.global_step}, _f, indent=2)
            logger.info(
                "Final model at step %d → saved to %s",
                self.state.global_step, last_dir,
            )
        return result
