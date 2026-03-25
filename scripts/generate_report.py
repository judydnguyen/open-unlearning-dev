#!/usr/bin/env python
"""Generate a Markdown report for a LatentUnlearning experiment.

Usage:
    python scripts/generate_report.py <experiment_dir> [--baselines <dir1> <dir2> ...]
    python scripts/generate_report.py saves/unlearn/latent_per_sample_v1.2
    python scripts/generate_report.py saves/unlearn/latent_per_sample_v1.2 --baselines saves/unlearn/rmu_baseline_compare saves/unlearn/latent_warmup_v1

The script reads TOFU_SUMMARY.json / TOFU_EVAL.json from each checkpoint
and produces a report.md in the experiment directory.
"""

import argparse
import json
import os
import re
import glob
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


SUMMARY_METRICS = [
    "model_utility",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "extraction_strength",
    "privleak",
]

GENERATION_KEYS = {
    "forget_Q_A_ROUGE": "Forget Set",
    "retain_Q_A_ROUGE": "Retain Set",
    "ra_Q_A_ROUGE": "Real Authors",
    "wf_Q_A_ROUGE": "World Facts",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_checkpoints(exp_dir):
    """Find all checkpoint dirs sorted by step number."""
    pattern = os.path.join(exp_dir, "checkpoint-*", "evals", "TOFU_SUMMARY.json")
    paths = glob.glob(pattern)
    results = []
    for p in paths:
        match = re.search(r"checkpoint-(\d+)", p)
        if match:
            step = int(match.group(1))
            results.append((step, p))
    results.sort(key=lambda x: x[0])
    return results


def find_best_checkpoint(checkpoints):
    """Find checkpoint with best model_utility among those with forget_truth_ratio > 0.8."""
    best = None
    best_utility = -1
    for step, path in checkpoints:
        data = load_json(path)
        ftr = data.get("forget_truth_ratio", 0)
        mu = data.get("model_utility", 0)
        if ftr > 0.8 and mu > best_utility:
            best_utility = mu
            best = (step, data)
    # Fallback: just pick highest utility
    if best is None:
        for step, path in checkpoints:
            data = load_json(path)
            mu = data.get("model_utility", 0)
            if mu > best_utility:
                best_utility = mu
                best = (step, data)
    return best


def get_final_checkpoint(checkpoints):
    """Return (step, data) for the last checkpoint."""
    step, path = checkpoints[-1]
    return step, load_json(path)


def get_baseline(checkpoints):
    """Return (step, data) for checkpoint-0 (pre-unlearning baseline)."""
    for step, path in checkpoints:
        if step == 0:
            return step, load_json(path)
    return None, None


def load_overrides(exp_dir):
    """Load hydra overrides if available."""
    path = os.path.join(exp_dir, ".hydra", "overrides.yaml")
    if not os.path.exists(path):
        return {}
    overrides = {}
    with open(path) as f:
        for line in f:
            line = line.strip().lstrip("- ")
            if "=" in line:
                k, v = line.split("=", 1)
                overrides[k] = v
    return overrides


def parse_training_log(exp_dir):
    """Extract phase 2 loss entries from LatentUnlearning.log."""
    log_path = os.path.join(exp_dir, "LatentUnlearning.log")
    if not os.path.exists(log_path):
        return []
    entries = []
    seen_steps = set()
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r"Phase 2.*step=(\d+), forget_loss=([\d.]+), retain_loss=([\d.]+), forget_weight=([\d.]+)",
                line,
            )
            if m:
                step = int(m.group(1))
                if step not in seen_steps:
                    seen_steps.add(step)
                    entries.append({
                        "step": step,
                        "forget_loss": float(m.group(2)),
                        "retain_loss": float(m.group(3)),
                        "forget_weight": float(m.group(4)),
                    })
    return entries


def load_eval_json_for_dir(exp_dir):
    """Load TOFU_EVAL.json from the final checkpoint of an experiment directory."""
    checkpoints = find_checkpoints(exp_dir)
    if not checkpoints:
        return None
    step, _ = get_final_checkpoint(checkpoints)
    eval_path = os.path.join(exp_dir, f"checkpoint-{step}", "evals", "TOFU_EVAL.json")
    if not os.path.exists(eval_path):
        return None
    return load_json(eval_path)


def build_response_comparison_table(methods_evals, key="forget_Q_A_ROUGE", n=10, max_len=150):
    """Build a markdown table comparing generations for the same forget prompts across methods.

    Args:
        methods_evals: list of (method_name, eval_json_dict)
    """
    # Collect per-index data: question/ground_truth + each method's generation
    all_data = {}  # str(idx) -> {"input": ..., "ground_truth": ..., method_name: generation}
    method_names = [name for name, _ in methods_evals]

    for method_name, eval_json in methods_evals:
        if key not in eval_json:
            continue
        entries = eval_json[key].get("value_by_index", {})
        for idx, entry in entries.items():
            if "generation" not in entry:
                continue
            if idx not in all_data:
                all_data[idx] = {
                    "input": entry.get("input", ""),
                    "ground_truth": entry.get("ground_truth", "").strip(),
                }
            gen = entry.get("generation", "").strip().replace("\n", " ")
            if len(gen) > max_len:
                gen = gen[:max_len] + "…"
            all_data[idx][method_name] = gen

    if not all_data:
        return []

    def cell(s):
        return str(s).replace("|", "\\|")

    lines = []
    col_header = " | ".join(cell(n) for n in method_names)
    lines.append(f"| # | Question | Ground Truth | {col_header} |")
    col_sep = " | ".join("---" for _ in method_names)
    lines.append(f"|---|---|---| {col_sep} |")

    for idx in sorted(all_data.keys(), key=lambda x: int(x))[:n]:
        d = all_data[idx]
        q = cell(extract_question(d["input"])[:120])
        gt = cell(d["ground_truth"][:max_len].replace("\n", " "))
        gens = " | ".join(cell(d.get(mn, "—")) for mn in method_names)
        lines.append(f"| {idx} | {q} | {gt} | {gens} |")

    return lines


def collect_generations(eval_json, key, n=10):
    """Collect up to n generation examples from TOFU_EVAL.json for a given metric key."""
    if key not in eval_json:
        return []
    entries = eval_json[key].get("value_by_index", {})
    examples = []
    for idx in sorted(entries.keys(), key=lambda x: int(x)):
        entry = entries[idx]
        if "generation" in entry:
            examples.append({
                "index": idx,
                "input": entry.get("input", "").strip(),
                "ground_truth": entry.get("ground_truth", "").strip(),
                "generation": entry.get("generation", "").strip(),
                "rouge_l": entry.get("rougeL_recall", None),
            })
        if len(examples) >= n:
            break
    return examples


def extract_question(input_text):
    """Extract just the question from the chat template input."""
    # Find the last "user\n" block
    parts = input_text.split("user\n")
    if len(parts) > 1:
        q = parts[-1].split("assistant")[0].strip()
        return q
    return input_text[-200:]


def fmt(val, precision=4):
    if val is None:
        return "—"
    if isinstance(val, float):
        if abs(val) < 1e-10:
            return f"{val:.2e}"
        return f"{val:.{precision}f}"
    return str(val)


PLOT_METRICS = [
    ("model_utility",      "Model Utility",      "tab:blue"),
    ("forget_truth_ratio", "Forget Truth Ratio",  "tab:orange"),
    ("forget_Q_A_Prob",    "Forget Q/A Prob",     "tab:red"),
    ("forget_quality",     "Forget Quality",      "tab:green"),
]


def _plot_metrics(checkpoints, exp_dir, best_step=None):
    """Plot TOFU_SUMMARY metrics over training steps. Returns saved path or None."""
    if not _MPL or not checkpoints:
        return None

    steps = []
    series = {key: [] for key, _, _ in PLOT_METRICS}

    for step, path in checkpoints:
        data = load_json(path)
        steps.append(step)
        for key, _, _ in PLOT_METRICS:
            series[key].append(data.get(key))

    fig, ax = plt.subplots(figsize=(9, 4))
    for key, label, color in PLOT_METRICS:
        vals = series[key]
        if any(v is not None for v in vals):
            ax.plot(steps, vals, label=label, color=color, marker="o", markersize=3)

    if best_step is not None:
        ax.axvline(best_step, color="gray", linestyle="--", linewidth=0.8, label=f"best (step {best_step})")

    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_title("TOFU Metrics Over Training")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(exp_dir, "metrics_plot.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def generate_report(exp_dir, baseline_dirs=None):
    checkpoints = find_checkpoints(exp_dir)
    if not checkpoints:
        print(f"No checkpoints found in {exp_dir}")
        return

    exp_name = os.path.basename(os.path.normpath(exp_dir))
    overrides = load_overrides(exp_dir)
    baseline_step, baseline_data = get_baseline(checkpoints)
    final_step, final_data = get_final_checkpoint(checkpoints)
    best = find_best_checkpoint(checkpoints)
    best_step, best_data = best if best else (final_step, final_data)
    log_entries = parse_training_log(exp_dir)

    lines = []
    w = lines.append

    # ---- Header ----
    w(f"# Experiment Report: `{exp_name}`\n")

    # ---- Config ----
    if overrides:
        model = overrides.get("model", "—")
        forget_split = overrides.get("forget_split", "—")
        retain_split = overrides.get("retain_split", "—")
        w(f"**Model**: {model} | **Forget**: {forget_split} | **Retain**: {retain_split}\n")

        w("## Configuration\n")
        w("| Parameter | Value |")
        w("|---|---|")
        for k, v in sorted(overrides.items()):
            if k in ("experiment", "task_name", "model", "forget_split", "retain_split"):
                continue
            short_k = k.replace("trainer.method_args.", "").replace("trainer.args.", "").replace("model.model_args.", "model.")
            w(f"| {short_k} | {v} |")
        w("")

    # ---- Results Comparison ----
    w("## Results\n")

    # Collect all methods to compare
    methods = []

    if baseline_data:
        methods.append(("Baseline (no unlearning)", baseline_data))

    if baseline_dirs:
        for bd in baseline_dirs:
            bd_name = os.path.basename(os.path.normpath(bd))
            bd_ckpts = find_checkpoints(bd)
            if bd_ckpts:
                bd_best = find_best_checkpoint(bd_ckpts)
                if bd_best:
                    methods.append((bd_name, bd_best[1]))

    methods.append((f"**{exp_name}** (best, step {best_step})", best_data))
    if best_step != final_step:
        methods.append((f"{exp_name} (final, step {final_step})", final_data))

    header_metrics = ["model_utility", "forget_truth_ratio", "forget_Q_A_Prob", "extraction_strength"]
    w("| Method | model_utility | forget_truth_ratio | forget_Q_A_Prob | extraction_strength |")
    w("|---|---|---|---|---|")
    for name, data in methods:
        row = " | ".join(fmt(data.get(m)) for m in header_metrics)
        w(f"| {name} | {row} |")
    w("")

    # Utility retention
    if baseline_data:
        base_util = baseline_data.get("model_utility", 1)
        if base_util > 0:
            w("### Utility Retention\n")
            for name, data in methods:
                if "Baseline" in name:
                    continue
                pct = data.get("model_utility", 0) / base_util * 100
                w(f"- {name}: **{pct:.1f}%**")
            w("")

    # ---- Training Trajectory ----
    w("## Training Trajectory\n")
    w("| Step | model_utility | forget_truth_ratio | forget_Q_A_Prob |")
    w("|---|---|---|---|")
    for step, path in checkpoints:
        data = load_json(path)
        mu = fmt(data.get("model_utility"))
        ftr = fmt(data.get("forget_truth_ratio"))
        fqa = fmt(data.get("forget_Q_A_Prob"))
        marker = " *" if step == best_step else ""
        w(f"| {step}{marker} | {mu} | {ftr} | {fqa} |")
    w("")

    # ---- Phase 2 Loss ----
    if log_entries:
        w("## Phase 2 Loss Dynamics\n")
        w("| Step | forget_loss | retain_loss | forget_weight |")
        w("|---|---|---|---|")
        for e in log_entries:
            w(f"| {e['step']} | {e['forget_loss']:.4f} | {e['retain_loss']:.4f} | {e['forget_weight']:.4f} |")
        w("")

    # ---- Generations ----
    # Load the best checkpoint's TOFU_EVAL.json for generation examples
    best_eval_path = os.path.join(exp_dir, f"checkpoint-{best_step}", "evals", "TOFU_EVAL.json")
    if os.path.exists(best_eval_path):
        eval_data = load_json(best_eval_path)

        w("## Generation Examples\n")
        for key, label in GENERATION_KEYS.items():
            examples = collect_generations(eval_data, key, n=10)
            if not examples:
                continue
            w(f"### {label} (`{key}`)\n")
            for ex in examples:
                q = extract_question(ex["input"])
                w(f"**Q{ex['index']}**: {q}")
                w(f"- **Ground truth**: {ex['ground_truth']}")
                gen_text = ex["generation"][:300]
                if len(ex["generation"]) > 300:
                    gen_text += "..."
                w(f"- **Generation**: {gen_text}")
                if ex["rouge_l"] is not None:
                    w(f"- ROUGE-L recall: {ex['rouge_l']:.3f}")
                w("")
    w("")

    # ---- Metrics plot ----
    plot_path = _plot_metrics(checkpoints, exp_dir, best_step)
    if plot_path:
        w("## Metrics Over Training\n")
        w(f"![Metrics]({os.path.basename(plot_path)})\n")

    # ---- Latent vector plots ----
    plots = []
    for name in ["latent_vectors_after_phase1.png", "latent_vectors_after_phase2.png"]:
        if os.path.exists(os.path.join(exp_dir, name)):
            plots.append(name)
    if plots:
        w("## Latent Vector Visualizations\n")
        for p in plots:
            label = "After Phase 1" if "phase1" in p else "After Phase 2"
            w(f"### {label}")
            w(f"![{label}]({p})\n")

    # ---- Response Comparison Table ----
    methods_evals = []

    # Baseline (checkpoint-0 of main experiment)
    baseline_eval_path = os.path.join(exp_dir, "checkpoint-0", "evals", "TOFU_EVAL.json")
    if os.path.exists(baseline_eval_path):
        methods_evals.append(("Baseline", load_json(baseline_eval_path)))

    # External baselines
    if baseline_dirs:
        for bd in baseline_dirs:
            bd_eval = load_eval_json_for_dir(bd)
            if bd_eval:
                methods_evals.append((os.path.basename(os.path.normpath(bd)), bd_eval))

    # Main experiment (final checkpoint)
    final_eval_path = os.path.join(exp_dir, f"checkpoint-{final_step}", "evals", "TOFU_EVAL.json")
    main_eval = load_json(final_eval_path) if os.path.exists(final_eval_path) else None
    if main_eval is not None:
        methods_evals.append((f"{exp_name} (final)", main_eval))

    if len(methods_evals) > 1:
        w("## Forget Set Response Comparison\n")
        table = build_response_comparison_table(methods_evals, key="forget_Q_A_ROUGE", n=10)
        for line in table:
            w(line)
        w("")

    # ---- Files ----
    w("## Files\n")
    w(f"- Experiment dir: `{exp_dir}`")
    w(f"- Best checkpoint: `{exp_dir}/checkpoint-{best_step}/`")
    if os.path.exists(os.path.join(exp_dir, "LatentUnlearning.log")):
        w(f"- Training log: `{exp_dir}/LatentUnlearning.log`")
    w("")

    report = "\n".join(lines)
    out_path = os.path.join(exp_dir, "report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Report written to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("exp_dir", help="Path to experiment directory")
    parser.add_argument("--baselines", nargs="*", default=[], help="Baseline experiment dirs to compare")
    args = parser.parse_args()
    generate_report(args.exp_dir, args.baselines)


if __name__ == "__main__":
    main()
