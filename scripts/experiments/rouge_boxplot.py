"""Per-example ROUGE boxplot comparison across unlearning methods.

For each method, reads `value_by_index` from the appropriate TOFU_EVAL.json field
(forget_Q_A_ROUGE for the forget set, retain_Q_A_ROUGE for the retain set), then
plots a 3-panel boxplot: ROUGE-1 Recall, ROUGE-L Recall, ROUGE-L F1.

Output files (one per metric set):
  analysis_out/rouge_boxplot_<split>_forget.png
  analysis_out/rouge_boxplot_<split>_retain.png

Usage:
  python scripts/experiments/rouge_boxplot.py --split forget01
  python scripts/experiments/rouge_boxplot.py --split forget10
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("rouge1_recall", "ROUGE-1 Recall"),
    ("rougeL_recall", "ROUGE-L Recall"),
    ("rougeL_f1",     "ROUGE-L F1"),
]

PALETTE = {
    "Retrained":  "#1a9641",   # oracle green
    "GradAscent": "#aec7e8",
    "GradDiff":   "#ffbb78",
    "NPO":        "#c5b0d5",
    "RMU":        "#98df8a",
    "SimNPO":     "#c49c94",
    "Ours":       "#e91e63",   # FLOUR pink (matches paper)
}


def load_per_sample(eval_path, eval_key):
    """Return a list of per-sample metric dicts from TOFU_EVAL.json[eval_key].value_by_index."""
    with open(eval_path) as f:
        data = json.load(f)
    if eval_key not in data:
        return None
    entry = data[eval_key]
    if not isinstance(entry, dict) or "value_by_index" not in entry:
        return None
    by_index = entry["value_by_index"]
    return [by_index[str(i)] for i in sorted([int(k) for k in by_index.keys()])]


def plot_rouge(method_rouge, metrics, palette, out_path, title):
    """Single-set boxplot (kept for backward compat). Renders 1×N panels."""
    names = list(method_rouge.keys())
    n_methods, n_metrics = len(names), len(metrics)
    fig, axes = plt.subplots(1, n_metrics,
                             figsize=(max(12, 3.5 * n_methods), 5),
                             sharey=False)
    if n_metrics == 1: axes = [axes]
    for ax, (metric_key, metric_label) in zip(axes, metrics):
        data        = [[r[metric_key] for r in method_rouge[n]] for n in names]
        colors      = [palette.get(n, "#bbbbbb") for n in names]
        edge_colors = ["#111111" if n == "Ours" else "#333333" for n in names]
        lws         = [2.0 if n == "Ours" else 0.8 for n in names]

        bp = ax.boxplot(
            data, patch_artist=True, widths=0.55,
            medianprops=dict(color="black", linewidth=1.8),
            flierprops=dict(marker="o", markersize=3, alpha=0.4,
                            markerfacecolor="black", markeredgecolor="black"),
            whiskerprops=dict(color="black", linewidth=0.9),
            capprops=dict(color="black", linewidth=0.9),
        )
        for patch, c, ec, lw in zip(bp["boxes"], colors, edge_colors, lws):
            patch.set_facecolor(c); patch.set_alpha(0.85)
            patch.set_edgecolor(ec); patch.set_linewidth(lw)
        ax.set_xticks(range(1, n_methods + 1))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=12)
        ax.set_title(metric_label, fontsize=14, pad=6)
        ax.set_ylabel("Score", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=600, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_rouge_tradeoff(method_rouge_forget, method_rouge_retain, metrics,
                         palette, out_path, title, oracle_name="Retrained"):
    """Scatter tradeoff — one point per method per metric.
       x = median retain ROUGE (utility, ↑ better)
       y = median forget ROUGE (forgetting, ↓ better)
       error bars = IQR  (25th–75th percentile)
       Oracle (Retrained) gets a star marker; FLOUR gets a thick black-outlined circle.
    """
    seen, names = set(), []
    for n in list(method_rouge_forget.keys()) + list(method_rouge_retain.keys()):
        if n not in seen and n in method_rouge_forget and n in method_rouge_retain:
            seen.add(n); names.append(n)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics,
                             figsize=(5.4 * n_metrics, 5.5),
                             sharey=False)
    if n_metrics == 1: axes = [axes]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        # Reference oracle position (used to draw guide lines)
        ox = oy = None
        if oracle_name in method_rouge_retain:
            ox = float(np.median([r[metric_key] for r in method_rouge_retain[oracle_name]]))
            oy = float(np.median([r[metric_key] for r in method_rouge_forget[oracle_name]]))

        for n in names:
            x_vals = np.array([r[metric_key] for r in method_rouge_retain[n]])
            y_vals = np.array([r[metric_key] for r in method_rouge_forget[n]])
            x_med, y_med = np.median(x_vals), np.median(y_vals)
            x_lo, x_hi = np.percentile(x_vals, 25), np.percentile(x_vals, 75)
            y_lo, y_hi = np.percentile(y_vals, 25), np.percentile(y_vals, 75)
            color = palette.get(n, "#bbbbbb")
            is_oracle = (n == oracle_name)
            is_flour = (n == "Ours")

            # Error bars (IQR)
            ax.errorbar(x_med, y_med,
                        xerr=[[x_med - x_lo], [x_hi - x_med]],
                        yerr=[[y_med - y_lo], [y_hi - y_med]],
                        fmt="none", ecolor=color, alpha=0.55,
                        capsize=3, capthick=1.0, elinewidth=1.2, zorder=2)
            # Marker
            marker = "*" if is_oracle else ("o" if not is_flour else "o")
            size = 280 if is_oracle else (200 if is_flour else 130)
            edge_color = "#111" if (is_oracle or is_flour) else "#444"
            edge_w = 1.8 if (is_oracle or is_flour) else 0.9
            ax.scatter(x_med, y_med, marker=marker, s=size, color=color,
                       edgecolors=edge_color, linewidths=edge_w, zorder=4)
            # Label
            ax.annotate(f"  {n}", xy=(x_med, y_med), fontsize=10,
                        color="#111", weight="bold" if is_flour else "normal", zorder=5)

        # Oracle guide lines
        if ox is not None:
            ax.axhline(oy, color="#666", linestyle=":", linewidth=1, alpha=0.7,
                       label=f"oracle forget ROUGE = {oy:.2f}")
            ax.axvline(ox, color="#666", linestyle=":", linewidth=1, alpha=0.7,
                       label=f"oracle retain ROUGE = {ox:.2f}")

        # Quadrant shading: ideal is HIGH retain (right) + LOW forget (bottom)
        ax.axhspan(0, oy if oy is not None else 0.0, alpha=0.05, color="green", zorder=0)
        ax.axvspan(ox if ox is not None else 1.0, 1.05, alpha=0.05, color="green", zorder=0)
        # bottom-right corner = ideal quadrant (oracle-like)
        if ox is not None and oy is not None:
            ax.annotate("oracle-like\n(↓ forget, ↑ retain)",
                        xy=(0.99, 0.02), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=9, color="#1a9641",
                        style="italic")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"Retain-set median {metric_label}  (↑ better)", fontsize=12)
        ax.set_ylabel(f"Forget-set median {metric_label}  (↓ better)", fontsize=12)
        ax.set_title(metric_label, fontsize=14, pad=6)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        if ox is not None:
            ax.legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=600, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
    print(f"        {out_path.with_suffix('.pdf')}")


def plot_rouge_dual(method_rouge_forget, method_rouge_retain, metrics,
                    palette, out_path, title, oracle_name="Retrained"):
    """Dual side-by-side boxplots — for each method, two boxes per metric panel:
       left box = forget-set (hatched), right box = retain-set (solid).
       Two horizontal reference lines: oracle median forget ROUGE and oracle median retain ROUGE."""
    # Methods present in both — preserve order from the union
    seen = set()
    names = []
    for n in list(method_rouge_forget.keys()) + list(method_rouge_retain.keys()):
        if n not in seen:
            seen.add(n)
            names.append(n)

    n_methods, n_metrics = len(names), len(metrics)
    fig, axes = plt.subplots(1, n_metrics,
                             figsize=(max(13, 4.0 * n_methods), 5.6),
                             sharey=False)
    if n_metrics == 1: axes = [axes]

    box_w = 0.36       # narrower so two fit in one slot
    offset = 0.21      # horizontal offset for forget vs retain

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        # Compute positions, data, and styling for forget (left) and retain (right)
        positions_f = [i + 1 - offset for i in range(n_methods)]
        positions_r = [i + 1 + offset for i in range(n_methods)]
        data_f, data_r = [], []
        for n in names:
            data_f.append([r[metric_key] for r in method_rouge_forget.get(n, [])] or [np.nan])
            data_r.append([r[metric_key] for r in method_rouge_retain.get(n, [])] or [np.nan])

        bp_f = ax.boxplot(
            data_f, positions=positions_f, widths=box_w,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.6),
            flierprops=dict(marker="o", markersize=2.5, alpha=0.35,
                            markerfacecolor="black", markeredgecolor="black"),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
        )
        bp_r = ax.boxplot(
            data_r, positions=positions_r, widths=box_w,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.6),
            flierprops=dict(marker="o", markersize=2.5, alpha=0.35,
                            markerfacecolor="black", markeredgecolor="black"),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
        )

        # Style: forget = method color hatched; retain = method color solid
        for n, patch in zip(names, bp_f["boxes"]):
            c = palette.get(n, "#bbbbbb")
            ec = "#111111" if n == "Ours" else "#333333"
            lw = 2.0 if n == "Ours" else 0.9
            patch.set_facecolor(c); patch.set_alpha(0.45)
            patch.set_edgecolor(ec); patch.set_linewidth(lw)
            patch.set_hatch("///")
        for n, patch in zip(names, bp_r["boxes"]):
            c = palette.get(n, "#bbbbbb")
            ec = "#111111" if n == "Ours" else "#333333"
            lw = 2.0 if n == "Ours" else 0.9
            patch.set_facecolor(c); patch.set_alpha(0.85)
            patch.set_edgecolor(ec); patch.set_linewidth(lw)

        # Oracle reference lines: median forget and median retain ROUGE for the metric.
        # Inline value tags placed just inside the right edge so they don't crop off boxes.
        if oracle_name in method_rouge_forget:
            oracle_f = float(np.median(
                [r[metric_key] for r in method_rouge_forget[oracle_name]]))
            ax.axhline(oracle_f, color="#d6604d", linestyle="--",
                       linewidth=1.7, alpha=0.85, zorder=5)
            ax.text(0.99, oracle_f + 0.015, f"oracle forget = {oracle_f:.2f}",
                    transform=ax.get_yaxis_transform(),
                    color="#d6604d", fontsize=9, va="bottom", ha="right",
                    fontweight="bold")
        if oracle_name in method_rouge_retain:
            oracle_r = float(np.median(
                [r[metric_key] for r in method_rouge_retain[oracle_name]]))
            ax.axhline(oracle_r, color="#1a9641", linestyle="--",
                       linewidth=1.7, alpha=0.85, zorder=5)
            ax.text(0.99, oracle_r - 0.015, f"oracle retain = {oracle_r:.2f}",
                    transform=ax.get_yaxis_transform(),
                    color="#1a9641", fontsize=9, va="top", ha="right",
                    fontweight="bold")

        ax.set_xticks(range(1, n_methods + 1))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=12)
        ax.set_title(metric_label, fontsize=14, pad=6)
        ax.set_ylabel("Score", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0.5, n_methods + 0.5)
        ax.grid(axis="y", alpha=0.3)

    # Single legend ABOVE the panels (outside the data area) so it doesn't cover boxes.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor="#bbbbbb", alpha=0.45, hatch="///", edgecolor="#333333",
              label="Forget set (↓ better)"),
        Patch(facecolor="#bbbbbb", alpha=0.85, edgecolor="#333333",
              label="Retain set (↑ better)"),
        Line2D([0], [0], color="#d6604d", linestyle="--", linewidth=1.7,
               label="oracle forget median"),
        Line2D([0], [0], color="#1a9641", linestyle="--", linewidth=1.7,
               label="oracle retain median"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), ncol=4,
               frameon=True, fontsize=10)
    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=600, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved {out_path}")
    print(f"        {out_path.with_suffix('.pdf')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forget01", choices=["forget01", "forget05", "forget10"])
    ap.add_argument("--out_dir", default="analysis_out")
    ap.add_argument("--model", default="Llama-3.2-1B-Instruct")
    args = ap.parse_args()

    SAVES = Path("saves")
    MODEL = args.model
    SPLIT = args.split

    # FLOUR (Ours) ckpt selection — closest fully-evaluated ckpt to the paper-table pick.
    # (forget10 ck50 has only forget-side fields populated; ck75 is the next adjacent
    # checkpoint with full retain_Q_A_ROUGE.)
    flour_paths = {
        "forget01": SAVES / f"unlearn/tofu_{MODEL}_forget01_LatentRMU_v4.8_sweep_01_coeff_5/checkpoint-50/evals/TOFU_EVAL.json",
        "forget05": SAVES / f"unlearn/tofu_{MODEL}_forget05_LatentRMU_v4.8_sweep_g0.50/checkpoint-50/evals/TOFU_EVAL.json",
        "forget10": SAVES / f"unlearn/tofu_{MODEL}_forget10_LatentRMU_v4.8_sweep_g0.50/checkpoint-75/evals/TOFU_EVAL.json",
    }
    retain_split = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}[SPLIT]

    METHODS = [
        ("Retrained",  SAVES / f"eval/tofu_{MODEL}_{retain_split}/TOFU_EVAL.json"),
        ("GradAscent", SAVES / f"unlearn/tofu_{MODEL}_{SPLIT}_GradAscent/evals/TOFU_EVAL.json"),
        ("GradDiff",   SAVES / f"unlearn/tofu_{MODEL}_{SPLIT}_GradDiff/evals/TOFU_EVAL.json"),
        ("NPO",        SAVES / f"unlearn/tofu_{MODEL}_{SPLIT}_NPO/evals/TOFU_EVAL.json"),
        ("RMU",        SAVES / f"unlearn/tofu_{MODEL}_{SPLIT}_RMU/evals/TOFU_EVAL.json"),
        ("SimNPO",     SAVES / f"unlearn/tofu_{MODEL}_{SPLIT}_SimNPO/evals/TOFU_EVAL.json"),
        ("Ours",       flour_paths[SPLIT]),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rouge_data = {}
    for which, eval_key in [("forget", "forget_Q_A_ROUGE"),
                            ("retain", "retain_Q_A_ROUGE")]:
        print(f"\n── {which} set ROUGE ──")
        rouge_data[which] = {}
        for name, path in METHODS:
            if not path.exists():
                print(f"  MISSING: {name}  ({path})")
                continue
            samples = load_per_sample(path, eval_key)
            if samples is None:
                print(f"  [skip] {name}: '{eval_key}' not in eval JSON")
                continue
            rouge_data[which][name] = samples
            agg = {m: np.mean([r[m] for r in samples]) for m, _ in METRICS}
            print(f"  {name:11s}  n={len(samples)}  " +
                  "  ".join(f"{lbl}={v:.3f}" for (_, lbl), v in zip(METRICS, agg.values())))

    # Tradeoff scatter — cleaner: one point per method, oracle marked, IQR error bars
    print(f"\n── TRADEOFF scatter (cleanest single-figure view) ──")
    tradeoff_path = out_dir / f"rouge_tradeoff_{SPLIT}.png"
    plot_rouge_tradeoff(
        rouge_data["forget"], rouge_data["retain"], METRICS, PALETTE, tradeoff_path,
        title=f"ROUGE forget/retain tradeoff — {SPLIT}   (★ = oracle, pink = Ours)",
    )

    # Optional dual-boxplot (denser; kept for backward compat)
    print(f"\n── DUAL boxplot (denser; per-example distribution) ──")
    dual_path = out_dir / f"rouge_boxplot_{SPLIT}_dual.png"
    plot_rouge_dual(
        rouge_data["forget"], rouge_data["retain"], METRICS, PALETTE, dual_path,
        title=f"Forget vs Retain ROUGE — {SPLIT}  "
              f"(hatched = forget, solid = retain;  oracle ref: {retain_split})",
    )


if __name__ == "__main__":
    main()
