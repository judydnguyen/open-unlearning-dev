"""
Plot GRPO training stats from grpo_log.jsonl.

Usage:
    # One-shot (saves PNGs)
    python scripts/plot_grpo_log.py <path/to/grpo_log.jsonl>

    # Live dashboard (polls the file, redraws every --interval seconds)
    python scripts/plot_grpo_log.py <path/to/grpo_log.jsonl> --watch

    # All options
    python scripts/plot_grpo_log.py <path/to/grpo_log.jsonl> \
        [--watch] [--interval 5] [--out <dir>] [--max_candidate_plots 6]

Produces:
    reward_stats.png        — mean ± std band, min/max, per-group variance over steps
    candidates_step<N>.png  — per-candidate reward bars with completion text snippets
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_log(path: Path):
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # skip partial lines being written
    except FileNotFoundError:
        pass
    return records


# ─────────────────────────────────────────────────────────────
# One-shot plots (saved to disk)
# ─────────────────────────────────────────────────────────────

def plot_reward_stats(records, out_dir: Path):
    if not records:
        return
    steps      = [r["step"]           for r in records]
    means      = [r["reward_mean"]    for r in records]
    variances  = [r["reward_var"]     for r in records]
    mins       = [r["reward_min"]     for r in records]
    maxs       = [r["reward_max"]     for r in records]
    group_vars = [r["group_var_mean"] for r in records]
    stds = np.sqrt(variances)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.fill_between(steps, mins, maxs, alpha=0.15, color="steelblue", label="min/max")
    ax.fill_between(
        steps,
        np.array(means) - stds, np.array(means) + stds,
        alpha=0.35, color="steelblue", label="mean ± std",
    )
    ax.plot(steps, means, color="steelblue", linewidth=1.5, label="mean reward")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Reward")
    ax.set_title("GRPO Reward over Training Steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, group_vars, color="darkorange", linewidth=1.5, label="group var (mean)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Within-group Variance")
    ax.set_title("Mean Within-group Reward Variance (collapse → 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "reward_stats.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_candidate_rewards(records, out_dir: Path, max_plots: int = 6):
    sample_records = [r for r in records if "samples" in r]
    if not sample_records:
        return

    indices = np.linspace(
        0, len(sample_records) - 1,
        min(max_plots, len(sample_records)), dtype=int
    )

    for i in indices:
        rec     = sample_records[i]
        step    = rec["step"]
        samples = rec["samples"]
        n_prompts = len(samples)
        G = len(samples[0]["candidates"])

        fig, axes = plt.subplots(
            1, n_prompts, figsize=(max(6, 3 * n_prompts), 5), squeeze=False
        )
        for b, sample in enumerate(samples):
            ax = axes[0][b]
            rewards = [c["reward"] for c in sample["candidates"]]
            colors = [
                "#d62728" if r == min(rewards)
                else "#2ca02c" if r == max(rewards)
                else "steelblue"
                for r in rewards
            ]
            ax.bar(range(G), rewards, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(G))
            ax.set_xticklabels([f"c{g}" for g in range(G)])
            ax.set_ylabel("Reward" if b == 0 else "")
            ax.set_title(f"prompt {b}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            for g, c in enumerate(sample["candidates"]):
                short = c["completion"][:40].replace("\n", " ")
                ax.annotate(
                    f'"{short}…"',
                    xy=(g, rewards[g]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", fontsize=5, rotation=45,
                )

        fig.suptitle(f"Candidate Rewards at Step {step}", fontsize=11)
        fig.tight_layout()
        out_path = out_dir / f"candidates_step{step:05d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# Live dashboard
# ─────────────────────────────────────────────────────────────

def _draw_live(fig, axes, records):
    """Redraw the live figure in-place from current records."""
    ax_reward, ax_var, ax_cand = axes

    for ax in axes:
        ax.cla()

    if not records:
        for ax in axes:
            ax.set_title("Waiting for data…")
        return

    steps      = [r["step"]           for r in records]
    means      = [r["reward_mean"]    for r in records]
    variances  = [r["reward_var"]     for r in records]
    mins       = [r["reward_min"]     for r in records]
    maxs       = [r["reward_max"]     for r in records]
    group_vars = [r["group_var_mean"] for r in records]
    stds = np.sqrt(variances)

    # ── Reward mean / std / min-max ────────────────────────────
    ax_reward.fill_between(steps, mins, maxs, alpha=0.15, color="steelblue", label="min/max")
    ax_reward.fill_between(
        steps,
        np.array(means) - stds, np.array(means) + stds,
        alpha=0.35, color="steelblue", label="mean ± std",
    )
    ax_reward.plot(steps, means, color="steelblue", linewidth=1.5, label="mean reward")
    ax_reward.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_reward.set_ylabel("Reward")
    ax_reward.set_title("GRPO Reward")
    ax_reward.legend(fontsize=7)
    ax_reward.grid(True, alpha=0.3)

    # ── Within-group variance ──────────────────────────────────
    ax_var.plot(steps, group_vars, color="darkorange", linewidth=1.5)
    ax_var.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_var.set_xlabel("Step")
    ax_var.set_ylabel("Group Variance")
    ax_var.set_title("Within-group Reward Variance")
    ax_var.grid(True, alpha=0.3)

    # ── Latest candidate bars ──────────────────────────────────
    sample_records = [r for r in records if "samples" in r]
    if sample_records:
        latest = sample_records[-1]
        step   = latest["step"]
        sample = latest["samples"][0]   # first prompt only in live view
        G      = len(sample["candidates"])
        rewards = [c["reward"] for c in sample["candidates"]]
        colors  = [
            "#d62728" if r == min(rewards)
            else "#2ca02c" if r == max(rewards)
            else "steelblue"
            for r in rewards
        ]
        ax_cand.bar(range(G), rewards, color=colors, edgecolor="black", linewidth=0.5)
        ax_cand.set_xticks(range(G))
        ax_cand.set_xticklabels([f"c{g}" for g in range(G)])
        ax_cand.set_title(f"Latest candidates (step {step}, prompt 0)")
        ax_cand.grid(axis="y", alpha=0.3)
        for g, c in enumerate(sample["candidates"]):
            short = c["completion"][:35].replace("\n", " ")
            ax_cand.annotate(
                f'"{short}…"',
                xy=(g, rewards[g]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center", fontsize=6, rotation=40,
            )
    else:
        ax_cand.set_title("Waiting for candidate samples…")

    fig.suptitle(f"GRPO Live — {len(records)} steps", fontsize=11)
    fig.tight_layout()


def watch_live(log_path: Path, interval: float, out_dir: Path, max_candidate_plots: int):
    """Interactive live dashboard using FuncAnimation."""
    matplotlib.use("TkAgg")   # interactive backend; falls back gracefully if unavailable

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle("GRPO Live Dashboard")

    _last_n = [0]

    def update(_frame):
        records = load_log(log_path)
        if len(records) == _last_n[0]:
            return          # nothing new
        _last_n[0] = len(records)
        _draw_live(fig, axes, records)
        # Also save static PNGs on each refresh
        plot_reward_stats(records, out_dir)
        plot_candidate_rewards(records, out_dir, max_plots=max_candidate_plots)

    ani = animation.FuncAnimation(
        fig, update,
        interval=int(interval * 1000),  # ms
        cache_frame_data=False,
    )

    try:
        plt.show()
    except Exception:
        # Headless fallback: just poll and save PNGs
        print("No display available — running in headless mode, saving PNGs only.")
        while True:
            records = load_log(log_path)
            plot_reward_stats(records, out_dir)
            plot_candidate_rewards(records, out_dir, max_plots=max_candidate_plots)
            print(f"[{time.strftime('%H:%M:%S')}] {len(records)} steps logged", flush=True)
            time.sleep(interval)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", help="Path to grpo_log.jsonl")
    parser.add_argument("--watch", action="store_true",
                        help="Live-update the plot as training writes new data")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Refresh interval in seconds (--watch only, default 5)")
    parser.add_argument("--out", default=None,
                        help="Output directory for PNG files (default: same as log)")
    parser.add_argument("--max_candidate_plots", type=int, default=6)
    args = parser.parse_args()

    log_path = Path(args.log_file)
    out_dir  = Path(args.out) if args.out else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.watch:
        print(f"Watching {log_path} (refresh every {args.interval}s) — Ctrl-C to stop")
        watch_live(log_path, args.interval, out_dir, args.max_candidate_plots)
    else:
        records = load_log(log_path)
        print(f"Loaded {len(records)} records from {log_path}")
        plot_reward_stats(records, out_dir)
        plot_candidate_rewards(records, out_dir, max_plots=args.max_candidate_plots)


if __name__ == "__main__":
    main()
