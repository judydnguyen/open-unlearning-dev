#!/usr/bin/env python
"""Extract and compare results of baseline unlearning methods from saves/unlearn.

Scans tofu_* directories, reads evals/TOFU_SUMMARY.json and evals/TOFU_EVAL.json,
and produces a Markdown report grouped by (model, split).

Usage:
    python scripts/extract_baseline_results.py
    python scripts/extract_baseline_results.py --unlearn-dir saves/unlearn --output saves/unlearn/baseline_results.md
    python scripts/extract_baseline_results.py --filter-model Llama-3.2-3B-Instruct --filter-split forget10
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict


SUMMARY_METRICS = [
    "model_utility",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "extraction_strength",
    "privleak",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_dir_name(name):
    """Parse tofu_{model}_{split}_{method} → (model, split, method) or None."""
    m = re.match(r"tofu_(.+?)_(forget\d+)_(.+)", name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def parse_steering_dir_name(name):
    """Parse steering_{variant}_{split} → (variant, split) or None."""
    m = re.match(r"steering_(.+?)_(forget\d+)$", name)
    if not m:
        return None
    return m.group(1), m.group(2)


def model_from_config(base):
    """Extract model name from config.json _name_or_path field."""
    config_path = os.path.join(base, "config.json")
    if not os.path.exists(config_path):
        return "unknown"
    try:
        cfg = load_json(config_path)
        name_or_path = cfg.get("_name_or_path", "")
        # e.g. "open-unlearning/tofu_Llama-3.2-1B-Instruct_full" → "Llama-3.2-1B-Instruct"
        part = name_or_path.split("/")[-1]
        m = re.search(r"(Llama[^_]+)", part)
        if m:
            return m.group(1)
        return part or "unknown"
    except Exception:
        return "unknown"


def extract_question(input_text):
    """Extract just the question from the chat-template input."""
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


def cell(s):
    return str(s).replace("|", "\\|")


def collect_forget_generations(eval_json, n=None, max_len=150):
    """Return list of dicts keyed by str(idx) with question/ground_truth/generation."""
    key = "forget_Q_A_ROUGE"
    if key not in eval_json:
        return {}
    entries = eval_json[key].get("value_by_index", {})
    result = {}
    for idx in sorted(entries.keys(), key=lambda x: int(x)):
        entry = entries[idx]
        if "generation" not in entry:
            continue
        gen = entry["generation"].strip().replace("\n", " ")
        if max_len and len(gen) > max_len:
            gen = gen[:max_len] + "…"
        result[idx] = {
            "input": entry.get("input", ""),
            "ground_truth": entry.get("ground_truth", "").strip(),
            "generation": gen,
            "rouge_l": entry.get("rougeL_recall"),
        }
        if n and len(result) >= n:
            break
    return result


def scan_runs(unlearn_dir):
    """Scan directory and return list of run dicts with parsed metadata and paths."""
    runs = []
    for name in sorted(os.listdir(unlearn_dir)):
        base = os.path.join(unlearn_dir, name)
        if not os.path.isdir(base):
            continue
        summary_path = os.path.join(base, "evals", "TOFU_SUMMARY.json")
        eval_path = os.path.join(base, "evals", "TOFU_EVAL.json")

        if name.startswith("tofu_"):
            parsed = parse_dir_name(name)
            if not parsed:
                continue
            model, split, method = parsed
            if not os.path.exists(summary_path):
                continue
        elif name.startswith("steering_"):
            parsed = parse_steering_dir_name(name)
            if not parsed:
                continue
            variant, split = parsed
            model = model_from_config(base)
            method = f"steering_{variant}"
            # Include even if eval not finished; summary_path may be None
            summary_path = summary_path if os.path.exists(summary_path) else None
        else:
            continue

        runs.append({
            "name": name,
            "model": model,
            "split": split,
            "method": method,
            "summary_path": summary_path,
            "eval_path": eval_path if os.path.exists(eval_path) else None,
        })
    return runs


def build_metrics_table(runs_group):
    """Markdown metrics table for a group of runs (same model+split)."""
    lines = []
    header = "| Method | " + " | ".join(SUMMARY_METRICS) + " |"
    sep = "|---| " + " | ".join("---" for _ in SUMMARY_METRICS) + " |"
    lines.append(header)
    lines.append(sep)
    for run in runs_group:
        if run["summary_path"]:
            data = load_json(run["summary_path"])
            vals = " | ".join(fmt(data.get(m)) for m in SUMMARY_METRICS)
        else:
            vals = " | ".join("*(pending)*" for _ in SUMMARY_METRICS)
        lines.append(f"| {run['method']} | {vals} |")
    return lines


def build_forget_comparison_table(runs_group, n=10, max_len=150):
    """Markdown table comparing forget-set generations across methods."""
    # Collect per-method generations
    method_gens = {}  # method -> {idx: {...}}
    for run in runs_group:
        if not run["eval_path"]:
            continue
        try:
            eval_json = load_json(run["eval_path"])
        except Exception:
            continue
        gens = collect_forget_generations(eval_json, n=n, max_len=max_len)
        if gens:
            method_gens[run["method"]] = gens

    if not method_gens:
        return []

    methods = list(method_gens.keys())

    # Union of all indices that appear in any method
    all_idxs = sorted(
        set(idx for gens in method_gens.values() for idx in gens),
        key=lambda x: int(x)
    )[:n]

    # Use the first available method to get question/ground_truth for each idx
    def get_meta(idx):
        for gens in method_gens.values():
            if idx in gens:
                return gens[idx]
        return {}

    col_header = " | ".join(cell(m) for m in methods)
    lines = []
    lines.append(f"| # | Question | Ground Truth | {col_header} |")
    lines.append("|---|---|---| " + " | ".join("---" for _ in methods) + " |")

    for idx in all_idxs:
        meta = get_meta(idx)
        q = cell(extract_question(meta.get("input", ""))[:120])
        gt = cell(meta.get("ground_truth", "")[:max_len].replace("\n", " "))
        gens = " | ".join(
            cell(method_gens[m].get(idx, {}).get("generation", "—"))
            for m in methods
        )
        lines.append(f"| {idx} | {q} | {gt} | {gens} |")

    return lines


def generate_report(unlearn_dir, output_path, filter_model=None, filter_split=None):
    runs = scan_runs(unlearn_dir)

    if filter_model:
        runs = [r for r in runs if filter_model in r["model"]]
    if filter_split:
        runs = [r for r in runs if r["split"] == filter_split]

    if not runs:
        print("No runs found.")
        return

    # Group by (model, split)
    groups = defaultdict(list)
    for run in runs:
        groups[(run["model"], run["split"])].append(run)

    lines = []
    w = lines.append

    w("# Baseline Unlearning Results\n")
    w(f"Source: `{unlearn_dir}`\n")

    # Table of contents
    w("## Contents\n")
    for (model, split) in sorted(groups.keys()):
        anchor = f"{model}-{split}".lower().replace(".", "").replace("_", "-")
        w(f"- [{model} / {split}](#{anchor})")
    w("")

    for (model, split) in sorted(groups.keys()):
        group = sorted(groups[(model, split)], key=lambda r: r["method"])
        anchor = f"{model}-{split}".lower().replace(".", "").replace("_", "-")
        w(f"## {model} / {split}\n")

        # Metrics table
        w("### Metrics\n")
        for line in build_metrics_table(group):
            w(line)
        w("")

        # Forget response comparison
        table = build_forget_comparison_table(group, n=10)
        if table:
            w("### Forget Set Responses\n")
            for line in table:
                w(line)
            w("")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report written to {output_path}")
    print(f"  {len(runs)} runs across {len(groups)} (model, split) groups")


def main():
    parser = argparse.ArgumentParser(description="Extract baseline unlearning results")
    parser.add_argument("--unlearn-dir", default="saves/unlearn",
                        help="Directory containing tofu_* run dirs (default: saves/unlearn)")
    parser.add_argument("--output", default=None,
                        help="Output markdown path (default: <unlearn-dir>/baseline_results.md)")
    parser.add_argument("--filter-model", default=None,
                        help="Only include runs whose model name contains this string")
    parser.add_argument("--filter-split", default=None,
                        help="Only include runs with this exact split (e.g. forget10)")
    args = parser.parse_args()

    output = args.output or os.path.join(args.unlearn_dir, "baseline_results.md")
    generate_report(args.unlearn_dir, output,
                    filter_model=args.filter_model,
                    filter_split=args.filter_split)


if __name__ == "__main__":
    main()
