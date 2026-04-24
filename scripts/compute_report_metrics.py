#!/usr/bin/env python
"""Compute Memorization Score, Privacy Score, and Utility for the paper report.

Reads TOFU_EVAL.json from saves/unlearn/*/checkpoint-*/evals/ and computes
composite scores normalized by checkpoint-0 (the init finetuned model).

Memorization Score = HM(1-ES_norm, [1-EM_norm,] 1-Para.Prob_norm, 1-TR_norm)
  All metrics normalized by checkpoint-0 so scores fall in [0,1].
  Note: EM is included only if exact_memorization was computed (requires re-eval).
  Truth Ratio uses the stored forget_truth_ratio (closer_to_1_better aggregator).

Privacy Score = privleak (MIA-based relative measure, already normalized)

Utility = model_utility / model_utility_init (normalized by init model)

Usage:
    python scripts/compute_report_metrics.py
    python scripts/compute_report_metrics.py --runs saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_LatentRMU_v4.8
    python scripts/compute_report_metrics.py --select_by utility --verbose
    python scripts/compute_report_metrics.py --missing_em_only  # list checkpoints needing re-eval
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import hmean


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_agg(eval_data, key):
    entry = eval_data.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("agg_value")
    return float(entry)


def normalize_memorized(val, ref_val):
    """Normalize metric where higher = more memorized. Returns val/ref capped at 1."""
    if ref_val is None or ref_val <= 0 or val is None:
        return None
    return min(val / ref_val, 1.0)


def normalize_tr(tr_val, tr_init):
    """Normalize forget_truth_ratio (closer_to_1_better: higher TR = more forgotten).
    Memorization signal = (1 - TR). Normalize: (1-TR)/(1-TR_init) in [0,1]."""
    if tr_val is None or tr_init is None:
        return None
    mem_signal = 1.0 - tr_val
    mem_signal_init = 1.0 - tr_init
    if mem_signal_init <= 0:
        return None
    return min(mem_signal / mem_signal_init, 1.0)


def compute_scores(eval_data, init_data):
    es = get_agg(eval_data, "extraction_strength")
    em = get_agg(eval_data, "exact_memorization")
    para_prob = get_agg(eval_data, "forget_Q_A_PARA_Prob")
    tr = get_agg(eval_data, "forget_truth_ratio")
    mu = get_agg(eval_data, "model_utility")
    privleak = get_agg(eval_data, "privleak")

    es_init = get_agg(init_data, "extraction_strength")
    em_init = get_agg(init_data, "exact_memorization")
    para_init = get_agg(init_data, "forget_Q_A_PARA_Prob")
    tr_init = get_agg(init_data, "forget_truth_ratio")
    mu_init = get_agg(init_data, "model_utility")

    es_norm = normalize_memorized(es, es_init)
    em_norm = normalize_memorized(em, em_init)
    pp_norm = normalize_memorized(para_prob, para_init)
    tr_norm = normalize_tr(tr, tr_init)

    # Forgetting components: each in [0,1] where higher = better forgetting
    components = {}
    if es_norm is not None:
        components["es"] = 1.0 - es_norm
    if em_norm is not None:
        components["em"] = 1.0 - em_norm
    if pp_norm is not None:
        components["para_prob"] = 1.0 - pp_norm
    if tr_norm is not None:
        components["tr"] = 1.0 - tr_norm

    if components:
        # hmean requires strictly positive values; clip to small epsilon
        vals = [max(v, 1e-10) for v in components.values()]
        mem_score = float(hmean(vals))
    else:
        mem_score = None

    utility = None
    if mu is not None and mu_init is not None and mu_init > 0:
        utility = min(mu / mu_init, 1.0)

    return {
        "mem_score": mem_score,
        "n_components": len(components),
        "has_em": em_norm is not None,
        "utility": utility,
        "privleak": privleak,
        "components": {
            "es_norm": es_norm,
            "em_norm": em_norm,
            "pp_norm": pp_norm,
            "tr_norm": tr_norm,
        },
        "raw": {
            "es": es, "em": em, "para_prob": para_prob,
            "tr": tr, "model_utility": mu,
        },
    }


def find_checkpoints(run_dir):
    pattern = str(run_dir / "checkpoint-*" / "evals" / "TOFU_EVAL.json")
    paths = glob.glob(pattern)
    results = []
    for p in paths:
        m = re.search(r"checkpoint-(\d+)", p)
        if m:
            results.append((int(m.group(1)), Path(p)))
    return sorted(results)


def parse_run_name(name):
    m = re.match(r"tofu_(.*?)_(forget\d+)_(.*)", name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return name, "", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saves_dir", default="saves/unlearn", help="Root saves directory")
    parser.add_argument("--runs", nargs="*", help="Specific run directories (optional)")
    parser.add_argument(
        "--select_by", default="mem_score", choices=["mem_score", "utility"],
        help="Metric to use for best-checkpoint selection (default: mem_score)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-component values")
    parser.add_argument(
        "--missing_em_only", action="store_true",
        help="Print eval commands for checkpoints missing exact_memorization, then exit"
    )
    parser.add_argument("--gpu", default="0", help="GPU id for re-eval commands")
    args = parser.parse_args()

    saves_dir = Path(args.saves_dir)

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        run_dirs = sorted(d for d in saves_dir.iterdir() if d.is_dir() and d.name.startswith("tofu_"))

    if args.missing_em_only:
        _print_reeval_commands(run_dirs, args.gpu)
        return

    hdr = f"{'Experiment':<58} {'Ckpt':>5} | {'MemScore':>9} {'Comp':>4} | {'Utility':>8} | {'Privleak':>9}"
    print(hdr)
    print("-" * len(hdr))

    for run_dir in run_dirs:
        checkpoints = find_checkpoints(run_dir)
        if not checkpoints:
            continue

        init_step, init_path = checkpoints[0]
        init_data = load_json(init_path)

        results = []
        for step, path in checkpoints:
            eval_data = load_json(path)
            scores = compute_scores(eval_data, init_data)
            scores["step"] = step
            results.append(scores)

        non_init = results[1:] if len(results) > 1 else results

        def sort_key(r):
            v = r.get(args.select_by)
            return v if v is not None else -1.0

        best = max(non_init, key=sort_key)

        model, split, method = parse_run_name(run_dir.name)
        label = f"{method} | {split} | {model[:28]}"
        mem = f"{best['mem_score']:.4f}" if best["mem_score"] is not None else "N/A"
        util = f"{best['utility']:.4f}" if best["utility"] is not None else "N/A"
        pl = f"{best['privleak']:.2f}" if best["privleak"] is not None else "N/A"
        nc = best["n_components"]
        em_flag = "" if best["has_em"] else " *"

        print(f"{label:<58} {best['step']:>5} | {mem:>9}{em_flag:<2} {nc:>4} | {util:>8} | {pl:>9}")

        if args.verbose:
            c = best["components"]
            em_str = f"{c['em_norm']:.3f}" if c["em_norm"] is not None else "missing"
            print(
                f"  ES_norm={c['es_norm']:.3f}  EM_norm={em_str}"
                f"  PP_norm={c['pp_norm']:.3f}  TR_norm={c['tr_norm']:.3f}"
            )

    print()
    print("* = exact_memorization missing; Mem Score uses 3-component HM.")
    print("  Run with --missing_em_only to get re-eval commands.")


def _print_reeval_commands(run_dirs, gpu):
    """Print eval commands for checkpoints missing exact_memorization."""
    print("# Re-eval commands to add exact_memorization to existing checkpoints")
    print("# Run after enabling exact_memorization in configs/eval/tofu.yaml\n")
    for run_dir in run_dirs:
        checkpoints = find_checkpoints(run_dir)
        for step, path in checkpoints:
            try:
                eval_data = load_json(path)
            except Exception:
                continue
            if get_agg(eval_data, "exact_memorization") is not None:
                continue  # already computed

            name = run_dir.name
            m = re.match(r"tofu_(.*?)_(forget\d+)_(.*)", name)
            if not m:
                continue
            model, forget_split, _ = m.group(1), m.group(2), m.group(3)

            retain_split = "retain" + str(100 - int(forget_split.replace("forget", "")))
            holdout_split = "holdout" + forget_split.replace("forget", "")

            retain_logs = f"saves/eval/tofu_{model}_{retain_split}/TOFU_EVAL.json"
            ckpt_path = run_dir / f"checkpoint-{step}"
            output_dir = ckpt_path / "evals"

            print(
                f"CUDA_VISIBLE_DEVICES={gpu} python src/eval.py --config-name=eval.yaml "
                f"experiment=eval/tofu/default "
                f"forget_split={forget_split} holdout_split={holdout_split} "
                f"model={model} task_name={name} "
                f"model.model_args.pretrained_model_name_or_path={ckpt_path} "
                f"paths.output_dir={output_dir} "
                f"eval.tofu.overwrite=true "
                f"retain_logs_path={retain_logs}"
            )
    print()


if __name__ == "__main__":
    main()
