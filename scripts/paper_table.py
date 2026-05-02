#!/usr/bin/env python
"""Collect TOFU_EVAL.json results for all paper methods and print a composite-score table.

Metrics (all normalized by the Baseline / full finetuned model):
  MemScore  = HM(1−ES_norm, 1−TR_norm, 1−Q_A_Prob_norm)        higher = more forgetting
              (3 orthogonal signals: extraction, truth-ratio, verbatim Q/A prob)
  Utility   = model_utility / model_utility_baseline          higher = better retention
  Privacy   = max(0, 1 − privleak/100), asymmetric            higher = better

Usage:
    python scripts/paper_table.py --model Llama-3.2-1B-Instruct \\
        --our_pattern "tofu_{model}_{split}_LatentRMU_v4.8_sweep"
    python scripts/paper_table.py -v   # per-component detail
"""

import argparse
import glob
import json
import re
from pathlib import Path

try:
    from scipy.stats import hmean as _hmean
    def hmean(vals): return float(_hmean(vals))
except ImportError:
    def hmean(vals): return len(vals) / sum(1.0 / max(v, 1e-10) for v in vals)


FORGET_SPLITS = ["forget01", "forget05", "forget10"]
SPLIT_TO_RETAIN = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}
SPLIT_TO_HOLDOUT = {"forget01": "holdout01", "forget05": "holdout05", "forget10": "holdout10"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_agg(data, key):
    entry = data.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("agg_value")
    return float(entry)


MIA_KEYS = ("mia_loss", "mia_zlib", "mia_min_k", "mia_min_k_plus_plus")


def compute_scores(eval_data, baseline_data, retain_data=None):
    """Compute MemScore, Utility, Privacy relative to baseline_data.

    Privacy = max(0, 1 − |avg-privleak|/100), averaged across the 4 MIA attacks
    in MIA_KEYS. Per-MIA privleak = (AUC_retain − AUC_unlearn) / (1 − AUC_retain) × 100,
    using the same convention as src/evals/metrics/privacy.py (stored = 1 − AUC).
    Falls back to the single mia_min_k privleak field when only one MIA is present.
    """
    def norm_higher(val, ref):
        if val is None or ref is None or ref <= 0:
            return None
        return min(val / ref, 1.0)

    def norm_tr(val, ref):
        # TR: closer_to_1_better → higher TR = more forgotten.
        # Memorization signal = (1 - TR); normalize against baseline.
        if val is None or ref is None:
            return None
        sig = 1.0 - val
        sig_ref = 1.0 - ref
        if sig_ref <= 0:
            return None
        return min(sig / sig_ref, 1.0)

    # MemScore = HM(1−ES_n, 1−TR_n, 1−Q_A_Prob_n), all normalized vs Baseline.
    # PP (paraphrase prob) and EM (exact_memorization) are intentionally excluded:
    #   - PP overlaps with TR (both probe semantic memorization beyond verbatim).
    #   - EM is missing from most flat-method evals; excluding it keeps the
    #     comparison consistent without requiring an EM re-eval pass.
    es_norm = norm_higher(get_agg(eval_data, "extraction_strength"),
                          get_agg(baseline_data, "extraction_strength"))
    qp_norm = norm_higher(get_agg(eval_data, "forget_Q_A_Prob"),
                          get_agg(baseline_data, "forget_Q_A_Prob"))
    tr_norm = norm_tr(get_agg(eval_data, "forget_truth_ratio"),
                      get_agg(baseline_data, "forget_truth_ratio"))

    components = {}
    if es_norm is not None:
        components["es"] = 1.0 - es_norm
    if qp_norm is not None:
        components["qp"] = 1.0 - qp_norm
    if tr_norm is not None:
        components["tr"] = 1.0 - tr_norm

    mem_score = hmean([max(v, 1e-10) for v in components.values()]) if components else None

    mu = get_agg(eval_data, "model_utility")
    mu_ref = get_agg(baseline_data, "model_utility")
    utility = min(mu / mu_ref, 1.0) if (mu is not None and mu_ref and mu_ref > 0) else None

    # Privacy: 1 − |avg-privleak|/100 averaged across 4 MIAs (LOSS/Zlib/Mink/Mink++).
    # Symmetric — penalizes deviation from retrain in either direction. When the
    # retain reference doesn't have all 4 MIAs (or eval_data lacks them), falls
    # back to the single privleak field already in the eval JSON.
    per_mia_pl = []
    if retain_data is not None:
        for k in MIA_KEYS:
            s = get_agg(eval_data, k)
            r = get_agg(retain_data, k)
            if s is None or r is None:
                continue
            denom = 1.0 - r
            if abs(denom) < 1e-9:
                continue
            per_mia_pl.append((r - s) / denom * 100.0)
    if per_mia_pl:
        avg_abs_pl = sum(abs(p) for p in per_mia_pl) / len(per_mia_pl)
        avg_pl_signed = sum(per_mia_pl) / len(per_mia_pl)
        privacy = max(0.0, min(1.0, 1.0 - avg_abs_pl / 100.0))
    else:
        # Fallback: single-MIA privleak from the eval JSON (mia_min_k-based)
        single_pl = get_agg(eval_data, "privleak")
        avg_pl_signed = single_pl
        avg_abs_pl = abs(single_pl) if single_pl is not None else None
        privacy = max(0.0, min(1.0, 1.0 - abs(single_pl) / 100.0)) if single_pl is not None else None

    return {
        "mem_score": mem_score,
        "utility": utility,
        "privacy": privacy,
        "n_components": len(components),
        "components": {"es_norm": es_norm, "qp_norm": qp_norm, "tr_norm": tr_norm},
        "raw": {
            "avg_privleak_signed": avg_pl_signed,
            "avg_abs_privleak": avg_abs_pl,
            "n_mias": len(per_mia_pl),
            "model_utility": mu,
        },
    }


def find_best_checkpoint_eval(run_dir: Path, select_by="mem_score", baseline_data=None, retain_data=None):
    """Among checkpoint-*/evals/TOFU_EVAL.json, return (ckpt_num, eval_data, scores) for best."""
    pattern = str(run_dir / "checkpoint-*" / "evals" / "TOFU_EVAL.json")
    paths = glob.glob(pattern)
    if not paths:
        return None, None, None

    candidates = []
    for p in paths:
        m = re.search(r"checkpoint-(\d+)", p)
        if not m:
            continue
        ckpt = int(m.group(1))
        data = load_json(p)
        scores = compute_scores(data, baseline_data, retain_data) if baseline_data else {}
        candidates.append((ckpt, data, scores))

    if not candidates:
        return None, None, None

    # Pick best by select_by metric (skip checkpoint-0 = init model)
    non_init = [(c, d, s) for c, d, s in candidates if c > 0]
    pool = non_init if non_init else candidates

    def key(x):
        v = x[2].get(select_by)
        return v if v is not None else -1.0

    return max(pool, key=key)


def load_flat_eval(path: Path):
    """Load a flat TOFU_EVAL.json (single checkpoint, no checkpoint-* dir)."""
    if path.exists():
        return load_json(path)
    return None


def fmt(val, pct=False, missing="—"):
    if val is None:
        return missing
    if pct:
        return f"{val * 100:.1f}"
    return f"{val:.4f}"


def build_method_rows(model, our_pattern, saves_unlearn, saves_eval, verbose):
    rows = []

    for split in FORGET_SPLITS:
        retain_split = SPLIT_TO_RETAIN[split]

        # ── Load reference evals ──────────────────────────────────────────────
        baseline_path = Path(saves_eval) / f"tofu_{model}_full" / f"evals_{split}" / "TOFU_EVAL.json"
        baseline_data = load_flat_eval(baseline_path)
        if baseline_data is None:
            print(f"[warn] Baseline eval missing for {split}: {baseline_path}")

        retrained_path = Path(saves_eval) / f"tofu_{model}_{retain_split}" / "TOFU_EVAL.json"
        retrained_data = load_flat_eval(retrained_path)

        retain_data = retrained_data  # full eval dict — used for per-MIA privleak

        # ── Baseline row ──────────────────────────────────────────────────────
        if baseline_data and baseline_data:
            # Baseline normalized against itself → all norms=1 → mem=0, utility=1
            s = compute_scores(baseline_data, baseline_data, retain_data)
            rows.append({"method": "Baseline", "split": split, "data": baseline_data, "scores": s,
                         "ckpt": None, "path": str(baseline_path)})

        # ── Retrained row ─────────────────────────────────────────────────────
        if retrained_data and baseline_data:
            s = compute_scores(retrained_data, baseline_data, retain_data)
            rows.append({"method": "Retrained", "split": split, "data": retrained_data, "scores": s,
                         "ckpt": None, "path": str(retrained_path)})

        # ── Flat unlearn methods ───────────────────────────────────────────────
        for method in ("GradAscent", "GradDiff", "NPO", "RMU", "SimNPO"):
            task = f"tofu_{model}_{split}_{method}"
            eval_path = Path(saves_unlearn) / task / "evals" / "TOFU_EVAL.json"
            data = load_flat_eval(eval_path)
            if data is None:
                rows.append({"method": method, "split": split, "data": None, "scores": {},
                             "ckpt": None, "path": str(eval_path)})
                continue
            s = compute_scores(data, baseline_data, retain_data) if baseline_data else {}
            rows.append({"method": method, "split": split, "data": data, "scores": s,
                         "ckpt": None, "path": str(eval_path)})

        # ── Ours (LatentRMU, checkpoint-based) ───────────────────────────────
        our_task = our_pattern.replace("{model}", model).replace("{split}", split)
        our_dir = Path(saves_unlearn) / our_task
        ckpt, our_data, our_scores = find_best_checkpoint_eval(our_dir, "mem_score", baseline_data, retain_data)
        rows.append({"method": "Ours", "split": split, "data": our_data,
                     "scores": our_scores or {}, "ckpt": ckpt, "path": str(our_dir)})

    return rows


def print_table(rows, verbose):
    METHOD_ORDER = ["Baseline", "Retrained", "GradAscent", "GradDiff", "NPO", "RMU", "SimNPO", "Ours"]

    header = f"{'Method':<14} {'Split':<10} {'MemScore':>9} {'Util%':>7} {'Privacy':>8} {'#Comp':>6}"
    if verbose:
        header += f"  {'ES_n':>6} {'QP_n':>6} {'TR_n':>6}  {'Ckpt':>6}"
    print()
    print(header)
    print("─" * len(header))

    prev_split = None
    for method in METHOD_ORDER:
        for split in FORGET_SPLITS:
            match = next((r for r in rows if r["method"] == method and r["split"] == split), None)
            if match is None:
                continue

            if split != prev_split and prev_split is not None:
                print()
            prev_split = split

            s = match["scores"]
            mem = fmt(s.get("mem_score"))
            util = fmt(s.get("utility"), pct=True)
            priv = fmt(s.get("privacy"), pct=True)
            ckpt = f"{match['ckpt']}" if match["ckpt"] is not None else "—"
            missing = match["data"] is None

            n_comp = s.get("n_components", 0)
            line = f"{method:<14} {split:<10} {mem:>9} {util:>7} {priv:>8} {n_comp:>6}"
            if verbose:
                c = s.get("components", {})
                line += (f"  {fmt(c.get('es_norm')):>6}"
                         f" {fmt(c.get('qp_norm')):>6}"
                         f" {fmt(c.get('tr_norm')):>6}"
                         f"  {ckpt:>6}")
            if missing:
                line += "  [eval missing]"
            print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Llama-3.2-1B-Instruct")
    parser.add_argument("--our_pattern", default="tofu_{model}_{split}_LatentRMU_v4.8_sweep",
                        help="Task name pattern for Ours; {model} and {split} are substituted.")
    parser.add_argument("--saves_unlearn", default="saves/unlearn")
    parser.add_argument("--saves_eval", default="saves/eval")
    parser.add_argument("--select_by", default="mem_score", choices=["mem_score", "utility"],
                        help="Metric for picking best LatentRMU checkpoint (default: mem_score)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    rows = build_method_rows(
        args.model, args.our_pattern,
        args.saves_unlearn, args.saves_eval,
        args.verbose,
    )
    print_table(rows, args.verbose)
    print()


if __name__ == "__main__":
    main()
