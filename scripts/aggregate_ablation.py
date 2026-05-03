"""Aggregate FLOUR Tier-1 ablation results into a CSV + Markdown table.

Reads saves/unlearn/tofu_*_LatentRMU_v4.8_abl_<name>/best/evals/TOFU_EVAL.json
Produces:
    analysis_out/ablation_main_table.csv
    analysis_out/ablation_main_table.md      (paper-ready)

Each row reports: variant, raw key metrics, and composite scores
(Mem-Progress capped at 1.0, Util normalized by retrain, Privacy from 4-MIA).

Usage:
    python scripts/aggregate_ablation.py                          # forget05 only
    python scripts/aggregate_ablation.py --split forget05         # explicit
"""

import argparse
import glob
import json
import re
from pathlib import Path
from statistics import mean

try:
    from scipy.stats import hmean as _hmean
    def hmean(vals): return float(_hmean([max(v, 1e-10) for v in vals]))
except ImportError:
    def hmean(vals): return len(vals) / sum(1.0 / max(v, 1e-10) for v in vals)


MIA_KEYS = ("mia_loss", "mia_zlib", "mia_min_k", "mia_min_k_plus_plus")


def g(d, k):
    if d is None: return None
    e = d.get(k)
    if e is None: return None
    return e.get("agg_value") if isinstance(e, dict) else float(e)


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def progress(x_method, x_b, x_r, lower_is_better=True):
    if x_method is None or x_b is None or x_r is None: return None
    denom = x_b - x_r if lower_is_better else x_r - x_b
    if abs(denom) < 1e-9: return None
    return (x_b - x_method) / denom if lower_is_better else (x_method - x_b) / denom


def compute(d, b, ref):
    """Per-method composite metrics under retrain-normalized scoring."""
    if d is None or b is None or ref is None:
        return None
    pe = progress(g(d, "extraction_strength"), g(b, "extraction_strength"), g(ref, "extraction_strength"), True)
    pq = progress(g(d, "forget_Q_A_Prob"), g(b, "forget_Q_A_Prob"), g(ref, "forget_Q_A_Prob"), True)
    pt = progress(g(d, "forget_truth_ratio"), g(b, "forget_truth_ratio"), g(ref, "forget_truth_ratio"), False)
    cp = [max(0.0, min(1.0, p)) for p in (pe, pq, pt) if p is not None]
    mem = hmean(cp) if cp else 0.0
    mu, ru = g(d, "model_utility"), g(ref, "model_utility")
    util = min(mu / ru, 1.0) if (mu and ru and ru > 0) else 0.0
    pls = []
    for k in MIA_KEYS:
        s, r = g(d, k), g(ref, k)
        if s is None or r is None: continue
        denom = 1 - r
        if abs(denom) < 1e-9: continue
        pls.append((r - s) / denom * 100)
    avg_abs = sum(abs(p) for p in pls) / len(pls) if pls else None
    priv = max(0.0, 1.0 - avg_abs / 100) if avg_abs is not None else None
    return {
        "mem": mem, "util": util, "priv": priv,
        "raw_mu": g(d, "model_utility"), "raw_es": g(d, "extraction_strength"),
        "raw_qa": g(d, "forget_Q_A_Prob"), "raw_tr": g(d, "forget_truth_ratio"),
        "raw_pl": g(d, "privleak"), "n_mia": len(pls),
    }


# Axis grouping for the paper table
AXIS_GROUPS = [
    ("A — Hook layer (default = layer 7)", [
        ("layer_3",  "layer 3"),  ("layer_5",  "layer 5"),
        ("default",  "**layer 7** (default)"),
        ("layer_9",  "layer 9"),  ("layer_11", "layer 11"), ("layer_13", "layer 13"),
    ]),
    ("B — Encoder design", [
        ("default",     "full encoder + orth (default)"),
        ("no_encoder",  "no encoder (random direction)"),
        ("no_orth",     "encoder w/o orthogonality reg"),
    ]),
    ("C — Steering coefficient (default = 8)", [
        ("coeff_2",  "coeff = 2"),  ("coeff_4",  "coeff = 4"),
        ("coeff_6",  "coeff = 6"),
        ("default",  "**coeff = 8** (default)"),
        ("coeff_10", "coeff = 10"), ("coeff_12", "coeff = 12"),
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forget05")
    ap.add_argument("--out_dir", default="analysis_out")
    ap.add_argument("--model", default="Llama-3.2-1B-Instruct")
    args = ap.parse_args()

    retain = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}[args.split]
    b = load(f"saves/eval/tofu_{args.model}_full/evals_{args.split}/TOFU_EVAL.json")
    ref = load(f"saves/eval/tofu_{args.model}_{retain}/TOFU_EVAL.json")
    if b is None or ref is None:
        raise SystemExit(f"Missing baseline ({args.split}) or retain ({retain}) eval JSON.")

    pattern = f"saves/unlearn/tofu_{args.model}_{args.split}_LatentRMU_v4.8_abl_*/best/evals/TOFU_EVAL.json"
    found = {}
    for p in glob.glob(pattern):
        m = re.search(r"_abl_([A-Za-z0-9_]+)/best/", p)
        if not m: continue
        name = m.group(1)
        d = load(p)
        s = compute(d, b, ref)
        if s is not None:
            s["hm"] = hmean([s["mem"], s["util"], s["priv"]]) if s["priv"] is not None else None
            found[name] = s

    if not found:
        raise SystemExit(f"No ablation evals found matching {pattern}")

    # CSV — one row per variant, all variants
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_main_table.csv"
    headers = ["variant", "split", "raw_mu", "raw_es", "raw_qa", "raw_tr", "raw_pl",
               "n_mia", "Mem", "Util", "Priv", "HM"]
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for name, s in sorted(found.items()):
            row = [name, args.split,
                   f"{s['raw_mu']:.4f}" if s.get('raw_mu') is not None else "",
                   f"{s['raw_es']:.4f}" if s.get('raw_es') is not None else "",
                   f"{s['raw_qa']:.4f}" if s.get('raw_qa') is not None else "",
                   f"{s['raw_tr']:.4f}" if s.get('raw_tr') is not None else "",
                   f"{s['raw_pl']:.2f}" if s.get('raw_pl') is not None else "",
                   str(s["n_mia"]),
                   f"{s['mem']:.4f}", f"{s['util']:.4f}",
                   f"{s['priv']:.4f}" if s['priv'] is not None else "",
                   f"{s['hm']:.4f}" if s.get("hm") is not None else ""]
            f.write(",".join(row) + "\n")
    print(f"Wrote {csv_path}  ({len(found)} variants)")

    # Markdown — paper-ready, axis-grouped
    md_path = out_dir / "ablation_main_table.md"
    lines = []
    lines.append(f"# FLOUR Tier-1 Ablation Results ({args.split})")
    lines.append("")
    lines.append("All scores in [0,1], higher = better. Composite **HM** = HM(Mem, Util, Priv).")
    lines.append("")
    for group_title, group in AXIS_GROUPS:
        lines.append(f"## {group_title}")
        lines.append("")
        lines.append("| Variant | Mem | Util | Priv | **HM** |")
        lines.append("|---|---|---|---|---|")
        for variant_key, display_name in group:
            s = found.get(variant_key)
            if s is None:
                lines.append(f"| {display_name} | — | — | — | — |")
                continue
            star = " ★" if s.get("hm") is not None and s["hm"] >= 0.85 else ""
            priv_s = f"{s['priv']:.3f}" if s['priv'] is not None else "—"
            hm_s   = f"{s['hm']:.4f}" if s.get("hm") is not None else "—"
            lines.append(f"| {display_name} | "
                         f"{s['mem']:.3f} | {s['util']:.3f} | "
                         f"{priv_s} | **{hm_s}**{star} |")
        lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")
    print()
    print("Preview:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
