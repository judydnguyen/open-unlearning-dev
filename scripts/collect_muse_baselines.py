#!/usr/bin/env python
"""Collect every MUSE_SUMMARY.json under saves/ and print one table per split.

Layout assumptions:
- Reference evals (target = pre-unlearn, retrain = gold) live at:
    saves/eval/muse_{model}_{split}_target/MUSE_SUMMARY.json
    saves/eval/muse_{model}_{split}_retrain/MUSE_SUMMARY.json
- Unlearn runs live at:
    saves/unlearn/{task_name}/evals/MUSE_SUMMARY.json                # final
    saves/unlearn/{task_name}/evals/{checkpoint-N|last}/MUSE_SUMMARY.json
  Task names follow muse_{model}_{split}_{method}{tag}.

Usage:
    python scripts/collect_muse_baselines.py
    python scripts/collect_muse_baselines.py --model Llama-2-7b-hf
    python scripts/collect_muse_baselines.py --splits News
    python scripts/collect_muse_baselines.py --include-checkpoints   # show per-ckpt rows
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
SAVES = REPO / "saves"

# Canonical column ordering: forget signals first, then retain, then privacy/MIA,
# then anything else we discover. Any key not listed here is appended at the end.
CANONICAL_ORDER = [
    "forget_knowmem_ROUGE",
    "forget_verbmem_ROUGE",
    "retain_knowmem_ROUGE",
    "exact_memorization",
    "extraction_strength",
    "privleak",
    "mia_loss",
    "mia_gradnorm",
    "mia_zlib",
    "mia_min_k",
    "mia_min_k_plus_plus",
    "mia_reference",
]
# Short labels for narrower headers; defaults to the field name otherwise.
SHORT = {
    "forget_knowmem_ROUGE":  "f_know",
    "forget_verbmem_ROUGE":  "f_verb",
    "retain_knowmem_ROUGE":  "r_know",
    "exact_memorization":    "ex_mem",
    "extraction_strength":   "extr_str",
    "privleak":              "privlk",
    "mia_loss":              "mia_loss",
    "mia_gradnorm":          "mia_gn",
    "mia_zlib":              "mia_zlib",
    "mia_min_k":             "mia_mnk",
    "mia_min_k_plus_plus":   "mia_mnk+",
    "mia_reference":         "mia_ref",
}


def load_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"_error": str(e)}


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def parse_task(name: str, model: str) -> tuple[Optional[str], Optional[str]]:
    """Return (split, method_label) or (None, None) if not parseable."""
    m = re.match(rf"^muse_{re.escape(model)}_(Books|News)_(.+)$", name)
    if not m:
        return None, None
    split, method = m.group(1), m.group(2)
    return split, method


def collect(model: str, splits: list[str], include_checkpoints: bool) -> dict[str, list[dict]]:
    """Returns {split: [row, ...]} where row keys = label, source, **metrics."""
    rows: dict[str, list[dict]] = {s: [] for s in splits}

    # Reference evals (target / retrain)
    for split in splits:
        for ref in ("target", "retrain"):
            p = SAVES / "eval" / f"muse_{model}_{split}_{ref}" / "MUSE_SUMMARY.json"
            if p.exists():
                row = {"label": f"[{ref}]", "source": str(p.relative_to(REPO))}
                row.update(load_summary(p))
                rows[split].append(row)

    # Unlearn runs
    unlearn_root = SAVES / "unlearn"
    if not unlearn_root.exists():
        return rows

    for task_dir in sorted(unlearn_root.iterdir()):
        if not task_dir.is_dir():
            continue
        split, method = parse_task(task_dir.name, model)
        if split is None or split not in splits:
            continue

        evals = task_dir / "evals"
        if not evals.exists():
            continue

        # Top-level summary (final eval, usually 'last')
        top = evals / "MUSE_SUMMARY.json"
        if top.exists():
            row = {"label": method, "source": str(top.relative_to(REPO))}
            row.update(load_summary(top))
            rows[split].append(row)

        # Per-checkpoint summaries
        if include_checkpoints:
            for sub in sorted(evals.iterdir()):
                if not sub.is_dir():
                    continue
                p = sub / "MUSE_SUMMARY.json"
                if not p.exists():
                    continue
                row = {"label": f"{method}/{sub.name}", "source": str(p.relative_to(REPO))}
                row.update(load_summary(p))
                rows[split].append(row)

    return rows


def discover_columns(entries: list[dict]) -> list[str]:
    """Return every metric key seen in any entry, ordered canonically first."""
    seen: set[str] = set()
    for r in entries:
        for k, v in r.items():
            if k in {"label", "source"}:
                continue
            if isinstance(v, (int, float)):
                seen.add(k)
    ordered = [c for c in CANONICAL_ORDER if c in seen]
    extras = sorted(c for c in seen if c not in CANONICAL_ORDER)
    return ordered + extras


def render(rows: dict[str, list[dict]]) -> None:
    for split, entries in rows.items():
        if not entries:
            continue
        cols = discover_columns(entries)
        label_w = max(len("method"), max(len(r["label"]) for r in entries))
        col_w = {c: max(len(SHORT.get(c, c)), 8) for c in cols}

        header = f"{'method':<{label_w}}  " + "  ".join(
            f"{SHORT.get(c, c):>{col_w[c]}}" for c in cols
        )
        sep = "-" * len(header)
        print(f"\n=== MUSE / {split} ===")
        print(header)
        print(sep)
        # References first, then alphabetically
        entries.sort(key=lambda r: (not r["label"].startswith("["), r["label"].lower()))
        for r in entries:
            line = f"{r['label']:<{label_w}}  " + "  ".join(
                f"{fmt(r.get(c)):>{col_w[c]}}" for c in cols
            )
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama-2-7b-hf")
    ap.add_argument("--splits", nargs="+", default=["Books", "News"])
    ap.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include one row per intermediate checkpoint-* eval.",
    )
    ap.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Restrict to a subset of metric columns (use the JSON field names). "
             "Default: every metric found across all loaded summaries.",
    )
    args = ap.parse_args()

    rows = collect(args.model, args.splits, args.include_checkpoints)
    if args.columns:
        # Drop unrequested numeric fields so discover_columns picks only these.
        keep = {"label", "source", *args.columns}
        for split_rows in rows.values():
            for r in split_rows:
                for k in list(r.keys()):
                    if k not in keep and isinstance(r[k], (int, float)):
                        del r[k]
    render(rows)

    # Quick health note: warn if reference rows missing
    for split, entries in rows.items():
        labels = {r["label"] for r in entries}
        for ref in ("[target]", "[retrain]"):
            if ref not in labels:
                print(f"\n[warn] {split}: {ref} reference missing — comparisons are partial.")


if __name__ == "__main__":
    main()
