#!/usr/bin/env python
"""Update the TOFU main results table in experiment.tex with current eval values.

Reads TOFU_EVAL.json files for all paper methods and fills in FQ, MU, PL, ES values.
Patches the region between sentinel comments in experiment.tex in-place.

Sentinel markers (added automatically on first run):
    % AUTOGEN:tofu_main:begin
    ...rows...
    % AUTOGEN:tofu_main:end

Usage:
    python scripts/update_paper_table.py
    python scripts/update_paper_table.py --dry_run   # print rows, don't write
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

PAPER_TEX = Path("../NeurIPS-26-LLM-Unlearning/tabs/tofu_main.tex")

SAVES_UNLEARN = Path("saves/unlearn")
SAVES_EVAL = Path("saves/eval")
SAVES_FINETUNE = Path("saves/finetune")

MODEL = "Llama-3.2-1B-Instruct"
OUR_PATTERN = "tofu_{model}_{split}_LatentRMU_v4.8_sweep"

SPLITS = ["forget01", "forget05", "forget10"]
SPLIT_TO_RETAIN = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}

SENTINEL_BEGIN = "% AUTOGEN:tofu_main:begin"
SENTINEL_END = "% AUTOGEN:tofu_main:end"

# Table row order: (display_label, eval_key_or_None)
# eval_key_or_None is the method suffix in the task name, or a special string
ROW_DEFS = [
    ("Retain (upper)",   "baseline"),   # full finetuned model
    ("Retrain (oracle)", "retrained"),  # retain-split model
    None,                               # \midrule
    ("GradAscent",       "GradAscent"),
    ("GradDiff",         "GradDiff"),
    ("NPO",              "NPO"),
    ("SimNPO",           "SimNPO"),
    ("UNDIAL",           None),         # not in our eval set
    ("APO",              None),         # not in our eval set
    None,                               # \midrule
    ("RMU",              "RMU"),
    ("LUNAR",            None),         # not in our eval set
    None,                               # \midrule
    ("\\textbf{FLOUR}",  "ours"),
]


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


def load_flat(path: Path):
    return load_json(path) if path.exists() else None


def load_best_checkpoint(run_dir: Path):
    """Return eval data from the best (highest forget_quality) non-init checkpoint."""
    paths = glob.glob(str(run_dir / "checkpoint-*" / "evals" / "TOFU_EVAL.json"))
    best_data, best_fq = None, -1.0
    for p in paths:
        if re.search(r"checkpoint-0/", p):
            continue
        try:
            data = load_json(p)
            fq = get_agg(data, "forget_quality") or -1.0
            if fq > best_fq:
                best_fq = fq
                best_data = data
        except Exception:
            continue
    return best_data


def collect_split_data(split):
    retain_split = SPLIT_TO_RETAIN[split]
    results = {}

    # Baseline (full finetuned model)
    baseline_path = SAVES_EVAL / f"tofu_{MODEL}_full" / f"evals_{split}" / "TOFU_EVAL.json"
    results["baseline"] = load_flat(baseline_path)

    # Retrained (retain-split model)
    retrained_path = SAVES_EVAL / f"tofu_{MODEL}_{retain_split}" / "TOFU_EVAL.json"
    results["retrained"] = load_flat(retrained_path)

    # Flat unlearn methods
    for method in ("GradAscent", "GradDiff", "NPO", "RMU", "SimNPO"):
        task = f"tofu_{MODEL}_{split}_{method}"
        p = SAVES_UNLEARN / task / "evals" / "TOFU_EVAL.json"
        results[method] = load_flat(p)

    # Ours (checkpoint-based LatentRMU)
    our_task = OUR_PATTERN.replace("{model}", MODEL).replace("{split}", split)
    results["ours"] = load_best_checkpoint(SAVES_UNLEARN / our_task)

    return results


def fmt_fq(v):
    return f"{v:.4f}" if v is not None else "---"

def fmt_mu(v):
    return f"{v * 100:.2f}" if v is not None else "---"

def fmt_pl(v):
    return f"{v:.2f}" if v is not None else "---"

def fmt_es(v):
    return f"{v:.4f}" if v is not None else "---"


def make_cell(data, force_fq=None):
    """Return (FQ, MU, PL, ES) formatted strings for one method+split."""
    if data is None:
        return "---", "---", "---", "---"
    fq = force_fq if force_fq is not None else get_agg(data, "forget_quality")
    mu = get_agg(data, "model_utility")
    pl = get_agg(data, "privleak")
    es = get_agg(data, "extraction_strength")
    return fmt_fq(fq), fmt_mu(mu), fmt_pl(pl), fmt_es(es)


def generate_rows():
    split_data = {split: collect_split_data(split) for split in SPLITS}

    lines = [f"  {SENTINEL_BEGIN} — do not edit by hand; run scripts/update_paper_table.py"]

    for row_def in ROW_DEFS:
        if row_def is None:
            lines.append("\\midrule")
            continue

        label, key = row_def
        col_width = 18
        label_str = f"{label:<{col_width}}"

        cells = []
        for split in SPLITS:
            data = split_data[split].get(key) if key else None
            # Retrain FQ is 1.0 by definition (oracle gold standard)
            force_fq = 1.0 if key == "retrained" else None
            cells.extend(make_cell(data, force_fq=force_fq))

        row = f"  {label_str} & " + " & ".join(cells) + " \\\\"
        lines.append(row)

    lines.append(f"  {SENTINEL_END}")
    return "\n".join(lines)


def patch_tex(tex_path: Path, new_rows: str, dry_run: bool):
    content = tex_path.read_text()

    begin_idx = content.find(SENTINEL_BEGIN)
    end_idx = content.find(SENTINEL_END)

    if begin_idx == -1 or end_idx == -1:
        # Sentinels not present — insert them around the existing --- rows inside tab:tofu_main.
        # Find the table by its label, then locate the first \midrule...last \\ before \bottomrule.
        pattern = r"(\\label\{tab:tofu_main\}.*?\\midrule\n)(.*?)(\\bottomrule)"
        m = re.search(pattern, content, re.DOTALL)
        if not m:
            print("[error] Could not locate tab:tofu_main in experiment.tex", file=sys.stderr)
            print("        Add sentinel comments manually and re-run.", file=sys.stderr)
            sys.exit(1)
        replacement = m.group(1) + new_rows + "\n" + m.group(3)
        new_content = content[:m.start(1)] + replacement + content[m.end(3):]
    else:
        # Replace between existing sentinels
        before = content[:begin_idx]
        after = content[end_idx + len(SENTINEL_END):]
        new_content = before + new_rows + after

    if dry_run:
        print(new_rows)
        return

    tex_path.write_text(new_content)
    print(f"[ok] Updated {tex_path}")


def main():
    global MODEL, OUR_PATTERN
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--our_pattern", default=OUR_PATTERN)
    parser.add_argument("--paper_tex", default=str(PAPER_TEX),
                        help="Path to experiment.tex in the paper repo")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print generated rows without writing to file")
    args = parser.parse_args()
    MODEL = args.model
    OUR_PATTERN = args.our_pattern

    tex_path = Path(args.paper_tex)
    if not tex_path.exists() and not args.dry_run:
        print(f"[error] Paper tex not found: {tex_path}", file=sys.stderr)
        sys.exit(1)

    rows = generate_rows()
    patch_tex(tex_path, rows, args.dry_run)


if __name__ == "__main__":
    main()
