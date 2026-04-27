"""
Distributional shift analysis on forget-set samples across three model states.

  Base model      — pre-finetuning (no exposure to forget set; same weights as retrained)
  Finetuned model — after SFT on the forget set (memorization introduced)
  Unlearned model — after running an unlearning method (e.g. SteerGRPO)

Usage
-----
python scripts/analysis.py \
  --base_model      <path or HF id>          \
  --finetuned_model <path>                   \
  --unlearned_model <path>                   \
  --forget_data     data/tofu/forget01.json  \
  --output_dir      analysis_out/            \
  --batch_size      4                        \
  --layer           -1                       \
  --max_new_tokens  200                      \
  --device          cuda                     \
  --seed            42

forget_data JSON format: list of {"question": ..., "answer": ...} dicts.
If omitted, falls back to HuggingFace locuslab/TOFU forget10 split.

Outputs
-------
  metrics_summary.json   — mean ROUGE scores + pairwise distances
  rouge_scores.csv       — per-sample ROUGE for every model
  rouge_comparison.png   — boxplot comparing the three models
  activation_pca.png     — PCA scatter of hidden states
  distance_heatmap.png   — pairwise distance heatmap (cosine / l2 / wasserstein)
  activation_tsne.png    — t-SNE scatter (only with --tsne)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_forget_samples(path_or_split: Optional[str]) -> List[Dict]:
    """Return a list of {"question": str, "answer": str} dicts."""
    if path_or_split and Path(path_or_split).exists():
        with open(path_or_split) as f:
            data = json.load(f)
        # Normalise key names — some TOFU JSON exports use "question"/"answer"
        # and others might use "prompt"/"output".
        normalised = []
        for item in data:
            q = item.get("question") or item.get("prompt") or item.get("input", "")
            a = item.get("answer") or item.get("output") or item.get("target", "")
            normalised.append({"question": q, "answer": a})
        print(f"Loaded {len(normalised)} forget samples from {path_or_split}")
        return normalised

    # Fallback: HuggingFace locuslab/TOFU
    split = path_or_split if path_or_split else "forget10"
    print(f"Loading locuslab/TOFU split '{split}' from HuggingFace …")
    try:
        from datasets import load_dataset
        ds = load_dataset("locuslab/TOFU", split)
        # TOFU HF configs have a single "train" split, not the config name.
        split_key = "train" if "train" in ds else list(ds.keys())[0]
        samples = [{"question": r["question"], "answer": r["answer"]} for r in ds[split_key]]
        print(f"Loaded {len(samples)} samples from HuggingFace.")
        return samples
    except Exception as e:
        raise RuntimeError(
            f"Could not load forget data. Provide --forget_data <path.json> or ensure "
            f"the HuggingFace dataset is reachable. Original error: {e}"
        )


def make_batches(
    samples: List[Dict],
    tokenizer: AutoTokenizer,
    batch_size: int,
    device: str,
) -> List[Tuple]:
    """
    Yield (prompt_batch, full_batch, ground_truths) tuples.

    prompt_batch — tokenised question only  (for generation)
    full_batch   — tokenised question+answer (for activation extraction)
                   labels are -100 on the question tokens, answer tokens otherwise
    ground_truths — list of raw answer strings
    """
    batches = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        questions = [s["question"] for s in chunk]
        answers   = [s["answer"]   for s in chunk]

        # ── prompt-only tokenisation ──────────────────────────────────────
        prompt_enc = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        prompt_batch = {k: v.to(device) for k, v in prompt_enc.items()}

        # ── full (question + answer) tokenisation with answer labels ──────
        full_texts = [f"{q} {a}" for q, a in zip(questions, answers)]
        full_enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        q_enc = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        labels = full_enc["input_ids"].clone()
        for i in range(len(chunk)):
            # Mask out question tokens so we pool only on answer positions.
            q_len = (q_enc["input_ids"][i] != tokenizer.pad_token_id).sum().item()
            labels[i, :q_len] = IGNORE_INDEX
        # Mask padding.
        labels[full_enc["attention_mask"] == 0] = IGNORE_INDEX

        full_batch = {
            "input_ids":      full_enc["input_ids"].to(device),
            "attention_mask": full_enc["attention_mask"].to(device),
            "labels":         labels.to(device),
        }

        batches.append((prompt_batch, full_batch, answers))

    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    path: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal-LM and its tokenizer from a local path or HF hub ID."""
    print(f"Loading model from: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # for generation; right is fine for forward

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE
# ─────────────────────────────────────────────────────────────────────────────

def generate_outputs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batches: List[Tuple],
    max_new_tokens: int,
) -> List[str]:
    """Generate completions for all prompt batches; return flat list of strings."""
    generated = []
    for prompt_batch, _, _ in batches:
        input_ids      = prompt_batch["input_ids"]
        attention_mask = prompt_batch["attention_mask"]
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens.
        new_tokens = out[:, input_ids.shape[-1]:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        generated.extend(t.strip() for t in texts)
    return generated


def compute_rouge(
    generated_texts: List[str],
    ground_truths: List[str],
) -> List[Dict]:
    """Return per-sample ROUGE-1/2/L scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    results = []
    for gen, gt in zip(generated_texts, ground_truths):
        scores = scorer.score(gt, gen)
        results.append(
            {
                "rouge1_r":  scores["rouge1"].recall,
                "rouge1_f1": scores["rouge1"].fmeasure,
                "rouge2_r":  scores["rouge2"].recall,
                "rouge2_f1": scores["rouge2"].fmeasure,
                "rougeL_r":  scores["rougeL"].recall,
                "rougeL_f1": scores["rougeL"].fmeasure,
            }
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Activations
# ─────────────────────────────────────────────────────────────────────────────

def extract_activations(
    model: AutoModelForCausalLM,
    batches: List[Tuple],
    layer: int,
) -> np.ndarray:
    """
    Extract mean-pooled hidden states from `layer` over answer token positions.

    Returns shape (N, hidden_dim) as a numpy array.
    """
    all_vecs = []
    for _, full_batch, _ in batches:
        input_ids      = full_batch["input_ids"]
        attention_mask = full_batch["attention_mask"]
        labels         = full_batch["labels"]

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden = out.hidden_states[layer]  # (B, T, D)

        for i in range(hidden.shape[0]):
            # Pool over positions where labels != IGNORE_INDEX (answer tokens).
            answer_mask = (labels[i] != IGNORE_INDEX)  # (T,)
            if answer_mask.sum() == 0:
                # Fallback: pool over all non-padding positions.
                answer_mask = attention_mask[i].bool()

            vecs = hidden[i][answer_mask]          # (n_ans_tokens, D)
            pooled = vecs.mean(dim=0).cpu().float().numpy()
            all_vecs.append(pooled)

    return np.stack(all_vecs, axis=0)  # (N, D)


# ─────────────────────────────────────────────────────────────────────────────
# Distances
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_distances(acts_a: np.ndarray, acts_b: np.ndarray) -> Dict:
    """
    Compute three pairwise distributional distance measures between two
    sets of activation vectors (N, D).

    cosine      — mean sample-wise cosine distance (1 − cosine_similarity)
    l2          — mean sample-wise L2 distance
    wasserstein — mean per-dimension Wasserstein-1 distance
    """
    a_t = torch.from_numpy(acts_a)
    b_t = torch.from_numpy(acts_b)

    cosine_sim  = F.cosine_similarity(a_t, b_t, dim=-1)          # (N,)
    cosine_dist = (1.0 - cosine_sim).mean().item()

    l2_dist = torch.norm(a_t - b_t, dim=-1).mean().item()

    D = acts_a.shape[1]
    w_dists = [wasserstein_distance(acts_a[:, d], acts_b[:, d]) for d in range(D)]
    wasserstein = float(np.mean(w_dists))

    return {
        "cosine":      cosine_dist,
        "l2":          l2_dist,
        "wasserstein": wasserstein,
    }


def compute_all_distances(
    acts_base: np.ndarray,
    acts_ft:   np.ndarray,
    acts_ul:   np.ndarray,
) -> Dict:
    return {
        "base_vs_finetuned": compute_pairwise_distances(acts_base, acts_ft),
        "base_vs_unlearned": compute_pairwise_distances(acts_base, acts_ul),
        "finetuned_vs_unlearned": compute_pairwise_distances(acts_ft, acts_ul),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Base":      "#4878cf",
    "Finetuned": "#d65f5f",
    "Unlearned": "#6acc65",
}

# Extra palette slots for additional methods beyond "Unlearned"
_EXTRA_PALETTE = [
    "#e6a817",  # amber
    "#9d4edd",  # purple
    "#2ec4b6",  # teal
    "#f77f00",  # orange
    "#e63946",  # crimson
]


def _color_for(name: str, extra_idx: int) -> str:
    if name in COLORS:
        return COLORS[name]
    return _EXTRA_PALETTE[extra_idx % len(_EXTRA_PALETTE)]


def _build_color_map(names: List[str]) -> Dict[str, str]:
    color_map: Dict[str, str] = {}
    extra_idx = 0
    for name in names:
        color_map[name] = _color_for(name, extra_idx)
        if name not in COLORS:
            extra_idx += 1
    return color_map


def visualize_rouge(
    model_rouge: Dict[str, List[Dict]],
    out_dir:     Path,
) -> None:
    """Boxplot comparing ROUGE scores across all models."""
    metrics    = ["rouge1_r", "rouge2_r", "rougeL_f1", "rougeL_r"]
    names      = list(model_rouge.keys())
    color_map  = _build_color_map(names)
    n_methods  = len(names)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(14, 3 * n_methods), 4))

    for ax, metric in zip(axes, metrics):
        data = [[r[metric] for r in model_rouge[n]] for n in names]
        bp   = ax.boxplot(data, patch_artist=True,
                          widths=0.55, medianprops=dict(color="black", linewidth=1.5))
        for patch, name in zip(bp["boxes"], names):
            patch.set_facecolor(color_map[name])
            patch.set_alpha(0.75)
        labels = ["Ours" if n == "LatentRMU" else n for n in names]
        ax.set_xticks(range(1, n_methods + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric.replace("_", " ").upper(), fontsize=10)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("ROUGE Score Comparison across Methods", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "rouge_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def visualize_rouge_histograms(
    model_rouge: Dict[str, List[Dict]],
    out_dir:     Path,
    bins:        int = 20,
) -> None:
    """Overlapping histograms of per-sample ROUGE scores for all models."""
    metrics   = ["rouge1_r", "rouge2_r", "rougeL_f1", "rougeL_r"]
    color_map = _build_color_map(list(model_rouge.keys()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        all_vals  = [v for rows in model_rouge.values() for v in [r[metric] for r in rows]]
        lo, hi    = min(all_vals), max(all_vals)
        if hi - lo < 1e-6:
            lo, hi = lo - 0.05, hi + 0.05
        bin_edges = np.linspace(lo, hi, bins + 1)

        for name, rows in model_rouge.items():
            vals = [r[metric] for r in rows]
            ax.hist(vals, bins=bin_edges, alpha=0.45,
                    color=color_map[name], label=name,
                    edgecolor="white", linewidth=0.4)
            ax.axvline(float(np.mean(vals)), color=color_map[name],
                       linewidth=1.8, linestyle="--", alpha=0.9)

        ax.set_title(metric.replace("_", " ").upper(), fontsize=10)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("ROUGE Score Distributions across Methods\n(dashed lines = means)",
                 fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "rouge_histograms.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def visualize_activations(
    model_acts: Dict[str, np.ndarray],
    out_dir:    Path,
    tsne:       bool = False,
) -> None:
    """PCA (and optionally t-SNE) scatter of activation vectors.

    model_acts — ordered dict mapping display name → (N, D) activations.
    """
    names    = list(model_acts.keys())
    all_acts = np.concatenate(list(model_acts.values()), axis=0)

    # Assign colors: fixed for Base/Finetuned, palette for the rest.
    extra_idx = 0
    color_map: Dict[str, str] = {}
    for name in names:
        color_map[name] = _color_for(name, extra_idx)
        if name not in COLORS:
            extra_idx += 1

    # ── PCA ──────────────────────────────────────────────────────
    n_components = min(2, all_acts.shape[1], all_acts.shape[0])
    pca = PCA(n_components=n_components, random_state=0)
    proj_all = pca.fit_transform(all_acts)
    if proj_all.shape[1] == 1:
        proj_all = np.column_stack([proj_all, np.zeros(len(proj_all))])

    fig, ax = plt.subplots(figsize=(7, 5))
    offset = 0
    for name, acts in model_acts.items():
        n = len(acts)
        p = proj_all[offset : offset + n]
        offset += n
        ax.scatter(p[:, 0], p[:, 1],
                   c=color_map[name], label=name,
                   alpha=0.7, edgecolors="none", s=40)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)" if len(var) > 1 else "PC2")
    ax.set_title("Activation Space — PCA Projection")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "activation_pca.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── t-SNE (optional) ─────────────────────────────────────────
    if tsne:
        from sklearn.manifold import TSNE
        N_min = min(len(a) for a in model_acts.values())
        perp  = min(30, max(5, N_min - 1))
        tsne_proj = TSNE(n_components=2, perplexity=perp, random_state=0).fit_transform(all_acts)
        fig, ax = plt.subplots(figsize=(7, 5))
        offset = 0
        for name, acts in model_acts.items():
            n = len(acts)
            p = tsne_proj[offset : offset + n]
            offset += n
            ax.scatter(p[:, 0], p[:, 1],
                       c=color_map[name], label=name,
                       alpha=0.7, edgecolors="none", s=40)
        ax.set_title("Activation Space — t-SNE Projection")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / "activation_tsne.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


def visualize_distances(distance_dict: Dict, out_dir: Path) -> None:
    """Heatmap of pairwise distances for each distance metric."""
    pair_labels = ["base_vs_finetuned", "base_vs_unlearned", "finetuned_vs_unlearned"]
    display_labels = ["Base vs FT", "Base vs UL", "FT vs UL"]
    dist_metrics = ["cosine", "l2", "wasserstein"]

    # Build a (3, 3) matrix: rows = pairs, cols = distance metrics
    matrix = np.array(
        [[distance_dict[p][m] for m in dist_metrics] for p in pair_labels],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(dist_metrics)))
    ax.set_xticklabels([m.capitalize() for m in dist_metrics])
    ax.set_yticks(range(len(display_labels)))
    ax.set_yticklabels(display_labels)

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, f"{matrix[r, c]:.4f}", ha="center", va="center", fontsize=9)

    ax.set_title("Pairwise Activation Distances")
    fig.tight_layout()
    out_path = out_dir / "distance_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics persistence
# ─────────────────────────────────────────────────────────────────────────────

def _mean_rouge(rouge_list: List[Dict]) -> Dict:
    if not rouge_list:
        return {}
    keys = rouge_list[0].keys()
    return {k: float(np.mean([r[k] for r in rouge_list])) for k in keys}


def _col(name: str, metric: str) -> str:
    """CSV column name: {model_name}_{metric}, e.g. GradAscent_rouge1_r."""
    return f"{name}_{metric}"


def save_metrics(
    model_rouge: Dict[str, List[Dict]],
    model_gen:   Dict[str, List[str]],
    distance_dict: Dict,
    samples: List[Dict],
    out_dir: Path,
) -> None:
    names = list(model_rouge.keys())
    rouge_keys = list(next(iter(model_rouge.values()))[0].keys()) if model_rouge else []

    # ── metrics_summary.json ──────────────────────────────────────
    summary = {
        "rouge":                {n: _mean_rouge(model_rouge[n]) for n in names},
        "activation_distances": distance_dict,
    }
    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # ── rouge_scores.csv ─────────────────────────────────────────
    csv_path = out_dir / "rouge_scores.csv"
    fieldnames = (
        ["idx", "question", "answer"]
        + [_col(n, k) for n in names for k in rouge_keys]
        + [f"{n}_gen" for n in names]
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        n_rows = len(samples)
        for i in range(n_rows):
            row: Dict = {
                "idx":      i,
                "question": samples[i]["question"],
                "answer":   samples[i]["answer"],
            }
            for name in names:
                for k in rouge_keys:
                    row[_col(name, k)] = model_rouge[name][i][k]
                row[f"{name}_gen"] = model_gen[name][i] if model_gen.get(name) else ""
            writer.writerow(row)
    print(f"Saved: {csv_path}")


def load_rouge_from_csv(
    csv_path: Path,
    names: List[str],
) -> Optional[Dict[str, List[Dict]]]:
    """Load per-model ROUGE lists from an existing rouge_scores.csv.

    Returns None if the CSV is missing or doesn't contain columns for all names.
    """
    if not csv_path.exists():
        return None
    import csv as csv_mod
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)
    if not rows:
        return None

    rouge_keys = ["rouge1_r", "rouge1_f1", "rouge2_r", "rouge2_f1", "rougeL_r", "rougeL_f1"]
    result: Dict[str, List[Dict]] = {}
    for name in names:
        # Accept both "{name}_{metric}" (new) and legacy prefixes for Base/Finetuned/Unlearned.
        legacy = {"Base": "base", "Finetuned": "finetuned", "Unlearned": "unlearned"}
        prefix = legacy.get(name, name)
        col0 = f"{prefix}_{rouge_keys[0]}"
        if col0 not in rows[0]:
            print(f"  [rouge_scores.csv] No columns for '{name}' — will regenerate.")
            return None
        result[name] = [
            {k: float(row[f"{prefix}_{k}"]) for k in rouge_keys}
            for row in rows
        ]
    print(f"  Loaded ROUGE scores from {csv_path}  (skipping generation).")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributional shift analysis on forget-set samples."
    )
    parser.add_argument("--base_model",      required=True, help="Path or HF ID of the base model.")
    parser.add_argument("--finetuned_model", required=True, help="Path to finetuned model checkpoint.")
    # Legacy single-method argument (kept for backward compatibility).
    parser.add_argument("--unlearned_model", default=None, help="Path to unlearned model checkpoint.")
    # Multi-method: one or more 'Name:path' pairs (e.g. RMU:saves/unlearn/... SimNPO:saves/unlearn/...)
    parser.add_argument("--methods", nargs="+", default=[],
                        metavar="NAME:PATH",
                        help="Additional unlearned models as name:path pairs.")
    parser.add_argument("--forget_data",     default=None,
                        help="JSON file with forget samples or TOFU split name (default: forget10).")
    parser.add_argument("--output_dir",      default="analysis_out", help="Directory to save results.")
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--layer",           type=int,   default=-1,
                        help="Hidden layer index for activation extraction (-1 = last).")
    parser.add_argument("--max_new_tokens",  type=int,   default=200)
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype",           default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--tsne",            action="store_true", help="Also produce t-SNE plot.")
    parser.add_argument("--load_sequential", action="store_true",
                        help="Load one model at a time to reduce peak GPU memory.")
    return parser.parse_args()


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def main():
    args  = parse_args()
    dtype = _dtype_from_str(args.dtype)

    # Reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    samples = load_forget_samples(args.forget_data)
    if not samples:
        print("No forget samples found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Build batches once using the base model tokenizer (shared across all).
    print("\nBuilding batches …")
    _, base_tokenizer = load_model(args.base_model, "cpu", dtype)
    batches = make_batches(samples, base_tokenizer, args.batch_size, args.device)
    del base_tokenizer

    all_ground_truths = []
    for _, _, gts in batches:
        all_ground_truths.extend(gts)

    # ── Build model list ──────────────────────────────────────────────────
    model_configs = [
        ("Base",      args.base_model),
        ("Finetuned", args.finetuned_model),
    ]
    # Legacy --unlearned_model
    if args.unlearned_model:
        model_configs.append(("Unlearned", args.unlearned_model))
    # --methods NAME:path ...
    for item in args.methods:
        name, path = item.split(":", 1)
        model_configs.append((name, path))

    if len(model_configs) < 3:
        print("Provide at least one unlearned model via --unlearned_model or --methods.",
              file=sys.stderr)
        sys.exit(1)

    # ── Check for existing ROUGE CSV (skip generation if present) ────────
    all_names  = [name for name, _ in model_configs]
    csv_path   = out_dir / "rouge_scores.csv"
    cached_rouge = load_rouge_from_csv(csv_path, all_names)

    # ── Run models ────────────────────────────────────────────────────────
    results = {}
    for name, path in model_configs:
        print(f"\n{'='*60}\n[{name}] {path}\n{'='*60}")
        model, tokenizer = load_model(path, args.device, dtype)

        if cached_rouge is not None:
            rouge_scores = cached_rouge[name]
            gen_texts    = []
        else:
            gen_texts    = generate_outputs(model, tokenizer, batches, args.max_new_tokens)
            rouge_scores = compute_rouge(gen_texts, all_ground_truths)

        acts = extract_activations(model, batches, args.layer)

        results[name] = {
            "gen":   gen_texts,
            "rouge": rouge_scores,
            "acts":  acts,
        }

        del model, tokenizer
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Use the first unlearned model for the 3-way distance table (backward compat).
    ul_name = model_configs[2][0]

    # ── Distances ─────────────────────────────────────────────────────────
    print("\nComputing pairwise activation distances …")
    distance_dict = compute_all_distances(
        results["Base"]["acts"],
        results["Finetuned"]["acts"],
        results[ul_name]["acts"],
    )

    # ── Save metrics (only if we ran generation) ──────────────────────────
    model_rouge = {name: results[name]["rouge"] for name, _ in model_configs}
    if cached_rouge is None:
        model_gen = {name: results[name]["gen"] for name, _ in model_configs}
        save_metrics(model_rouge, model_gen, distance_dict, samples, out_dir)

    # ── Visualise ─────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    visualize_rouge(model_rouge, out_dir)
    visualize_rouge_histograms(model_rouge, out_dir)
    model_acts = {name: results[name]["acts"] for name, _ in model_configs}
    visualize_activations(model_acts, out_dir, tsne=args.tsne)
    visualize_distances(distance_dict, out_dir)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, _ in model_configs:
        mean = _mean_rouge(results[name]["rouge"])
        print(
            f"  {name:14s} | "
            f"ROUGE-1r={mean.get('rouge1_r', 0):.3f}  "
            f"ROUGE-2r={mean.get('rouge2_r', 0):.3f}  "
            f"ROUGE-Lf={mean.get('rougeL_f1', 0):.3f}"
        )
    print()
    for pair, dists in distance_dict.items():
        print(f"  {pair}")
        for m, v in dists.items():
            print(f"    {m:>12s}: {v:.6f}")
    print()
    print(f"All outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
