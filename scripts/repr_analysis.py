"""
Multi-method latent representation analysis for LLM unlearning.

Compares forget-set and retain-set activations across any number of models
(base, retrained oracle, and unlearning methods) to show:

  1. Where each method steers the forget-set representations (PCA / t-SNE)
  2. Layer-wise similarity to the retrained oracle (which layers are steered)
  3. Distance to retrained on forget-set (proxy for forget_quality)
  4. Distance to retrained on retain-set (proxy for model_utility)
  5. Forget/retain cluster gap per method (proxy for privacy leakage / MIA)
  6. Centroid distance heatmap across all methods

Usage
-----
python scripts/repr_analysis.py --config analysis_configs/tofu_1b_forget10.json

or inline:
python scripts/repr_analysis.py \\
  --base_model      open-unlearning/tofu_Llama-3.2-1B-Instruct_full \\
  --retrained_model saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90 \\
  --models          GradAscent:saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradAscent \\
                    GradDiff:saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradDiff \\
                    NPO:saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO \\
                    RMU:saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_RMU \\
                    LatentRMU:saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_LatentRMU_v4.8 \\
  --forget_data     forget10 \\
  --retain_data     retain90 \\
  --layers          -1 16 8 \\
  --output_dir      analysis_out/repr_study_forget10

Config JSON format
------------------
{
  "base_model": "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
  "retrained_model": "saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90",
  "methods": [
    {"name": "GradAscent", "path": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradAscent"},
    {"name": "GradDiff",   "path": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_GradDiff"},
    {"name": "NPO",        "path": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO"},
    {"name": "RMU",        "path": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_RMU"},
    {"name": "SimNPO",     "path": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_SimNPO"},
    {"name": "LatentRMU",  "path": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_LatentRMU_v4.8"}
  ],
  "forget_data": "forget10",
  "retain_data": "retain90",
  "layers": [-1, 16, 8],
  "max_forget_samples": 200,
  "max_retain_samples": 200,
  "batch_size": 4,
  "output_dir": "analysis_out/repr_study_forget10"
}

Outputs
-------
  forget_pca.png             — PCA of forget-set reps, all models, last layer
  retain_pca.png             — PCA of retain-set reps, all models, last layer
  combined_pca.png           — Forget (•) and retain (▲) for each model in one plot
  layer_similarity.png       — Cosine similarity to retrained per layer × method (heatmap)
  forget_distance_bar.png    — Distance to retrained on forget set (forget_quality proxy)
  retain_distance_bar.png    — Distance to retrained on retain set (model_utility proxy)
  forget_retain_gap.png      — Within-method forget/retain cluster gap (privleak proxy)
  centroid_heatmap.png       — Pairwise centroid distances on forget set
  tsne_forget.png            — t-SNE on forget set (with --tsne)
  repr_metrics.json          — All computed distances
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
os.makedirs(os.environ.setdefault("MPLCONFIGDIR", "/tank/home/judy/.cache/matplotlib"), exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100

# ─────────────────────────────────────────────────────────────────────────────
# Color palette — enough for base + retrained + 8 methods
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = [
    "#2c7bb6",   # Base      — blue
    "#1a9641",   # Retrained — green (oracle)
    "#aec7e8",   # baseline 1 — muted sky blue
    "#ffbb78",   # baseline 2 — muted peach
    "#c5b0d5",   # baseline 3 — muted lavender
    "#98df8a",   # baseline 4 — muted sage
    "#c49c94",   # baseline 5 — muted rose
    "#dbdb8d",   # baseline 6 — muted yellow
    "#9edae5",   # baseline 7 — muted cyan
    "#f7b6d2",   # baseline 8 — muted pink
]

OURS_COLOR = "#d62728"   # vivid red — for LatentRMU / highlighted method


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(path_or_split: str, max_samples: int = 500) -> List[Dict]:
    """Return list of {"question": str, "answer": str} dicts."""
    if path_or_split and Path(path_or_split).exists():
        with open(path_or_split) as f:
            data = json.load(f)
        samples = []
        for item in data:
            q = item.get("question") or item.get("prompt") or item.get("input", "")
            a = item.get("answer") or item.get("output") or item.get("target", "")
            samples.append({"question": q, "answer": a})
        return samples[:max_samples]

    split = path_or_split if path_or_split else "forget10"
    print(f"Loading locuslab/TOFU split '{split}' from HuggingFace …")
    try:
        from datasets import load_dataset
        ds = load_dataset("locuslab/TOFU", split)
        split_key = "train" if "train" in ds else list(ds.keys())[0]
        samples = [{"question": r["question"], "answer": r["answer"]} for r in ds[split_key]]
        return samples[:max_samples]
    except Exception as e:
        raise RuntimeError(f"Could not load data for split '{split}': {e}")


def build_batches(
    samples: List[Dict],
    tokenizer: AutoTokenizer,
    batch_size: int,
    device: str,
) -> List[Tuple]:
    batches = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        questions = [s["question"] for s in chunk]
        answers   = [s["answer"]   for s in chunk]

        full_texts = [f"{q} {a}" for q, a in zip(questions, answers)]
        full_enc = tokenizer(full_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=512)
        q_enc = tokenizer(questions, return_tensors="pt", padding=True,
                          truncation=True, max_length=512)

        labels = full_enc["input_ids"].clone()
        for i in range(len(chunk)):
            q_len = (q_enc["input_ids"][i] != tokenizer.pad_token_id).sum().item()
            labels[i, :q_len] = IGNORE_INDEX
        labels[full_enc["attention_mask"] == 0] = IGNORE_INDEX

        batches.append({
            "input_ids":      full_enc["input_ids"].to(device),
            "attention_mask": full_enc["attention_mask"].to(device),
            "labels":         labels.to(device),
        })
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str, device: str, dtype: torch.dtype) -> Tuple:
    print(f"  Loading: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, device_map=device,
    )
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Activation extraction — multiple layers at once
# ─────────────────────────────────────────────────────────────────────────────

def extract_activations_multilayer(
    model: AutoModelForCausalLM,
    batches: List[Dict],
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """
    Extract mean-pooled answer-token hidden states for each requested layer.

    Returns {layer_idx: np.ndarray of shape (N, D)}.
    Negative layer indices are resolved against the actual number of layers.
    """
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer

    def resolve(l: int) -> int:
        return l if l >= 0 else n_layers + l

    resolved = [resolve(l) for l in layers]

    store: Dict[int, List[np.ndarray]] = {l: [] for l in resolved}

    for batch in batches:
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels         = batch["labels"]

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)

        for layer_idx in resolved:
            hidden = out.hidden_states[layer_idx]  # (B, T, D)
            for i in range(hidden.shape[0]):
                answer_mask = (labels[i] != IGNORE_INDEX)
                if answer_mask.sum() == 0:
                    answer_mask = attention_mask[i].bool()
                vecs = hidden[i][answer_mask]
                pooled = vecs.mean(dim=0).cpu().float().numpy()
                store[layer_idx].append(pooled)

    return {l: np.stack(vs, axis=0) for l, vs in store.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Distance utilities
# ─────────────────────────────────────────────────────────────────────────────

def cosine_dist_mean(a: np.ndarray, b: np.ndarray) -> float:
    at = torch.from_numpy(a)
    bt = torch.from_numpy(b)
    return (1.0 - F.cosine_similarity(at, bt, dim=-1)).mean().item()


def l2_dist_mean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=-1).mean())


def centroid_cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    ca = torch.from_numpy(a.mean(axis=0, keepdims=True))
    cb = torch.from_numpy(b.mean(axis=0, keepdims=True))
    return (1.0 - F.cosine_similarity(ca, cb, dim=-1)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_pca(
    act_dict: Dict[str, np.ndarray],
    colors: Dict[str, str],
    title: str,
    out_path: Path,
    markers: Optional[Dict[str, str]] = None,
    highlight: Optional[str] = None,
) -> PCA:
    all_acts = np.concatenate(list(act_dict.values()), axis=0)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(all_acts)

    # Draw order: baselines first, then Retrained, then OURS on top.
    draw_names = [n for n in act_dict if n not in ("Finetuned",)]
    draw_names.sort(key=lambda n: (n == "Retrained", bool(highlight and n == highlight)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in draw_names:
        acts   = act_dict[name]
        proj   = pca.transform(acts)
        is_ref = name == "Retrained"
        is_hl  = bool(highlight and name == highlight)

        if is_hl:
            m, zord, size, alpha = "*", 12, 250, 0.95
            edge, lw = "black", 1.0
            c = OURS_COLOR
        elif is_ref:
            m = markers.get(name, "P") if markers else "P"
            zord, size, alpha = 10, 90, 0.9
            edge, lw = "black", 0.8
            c = colors[name]
        else:
            m = markers.get(name, "o") if markers else "o"
            zord, size, alpha = 2, 30, 0.65
            edge, lw = "none", 0.5
            c = colors[name]

        label = "OURS" if name == "LatentRMU" else name
        ax.scatter(proj[:, 0], proj[:, 1], c=c, marker=m,
                   label=label, alpha=alpha, edgecolors=edge,
                   linewidths=lw, s=size, zorder=zord)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return pca


def plot_forget_pca(
    model_acts_forget: Dict[str, np.ndarray],
    colors: Dict[str, str],
    out_dir: Path,
    highlight: Optional[str] = None,
) -> None:
    _scatter_pca(
        model_acts_forget, colors,
        "Forget-set Representations — PCA (last layer)",
        out_dir / "forget_pca.png",
        highlight=highlight,
    )


def plot_retain_pca(
    model_acts_retain: Dict[str, np.ndarray],
    colors: Dict[str, str],
    out_dir: Path,
) -> None:
    _scatter_pca(
        model_acts_retain, colors,
        "Retain-set Representations — PCA (last layer)",
        out_dir / "retain_pca.png",
    )


def plot_combined_pca(
    model_acts_forget: Dict[str, np.ndarray],
    model_acts_retain: Dict[str, np.ndarray],
    colors: Dict[str, str],
    out_dir: Path,
    highlight: Optional[str] = None,
) -> None:
    """Single PCA fitted on all vectors; forget=circle, retain=triangle."""
    all_acts = np.concatenate(
        list(model_acts_forget.values()) + list(model_acts_retain.values()), axis=0
    )
    pca = PCA(n_components=2, random_state=0)
    pca.fit(all_acts)

    fig, ax = plt.subplots(figsize=(9, 6))
    legend_handles = []
    for name in model_acts_forget:
        c = colors[name]
        pf = pca.transform(model_acts_forget[name])
        pr = pca.transform(model_acts_retain[name])
        zord = 5 if (highlight and name == highlight) else 2
        size_f = 50 if (highlight and name == highlight) else 30
        size_r = 50 if (highlight and name == highlight) else 30
        ax.scatter(pf[:, 0], pf[:, 1], c=c, marker="o", alpha=0.65, s=size_f,
                   edgecolors="black" if (highlight and name == highlight) else "none",
                   linewidths=0.6, zorder=zord)
        ax.scatter(pr[:, 0], pr[:, 1], c=c, marker="^", alpha=0.35, s=size_r,
                   zorder=zord)
        legend_handles.append(mpatches.Patch(color=c, label=name))

    # Marker legend
    from matplotlib.lines import Line2D
    legend_handles += [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=7, label="Forget set"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
               markersize=7, alpha=0.5, label="Retain set"),
    ]

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=11)
    ax.set_title("Forget (●) and Retain (▲) Representations — PCA", fontsize=12)
    ax.legend(handles=legend_handles, fontsize=8, loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / "combined_pca.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_layer_similarity(
    model_layer_acts: Dict[str, Dict[int, np.ndarray]],
    retrained_name: str,
    resolved_layers: List[int],
    colors: Dict[str, str],
    out_dir: Path,
) -> None:
    """
    Heatmap: rows = methods (excluding retrained), cols = layers.
    Cell = mean cosine similarity of that method's forget-set reps to retrained's.
    """
    retrained_acts = model_layer_acts.get(retrained_name, {})
    method_names = [n for n in model_layer_acts if n != retrained_name]
    if not retrained_acts or not method_names:
        return

    matrix = np.zeros((len(method_names), len(resolved_layers)))
    for i, name in enumerate(method_names):
        for j, layer in enumerate(resolved_layers):
            a = model_layer_acts[name].get(layer)
            b = retrained_acts.get(layer)
            if a is not None and b is not None:
                n = min(len(a), len(b))
                # Cosine similarity (higher = more similar to retrained)
                at = torch.from_numpy(a[:n])
                bt = torch.from_numpy(b[:n])
                sim = F.cosine_similarity(at, bt, dim=-1).mean().item()
                matrix[i, j] = sim

    fig, ax = plt.subplots(figsize=(max(6, len(resolved_layers) * 1.2), max(4, len(method_names) * 0.65)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.7, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Cosine Similarity to Retrained")

    layer_labels = [f"L{l}" for l in resolved_layers]
    ax.set_xticks(range(len(resolved_layers)))
    ax.set_xticklabels(layer_labels, fontsize=10)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    ax.set_title(f"Layer-wise Cosine Similarity to {retrained_name} (forget set)", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "layer_similarity.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_distance_bar(
    distances: Dict[str, float],
    colors: Dict[str, str],
    title: str,
    ylabel: str,
    out_path: Path,
    highlight: Optional[str] = None,
) -> None:
    names = list(distances.keys())
    vals  = [distances[n] for n in names]
    bar_colors = [colors.get(n, "#888888") for n in names]
    edge_colors = ["black" if (highlight and n == highlight) else "none" for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.1), 4))
    bars = ax.bar(range(len(names)), vals, color=bar_colors, edgecolor=edge_colors, linewidth=1.2)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_centroid_heatmap(
    model_acts: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Pairwise centroid cosine distances between all models on forget set."""
    names = list(model_acts.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i != j:
                matrix[i, j] = centroid_cosine_dist(model_acts[na], model_acts[nb])

    fig, ax = plt.subplots(figsize=(max(5, n * 0.9), max(4, n * 0.7)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Centroid Cosine Distance")

    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=7)

    ax.set_title("Pairwise Centroid Distances — Forget Set", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "centroid_heatmap.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_forget_retain_gap(
    model_acts_forget: Dict[str, np.ndarray],
    model_acts_retain: Dict[str, np.ndarray],
    colors: Dict[str, str],
    out_dir: Path,
    highlight: Optional[str] = None,
) -> None:
    """
    For each model: cosine distance between the forget-set centroid and
    retain-set centroid.  Smaller = forget reps look like retain reps
    (good privacy / hard to distinguish via MIA).
    """
    gaps = {}
    for name in model_acts_forget:
        if name in model_acts_retain:
            gaps[name] = centroid_cosine_dist(model_acts_forget[name],
                                              model_acts_retain[name])

    plot_distance_bar(
        gaps, colors,
        "Forget/Retain Centroid Gap (Cosine)\n↓ lower = forget reps blend with retain (better privacy)",
        "Centroid Cosine Distance",
        out_dir / "forget_retain_gap.png",
        highlight=highlight,
    )


def plot_tsne(
    model_acts: Dict[str, np.ndarray],
    colors: Dict[str, str],
    title: str,
    out_path: Path,
    highlight: Optional[str] = None,
) -> None:
    from sklearn.manifold import TSNE
    all_acts = np.concatenate(list(model_acts.values()), axis=0)
    n_each = {name: len(a) for name, a in model_acts.items()}
    perp = min(30, max(5, min(n_each.values()) - 1))
    proj = TSNE(n_components=2, perplexity=perp, random_state=0).fit_transform(all_acts)

    fig, ax = plt.subplots(figsize=(8, 6))
    offset = 0
    for name, acts in model_acts.items():
        n = len(acts)
        p = proj[offset : offset + n]
        offset += n
        zord = 5 if (highlight and name == highlight) else 2
        ax.scatter(p[:, 0], p[:, 1], c=colors[name], label=name, alpha=0.75,
                   edgecolors="black" if (highlight and name == highlight) else "none",
                   linewidths=0.8, s=40, zorder=zord)

    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# New targeted visualisations (use LatentRMU's focal layer)
# ─────────────────────────────────────────────────────────────────────────────

def plot_activation_trajectory(
    model_acts: Dict[str, np.ndarray],
    finetuned_name: str,
    retrained_name: str,
    colors: Dict[str, str],
    out_dir: Path,
    layer_label: str = "",
    highlight: Optional[str] = None,
    trajectory_methods: Optional[List[str]] = None,
) -> None:
    """
    Arrow-based trajectory plot: how each method shifts the forget-set centroid
    relative to the memorised (finetuned) baseline.

    Layout
    ------
    - Individual forget-set samples are shown as faded dots for density context.
    - Centroids are plotted as larger markers (◆ finetuned, ★ retrained, ● methods).
    - Arrows run from the Finetuned centroid to every other centroid.
      The Retrained oracle arrow is dashed; the highlighted method arrow is bold.
    - Axes are flipped after PCA so the Finetuned centroid sits in the
      bottom-left quadrant, making the "shift direction" visually consistent.

    Parameters
    ----------
    trajectory_methods : if given, only these model names are shown (plus
                         finetuned and retrained which are always included).
    """
    # Filter to requested methods (always keep Finetuned + Retrained).
    if trajectory_methods is not None:
        keep = set(trajectory_methods) | {finetuned_name, retrained_name}
        model_acts = {k: v for k, v in model_acts.items() if k in keep}

    if finetuned_name not in model_acts:
        print("  [skip trajectory] Finetuned model not found in activations.")
        return

    # Fit PCA on all samples from all models.
    all_acts = np.concatenate(list(model_acts.values()), axis=0)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(all_acts)

    # Project every model's samples.
    proj_samples  = {name: pca.transform(acts) for name, acts in model_acts.items()}
    proj_centroids = {name: p.mean(axis=0) for name, p in proj_samples.items()}

    # Flip axes so Finetuned centroid lands in the bottom-left quadrant.
    fc = proj_centroids[finetuned_name]
    flip = np.array([-1.0 if fc[0] > 0 else 1.0,
                     -1.0 if fc[1] > 0 else 1.0])
    proj_samples   = {k: v * flip for k, v in proj_samples.items()}
    proj_centroids = {k: v * flip for k, v in proj_centroids.items()}
    fc = proj_centroids[finetuned_name]

    fig, ax = plt.subplots(figsize=(9, 7))

    # ── Sample scatter (background density) ──────────────────────────────
    for name, pts in proj_samples.items():
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=colors.get(name, "#aaa"), alpha=0.13, s=18,
                   edgecolors="none", zorder=1)

    # ── Arrows: Finetuned centroid → each other centroid ─────────────────
    drawn_order = []  # draw non-highlighted first, highlighted last
    for name in proj_centroids:
        if name == finetuned_name:
            continue
        is_ret = name == retrained_name
        is_hl  = bool(highlight and name == highlight)
        drawn_order.append((is_ret or is_hl, name))
    drawn_order.sort(key=lambda t: t[0])   # False before True

    for _, name in drawn_order:
        c   = proj_centroids[name]
        is_ret = name == retrained_name
        is_hl  = bool(highlight and name == highlight)

        arrow_kw = dict(
            arrowstyle="-|>",
            color=colors.get(name, "#aaa"),
            lw=2.8 if (is_hl or is_ret) else 1.6,
            linestyle="dashed" if is_ret else "solid",
            mutation_scale=20 if (is_hl or is_ret) else 13,
        )
        ax.annotate("", xy=(c[0], c[1]), xytext=(fc[0], fc[1]),
                    arrowprops=arrow_kw,
                    zorder=5 if (is_hl or is_ret) else 3)

    # ── Centroids ─────────────────────────────────────────────────────────
    for name, c in proj_centroids.items():
        is_ret = name == retrained_name
        is_ft  = name == finetuned_name
        is_hl  = bool(highlight and name == highlight)

        marker = "*" if is_ret else ("D" if is_ft else "o")
        size   = 350 if is_ret else (200 if is_ft else (180 if is_hl else 110))
        lw_e   = 1.5 if (is_hl or is_ret or is_ft) else 0.8
        # Highlighted method centroid draws above Finetuned/Retrained markers.
        zord   = 9 if is_hl else 7

        ax.scatter(c[0], c[1],
                   c=colors.get(name, "#aaa"), marker=marker,
                   s=size, edgecolors="black", linewidths=lw_e,
                   zorder=zord, label=name)

        ax.annotate(name, xy=(c[0], c[1]),
                    xytext=(0, 9), textcoords="offset points",
                    fontsize=8, ha="center", va="bottom",
                    fontweight="bold" if is_hl else "normal",
                    color=colors.get(name, "#333"),
                    zorder=8)

    var = pca.explained_variance_ratio_
    layer_str = f" (Layer {layer_label})" if layer_label else ""
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=11)
    ax.set_title(
        f"Activation Trajectory — Forget Set{layer_str}\n"
        "Arrows from Finetuned centroid (◆)  ·  ★ = Retrained oracle  ·  dashed = oracle direction",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85,
              markerscale=0.7, handlelength=1.2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path = out_dir / "activation_trajectory.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_delta_pca(
    model_acts: Dict[str, np.ndarray],
    finetuned_name: str,
    retrained_name: str,
    colors: Dict[str, str],
    out_dir: Path,
    highlight: Optional[str] = None,
) -> None:
    """
    PCA on delta vectors: (method - finetuned) per sample.

    Removes shared linguistic structure and shows only what each method
    changed relative to the memorised baseline.  The Retrained delta is
    the target direction; methods close to it are doing the right thing.
    """
    base_acts = model_acts.get(finetuned_name)
    if base_acts is None:
        print("  [skip delta_pca] Finetuned model not found.")
        return

    delta_dict: Dict[str, np.ndarray] = {}
    for name, acts in model_acts.items():
        n = min(len(acts), len(base_acts))
        delta_dict[name] = acts[:n] - base_acts[:n]

    all_deltas = np.concatenate(list(delta_dict.values()), axis=0)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(all_deltas)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, delta in delta_dict.items():
        proj = pca.transform(delta)
        is_hl   = highlight and name == highlight
        is_ref  = name in (finetuned_name, retrained_name)
        marker  = "X" if is_ref else "o"
        size    = 80 if is_ref else (60 if is_hl else 30)
        alpha   = 0.9 if is_ref else (0.85 if is_hl else 0.55)
        edge    = "black" if is_hl else "none"
        # Highlighted method draws above everything (including reference X markers).
        zord    = 8 if is_hl else (6 if is_ref else 2)
        ax.scatter(proj[:, 0], proj[:, 1], c=colors[name], marker=marker,
                   label=name, alpha=alpha, edgecolors=edge,
                   linewidths=0.8, s=size, zorder=zord)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=11)
    ax.set_title("Δ-Representation PCA  (method − finetuned)\n"
                 "Methods near Retrained (✕) steer forget-set reps toward the oracle",
                 fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path = out_dir / "delta_pca.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_forget_direction(
    model_acts: Dict[str, np.ndarray],
    finetuned_name: str,
    retrained_name: str,
    colors: Dict[str, str],
    out_dir: Path,
    highlight: Optional[str] = None,
) -> None:
    """
    Project each method's per-sample delta onto the oracle direction
    (retrained_centroid − finetuned_centroid), normalised to unit length.

    Positive projection = moved toward the oracle.
    Higher bar = better alignment with the target unlearning direction.
    """
    base_acts      = model_acts.get(finetuned_name)
    retrained_acts = model_acts.get(retrained_name)
    if base_acts is None or retrained_acts is None:
        print("  [skip forget_direction] Need both Finetuned and Retrained.")
        return

    oracle_dir = retrained_acts.mean(axis=0) - base_acts.mean(axis=0)
    norm = np.linalg.norm(oracle_dir)
    if norm < 1e-8:
        print("  [skip forget_direction] Oracle direction is zero-length.")
        return
    oracle_dir = oracle_dir / norm  # unit vector

    projections = {}
    for name, acts in model_acts.items():
        n = min(len(acts), len(base_acts))
        delta = acts[:n] - base_acts[:n]               # (N, D)
        proj  = delta @ oracle_dir                     # (N,) — scalar per sample
        projections[name] = float(proj.mean())

    # Plot all methods except Finetuned (its projection is ~0 by construction)
    plot_dict = {n: v for n, v in projections.items() if n != finetuned_name}

    names      = list(plot_dict.keys())
    vals       = [plot_dict[n] for n in names]
    bar_colors = [colors.get(n, "#888888") for n in names]
    edge_c     = ["black" if (highlight and n == highlight) else "none" for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.1), 4))
    bars = ax.bar(range(len(names)), vals, color=bar_colors,
                  edgecolor=edge_c, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean Projection onto Oracle Direction", fontsize=10)
    ax.set_title("Forget-Direction Projection\n"
                 "↑ higher = forget-set reps moved more toward the retrained oracle",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar, v in zip(bars, vals):
        ypos = bar.get_height() + abs(max(vals) - min(vals)) * 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "forget_direction_proj.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _rms_norm(acts: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rms = np.sqrt((acts ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    return acts / rms * weight


def plot_logit_lens(
    model_lm_heads: Dict[str, Tuple],
    model_layer_acts: Dict[str, Dict[int, np.ndarray]],
    resolved_layers: List[int],
    colors: Dict[str, str],
    out_dir: Path,
    latent_layer: Optional[int] = None,
    highlight: Optional[str] = None,
) -> None:
    """
    Project hidden states at each layer through the LM head and compute the
    entropy of the resulting token distribution on forget-set samples.

    ↑ higher entropy = model is more uncertain about the memorised answer
    = better forgetting at that layer.

    A vertical dashed line marks LatentRMU's focal layer.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for name in model_layer_acts:
        if name not in model_lm_heads:
            continue
        lm_head_w, norm_w, _ = model_lm_heads[name]   # (V,D), (D,), None

        layer_acts = model_layer_acts[name]
        sorted_layers = sorted(l for l in resolved_layers if l in layer_acts)
        if not sorted_layers:
            continue

        ys = []
        for layer in sorted_layers:
            acts = layer_acts[layer].copy()          # (N, D)
            if norm_w is not None:
                acts = _rms_norm(acts, norm_w)
            logits = acts @ lm_head_w.T              # (N, V)
            # numerically stable softmax
            logits -= logits.max(axis=-1, keepdims=True)
            exp_l  = np.exp(logits)
            probs  = exp_l / exp_l.sum(axis=-1, keepdims=True)
            entropy = -(probs * np.log(probs + 1e-10)).sum(axis=-1)  # (N,)
            ys.append(float(entropy.mean()))

        lw   = 2.5 if (highlight and name == highlight) else 1.5
        zord = 5   if (highlight and name == highlight) else 2
        ax.plot(sorted_layers, ys, color=colors[name], label=name,
                linewidth=lw, zorder=zord, marker="o", markersize=4)

    if latent_layer is not None and latent_layer in resolved_layers:
        ax.axvline(latent_layer, color="gray", linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"LatentRMU layer (L{latent_layer})")

    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Mean Entropy (nats)", fontsize=11)
    ax.set_title("Logit-Lens Entropy on Forget Set\n"
                 "↑ higher = model more uncertain about memorised answer",
                 fontsize=12)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "logit_lens_entropy.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics(
    model_acts_forget: Dict[str, np.ndarray],
    model_acts_retain: Dict[str, np.ndarray],
    retrained_name: str,
) -> Dict:
    retrained_forget = model_acts_forget.get(retrained_name)
    retrained_retain = model_acts_retain.get(retrained_name)
    out = {}

    for name in model_acts_forget:
        af = model_acts_forget[name]
        ar = model_acts_retain.get(name)

        row = {}
        if retrained_forget is not None:
            n = min(len(af), len(retrained_forget))
            row["forget_cosine_dist_to_retrained"] = cosine_dist_mean(af[:n], retrained_forget[:n])
            row["forget_l2_dist_to_retrained"]     = l2_dist_mean(af[:n], retrained_forget[:n])
            row["forget_centroid_cosine_to_retrained"] = centroid_cosine_dist(af, retrained_forget)

        if ar is not None and retrained_retain is not None:
            n = min(len(ar), len(retrained_retain))
            row["retain_cosine_dist_to_retrained"] = cosine_dist_mean(ar[:n], retrained_retain[:n])
            row["retain_l2_dist_to_retrained"]     = l2_dist_mean(ar[:n], retrained_retain[:n])
            row["retain_centroid_cosine_to_retrained"] = centroid_cosine_dist(ar, retrained_retain)

        if ar is not None:
            row["forget_retain_centroid_gap"] = centroid_cosine_dist(af, ar)

        out[name] = row

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Activation cache  (avoids re-loading large models on repeated runs)
# ─────────────────────────────────────────────────────────────────────────────

def _acts_cache_dir(out_dir: Path) -> Path:
    d = out_dir / ".acts_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_acts_cache(
    cache_dir: Path,
    name: str,
    forget_acts: Dict[int, np.ndarray],
    retain_acts: Dict[int, np.ndarray],
    lm_head:     Tuple,
    model_path:  str,
    layers:      List[int],
) -> None:
    slug = name.replace("/", "_")
    np.savez(cache_dir / f"{slug}_forget.npz",
             **{str(k): v for k, v in forget_acts.items()})
    np.savez(cache_dir / f"{slug}_retain.npz",
             **{str(k): v for k, v in retain_acts.items()})
    lm_w, norm_w, norm_b = lm_head
    np.savez(cache_dir / f"{slug}_lmhead.npz",
             lm_w=lm_w,
             norm_w=(norm_w if norm_w is not None else np.array([])),
             norm_b=(norm_b if norm_b is not None else np.array([])))
    with open(cache_dir / f"{slug}_meta.json", "w") as f:
        json.dump({"path": model_path, "layers": sorted(layers)}, f)
    print(f"  Cached activations → {cache_dir}/{slug}_*.npz")


def _load_acts_cache(
    cache_dir: Path,
    name: str,
    model_path: str,
    layers: List[int],
) -> Optional[Tuple]:
    slug = name.replace("/", "_")
    meta_p = cache_dir / f"{slug}_meta.json"
    if not meta_p.exists():
        return None
    with open(meta_p) as f:
        meta = json.load(f)
    if meta.get("path") != model_path or meta.get("layers") != sorted(layers):
        print(f"  Cache miss for [{name}] (path or layers changed).")
        return None
    for suffix in ("forget", "retain", "lmhead"):
        if not (cache_dir / f"{slug}_{suffix}.npz").exists():
            return None

    def _load_layer_npz(p: Path) -> Dict[int, np.ndarray]:
        d = np.load(p)
        return {int(k): d[k] for k in d.files}

    forget_acts = _load_layer_npz(cache_dir / f"{slug}_forget.npz")
    retain_acts = _load_layer_npz(cache_dir / f"{slug}_retain.npz")
    lm_data     = np.load(cache_dir / f"{slug}_lmhead.npz")
    norm_w = lm_data["norm_w"] if lm_data["norm_w"].size > 0 else None
    norm_b = lm_data["norm_b"] if lm_data["norm_b"].size > 0 else None
    lm_head = (lm_data["lm_w"], norm_w, norm_b)
    print(f"  Loaded cached activations for [{name}].")
    return forget_acts, retain_acts, lm_head


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="JSON config file (preferred).")
    p.add_argument("--base_model",      default=None)
    p.add_argument("--retrained_model", default=None)
    p.add_argument("--models", nargs="+", default=[],
                   help="name:path pairs, e.g. GradDiff:saves/unlearn/...")
    p.add_argument("--forget_data", default="forget10")
    p.add_argument("--retain_data",  default="retain90")
    p.add_argument("--max_forget_samples", type=int, default=200)
    p.add_argument("--max_retain_samples", type=int, default=200)
    p.add_argument("--layers", nargs="+", type=int, default=[-1, 16, 8],
                   help="Layer indices (negative OK).")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--output_dir",  default="analysis_out/repr_study")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",       default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--tsne",          action="store_true")
    p.add_argument("--highlight",     default=None,
                   help="Name of the focal method to visually highlight.")
    p.add_argument("--latent_layer",  type=int, default=8,
                   help="Hidden-state index of LatentRMU's steering layer "
                        "(model.layers.7 → index 8 for Llama-3.2-1B).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _dtype(s):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load config ────────────────────────────────────────────────────────
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)

    base_model_path      = cfg.get("base_model",      args.base_model)
    retrained_model_path = cfg.get("retrained_model", args.retrained_model)
    forget_data          = cfg.get("forget_data",     args.forget_data)
    retain_data          = cfg.get("retain_data",     args.retain_data)
    max_forget           = cfg.get("max_forget_samples", args.max_forget_samples)
    max_retain           = cfg.get("max_retain_samples", args.max_retain_samples)
    layers               = list(cfg.get("layers",          args.layers))
    batch_size           = cfg.get("batch_size",      args.batch_size)
    out_dir              = Path(cfg.get("output_dir", args.output_dir))
    highlight            = cfg.get("highlight",       args.highlight)
    dtype                = _dtype(cfg.get("dtype", args.dtype))
    latent_layer         = cfg.get("latent_layer",    args.latent_layer)

    # Ensure latent_layer is always extracted (it may not be in the config list).
    if latent_layer not in layers:
        layers.append(latent_layer)

    # Build model list: base + retrained + all methods
    method_list = []
    if base_model_path:
        method_list.append(("Finetuned", base_model_path))
    if retrained_model_path:
        method_list.append(("Retrained", retrained_model_path))

    # From config JSON
    for m in cfg.get("methods", []):
        method_list.append((m["name"], m["path"]))

    # From CLI --models
    for item in args.models:
        name, path = item.split(":", 1)
        method_list.append((name, path))

    retrained_name = "Retrained"

    if not method_list:
        print("No models specified. Use --config or --base_model / --models.", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading forget data …")
    forget_samples = load_samples(forget_data, max_forget)
    print(f"  {len(forget_samples)} forget samples")

    print("Loading retain data …")
    retain_samples = load_samples(retain_data, max_retain)
    print(f"  {len(retain_samples)} retain samples")

    # ── Build tokenizer and batches from first model's tokenizer ──────────
    print(f"\nBuilding batches using tokenizer from: {method_list[0][1]}")
    tok = AutoTokenizer.from_pretrained(method_list[0][1], use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    forget_batches = build_batches(forget_samples, tok, batch_size, args.device)
    retain_batches = build_batches(retain_samples, tok, batch_size, args.device)
    del tok

    # ── Resolve layer indices ──────────────────────────────────────────────
    # We need a model to know n_layers; resolve lazily on first model load.
    resolved_layers = None

    # ── Run all models ─────────────────────────────────────────────────────
    model_acts_forget: Dict[str, Dict[int, np.ndarray]] = {}
    model_acts_retain: Dict[str, Dict[int, np.ndarray]] = {}
    model_lm_heads:    Dict[str, Tuple]                  = {}

    cache_dir = _acts_cache_dir(out_dir)

    for name, path in method_list:
        print(f"\n{'='*60}\n[{name}]\n{'='*60}")

        # ── Try cache first ──────────────────────────────────────────────
        cached = _load_acts_cache(cache_dir, name, path, layers)
        if cached is not None:
            model_acts_forget[name], model_acts_retain[name], model_lm_heads[name] = cached
            # Resolve layers from cache keys on first hit.
            if resolved_layers is None:
                resolved_layers = sorted(model_acts_forget[name].keys())
                print(f"  Resolved layer indices (from cache): {resolved_layers}")
            continue

        model, _ = load_model(path, args.device, dtype)

        if resolved_layers is None:
            n_total = model.config.num_hidden_layers + 1
            resolved_layers = [l if l >= 0 else n_total + l for l in layers]
            print(f"  Resolved layer indices: {resolved_layers} (of {n_total} total)")

        print("  Extracting forget-set activations …")
        model_acts_forget[name] = extract_activations_multilayer(model, forget_batches, layers)
        print("  Extracting retain-set activations …")
        model_acts_retain[name] = extract_activations_multilayer(model, retain_batches, layers)

        # Save lm_head + final norm for logit-lens.
        lm_head_w = model.lm_head.weight.detach().cpu().float().numpy()
        base_model = getattr(model, "model", None)
        norm = getattr(base_model, "norm", None) if base_model is not None else None
        norm_w = norm.weight.detach().cpu().float().numpy() if norm is not None else None
        norm_b = getattr(norm, "bias", None)
        norm_b = norm_b.detach().cpu().float().numpy() if norm_b is not None else None
        model_lm_heads[name] = (lm_head_w, norm_w, norm_b)

        _save_acts_cache(cache_dir, name,
                         model_acts_forget[name], model_acts_retain[name],
                         model_lm_heads[name], path, resolved_layers)

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ── Pick the last resolved layer for 2D scatter plots ─────────────────
    last_layer = max(resolved_layers)  # highest resolved index = actual last hidden layer

    forget_last = {name: model_acts_forget[name][last_layer] for name in model_acts_forget}
    retain_last = {name: model_acts_retain[name][last_layer] for name in model_acts_retain}

    # ── Assign colors ─────────────────────────────────────────────────────
    all_names = list(model_acts_forget.keys())
    colors = {}
    palette_idx = 0
    for name in all_names:
        if name == "Finetuned":
            colors[name] = PALETTE[0]
        elif name == retrained_name:
            colors[name] = PALETTE[1]
        elif highlight and name == highlight:
            colors[name] = OURS_COLOR
        else:
            palette_idx += 2  # skip first two (Base, Retrained slots)
            colors[name] = PALETTE[min(palette_idx, len(PALETTE) - 1)]
            palette_idx += 1

    # ── Metrics ────────────────────────────────────────────────────────────
    print("\nComputing distance metrics …")
    metrics = build_metrics(forget_last, retain_last, retrained_name)

    metrics_path = out_dir / "repr_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # ── Visualise ──────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    plot_forget_pca(forget_last, colors, out_dir, highlight=highlight)
    plot_retain_pca(retain_last, colors, out_dir)
    plot_combined_pca(forget_last, retain_last, colors, out_dir, highlight=highlight)
    plot_centroid_heatmap(forget_last, out_dir)

    # Layer-wise similarity
    plot_layer_similarity(model_acts_forget, retrained_name, resolved_layers,
                          colors, out_dir)

    # Distance-to-retrained bar charts
    forget_dists = {
        name: metrics[name]["forget_centroid_cosine_to_retrained"]
        for name in metrics
        if "forget_centroid_cosine_to_retrained" in metrics[name] and name != retrained_name
    }
    retain_dists = {
        name: metrics[name]["retain_centroid_cosine_to_retrained"]
        for name in metrics
        if "retain_centroid_cosine_to_retrained" in metrics[name] and name != retrained_name
    }

    if forget_dists:
        plot_distance_bar(
            forget_dists, colors,
            "Forget-set Centroid Distance to Retrained\n↓ lower = better forgetting",
            "Centroid Cosine Distance",
            out_dir / "forget_distance_bar.png",
            highlight=highlight,
        )
    if retain_dists:
        plot_distance_bar(
            retain_dists, colors,
            "Retain-set Centroid Distance to Retrained\n↓ lower = better utility preservation",
            "Centroid Cosine Distance",
            out_dir / "retain_distance_bar.png",
            highlight=highlight,
        )

    plot_forget_retain_gap(forget_last, retain_last, colors, out_dir, highlight=highlight)

    # ── New targeted plots at LatentRMU's focal layer ──────────────────────
    finetuned_name = "Finetuned"
    latent_layer_resolved = latent_layer if latent_layer >= 0 else (resolved_layers[0] + latent_layer)

    forget_focal = {name: model_acts_forget[name][latent_layer_resolved]
                    for name in model_acts_forget
                    if latent_layer_resolved in model_acts_forget[name]}

    if forget_focal:
        print(f"\n  Using layer {latent_layer_resolved} (LatentRMU focal layer) for targeted plots …")
        plot_activation_trajectory(
            forget_focal, finetuned_name, retrained_name,
            colors, out_dir,
            layer_label=str(latent_layer_resolved),
            highlight=highlight,
            trajectory_methods=[retrained_name, "RMU", "SimNPO", highlight]
            if highlight else None,
        )
        plot_delta_pca(forget_focal, finetuned_name, retrained_name,
                       colors, out_dir, highlight=highlight)
        plot_forget_direction(forget_focal, finetuned_name, retrained_name,
                              colors, out_dir, highlight=highlight)

    plot_logit_lens(model_lm_heads, model_acts_forget, resolved_layers,
                    colors, out_dir, latent_layer=latent_layer_resolved,
                    highlight=highlight)

    if args.tsne:
        plot_tsne(forget_last, colors,
                  "Forget-set Representations — t-SNE (last layer)",
                  out_dir / "tsne_forget.png", highlight=highlight)
        plot_tsne(retain_last, colors,
                  "Retain-set Representations — t-SNE (last layer)",
                  out_dir / "tsne_retain.png", highlight=highlight)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"{'Model':20s}  {'FQ proxy':10s}  {'MU proxy':10s}  {'PL proxy':10s}")
    print(f"{'':20s}  {'(↓ fgt→ret)':10s}  {'(↓ ret→ret)':10s}  {'(↓ fgt-ret)':10s}")
    print("-" * 62)
    for name, row in metrics.items():
        fq = row.get("forget_centroid_cosine_to_retrained", float("nan"))
        mu = row.get("retain_centroid_cosine_to_retrained", float("nan"))
        pl = row.get("forget_retain_centroid_gap", float("nan"))
        print(f"  {name:18s}  {fq:10.4f}  {mu:10.4f}  {pl:10.4f}")
    print("=" * 62)
    print(f"\nAll outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
