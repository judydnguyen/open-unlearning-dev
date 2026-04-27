"""
Activation trajectory analysis for LLM unlearning.

Supports two kinds of models:
  --runs   name:run_dir   Trajectory: finds all checkpoint-N dirs with model
                          weights and plots a connected path (light→dark gradient).
  --models name:path      Single point: one final checkpoint, shown as a lone
                          centroid marker with an arrow from Finetuned.

Anchors (always shown):
  --base_model   finetuned (memorised) model  →  ◆ forced to bottom-left
  --retrained    retrained oracle             →  ★

Usage
-----
python scripts/traj_analysis.py \\
  --base_model saves/finetune/tofu_Llama-3.2-1B-Instruct_full \\
  --retrained  saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90 \\
  --runs    LatentRMU:saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_LatentRMU_sweep_v1 \\
  --models  RMU:saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_RMU \\
            SimNPO:saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_SimNPO/last \\
  --forget_data forget10 --latent_layer 8 \\
  --output_dir  analysis_out/traj_forget10
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
os.makedirs(os.environ.setdefault("MPLCONFIGDIR", "/tank/home/judy/.cache/matplotlib"), exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100

# Each trajectory method gets its own colormap (light→dark = early→late).
# Single-point methods get a fixed colour.
TRAJ_CMAPS = ["Blues", "Oranges", "Greens", "Purples", "Reds"]
SINGLE_COLORS = ["#e41a1c", "#ff7f00", "#984ea3", "#a65628"]

ANCHOR_COLORS = {
    "Finetuned": "#444444",
    "Retrained": "#1a9641",
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_samples(path_or_split: str, max_samples: int) -> List[Dict]:
    if path_or_split and Path(path_or_split).exists():
        with open(path_or_split) as f:
            data = json.load(f)
        out = []
        for item in data:
            q = item.get("question") or item.get("prompt") or ""
            a = item.get("answer") or item.get("output") or ""
            out.append({"question": q, "answer": a})
        return out[:max_samples]
    split = path_or_split or "forget10"
    print(f"Loading locuslab/TOFU '{split}' from HuggingFace …")
    from datasets import load_dataset
    ds = load_dataset("locuslab/TOFU", split)
    key = "train" if "train" in ds else list(ds.keys())[0]
    return [{"question": r["question"], "answer": r["answer"]} for r in ds[key]][:max_samples]


def build_batches(samples, tokenizer, batch_size, device):
    batches = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        questions = [s["question"] for s in chunk]
        answers   = [s["answer"]   for s in chunk]
        full_enc = tokenizer(
            [f"{q} {a}" for q, a in zip(questions, answers)],
            return_tensors="pt", padding=True, truncation=True, max_length=512,
        )
        q_enc = tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True, max_length=512,
        )
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


def load_model(path, device, dtype):
    print(f"  Loading: {path}")
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map=device)
    model.eval()
    return model, tok


def extract_samples(model, batches, layer_idx) -> np.ndarray:
    """Return mean-pooled answer-token hidden states, shape (N, D)."""
    vecs = []
    for batch in batches:
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
        hidden = out.hidden_states[layer_idx]
        for i in range(hidden.shape[0]):
            mask = batch["labels"][i] != IGNORE_INDEX
            if mask.sum() == 0:
                mask = batch["attention_mask"][i].bool()
            vecs.append(hidden[i][mask].mean(dim=0).cpu().float())
    return torch.stack(vecs).numpy()


# ── Checkpoint discovery ───────────────────────────────────────────────────────

def find_checkpoints(run_dir: Path, n: int) -> List[Tuple[int, Path]]:
    """Return up to n evenly-spaced checkpoints (by step) that have model weights."""
    pat = re.compile(r"^checkpoint-(\d+)$")
    found = []
    for d in sorted(run_dir.iterdir()):
        m = pat.match(d.name)
        if not m:
            continue
        if (d / "model.safetensors").exists() or list(d.glob("pytorch_model*.bin")):
            found.append((int(m.group(1)), d))
    found.sort(key=lambda t: t[0])

    # Include last/ as the final point if it exists and has weights.
    last_dir = run_dir / "last"
    if last_dir.exists() and (last_dir / "model.safetensors").exists():
        max_step = found[-1][0] if found else 0
        # Only add if it's not duplicating the final numbered checkpoint.
        if not found or max_step < 10_000:
            found.append((max_step + 1, last_dir))

    if not found:
        return []
    if len(found) <= n:
        return found

    indices = sorted({round(i * (len(found) - 1) / (n - 1)) for i in range(n)})
    return [found[i] for i in indices]


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_trajectory(
    anchor_acts: Dict[str, np.ndarray],
    traj_acts:   Dict[str, List[Tuple[int, np.ndarray]]],   # name→[(step, (N,D))]
    single_acts: Dict[str, np.ndarray],                      # name→(N,D)
    traj_cmaps:  Dict[str, str],
    single_colors: Dict[str, str],
    out_path: Path,
    layer_label: str = "",
) -> None:
    # ── Fit PCA on all data ───────────────────────────────────────────────
    all_chunks = list(anchor_acts.values()) + list(single_acts.values())
    for ckpt_list in traj_acts.values():
        for _, acts in ckpt_list:
            all_chunks.append(acts)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(np.concatenate(all_chunks, axis=0))

    # Compute centroids in PCA space.
    def centroid(acts):
        return pca.transform(acts).mean(axis=0)

    anchor_c = {k: centroid(v) for k, v in anchor_acts.items()}
    single_c = {k: centroid(v) for k, v in single_acts.items()}
    traj_c   = {name: [(s, centroid(a)) for s, a in ckpts]
                for name, ckpts in traj_acts.items()}

    # Flip axes so Finetuned sits at bottom-left.
    ft = anchor_c["Finetuned"]
    flip = np.array([-1.0 if ft[0] > 0 else 1.0,
                     -1.0 if ft[1] > 0 else 1.0])

    anchor_c = {k: v * flip for k, v in anchor_c.items()}
    single_c = {k: v * flip for k, v in single_c.items()}
    traj_c   = {n: [(s, c * flip) for s, c in ckpts] for n, ckpts in traj_c.items()}

    # Projected sample scatter (for background density).
    anchor_pts = {k: pca.transform(v) * flip for k, v in anchor_acts.items()}
    single_pts = {k: pca.transform(v) * flip for k, v in single_acts.items()}

    fig, ax = plt.subplots(figsize=(10, 8))

    # ── Faded background scatter ──────────────────────────────────────────
    for name, pts in anchor_pts.items():
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=ANCHOR_COLORS.get(name, "#aaa"), alpha=0.10, s=14,
                   edgecolors="none", zorder=1)
    for name, pts in single_pts.items():
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=single_colors.get(name, "#aaa"), alpha=0.10, s=14,
                   edgecolors="none", zorder=1)

    ft_c = anchor_c["Finetuned"]

    # ── Single-point methods: arrow from Finetuned + lone marker ─────────
    for name, c in single_c.items():
        color = single_colors.get(name, "#888")
        ax.annotate("", xy=(c[0], c[1]), xytext=(ft_c[0], ft_c[1]),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=2.0, mutation_scale=15), zorder=3)
        ax.scatter(c[0], c[1], c=color, s=160, marker="o",
                   edgecolors="black", linewidths=1.0, zorder=6, label=name)
        ax.annotate(name, xy=(c[0], c[1]),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=9, ha="center", color=color,
                    fontweight="bold", zorder=7)

    # ── Trajectory methods: gradient line + dots ──────────────────────────
    for name, ckpts in traj_c.items():
        cmap  = matplotlib.colormaps.get_cmap(traj_cmaps.get(name, "Blues"))
        n_ck  = len(ckpts)
        xs    = [c[0] for _, c in ckpts]
        ys    = [c[1] for _, c in ckpts]

        # Arrow from Finetuned to first checkpoint.
        ax.annotate("", xy=(xs[0], ys[0]), xytext=(ft_c[0], ft_c[1]),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=cmap(0.35), lw=1.6,
                                   linestyle="dashed", mutation_scale=13),
                    zorder=3)

        # Gradient line segments between checkpoints.
        for i in range(n_ck - 1):
            seg_color = cmap(0.35 + 0.6 * (i + 0.5) / max(n_ck - 1, 1))
            ax.annotate("", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="-|>", color=seg_color,
                                        lw=2.2, mutation_scale=14), zorder=4)

        # Dots: light → dark.
        for i, (step, c) in enumerate(ckpts):
            frac  = i / max(n_ck - 1, 1)
            color = cmap(0.35 + 0.6 * frac)
            size  = 100 if i in (0, n_ck - 1) else 55
            ax.scatter(c[0], c[1], color=color, s=size,
                       edgecolors="black", linewidths=0.8, zorder=5)
            if i == 0 or i == n_ck - 1:
                label_step = "start" if i == 0 else f"step {step}"
                ax.annotate(label_step, xy=(c[0], c[1]),
                            xytext=(0, 9), textcoords="offset points",
                            fontsize=7, ha="center",
                            color=cmap(0.35 + 0.6 * frac), zorder=6)

        # Legend entry using the dark colour.
        ax.plot([], [], color=cmap(0.85), linewidth=2.5,
                marker="o", markersize=6, label=name)

    # ── Anchor centroids ──────────────────────────────────────────────────
    for name, c in anchor_c.items():
        color  = ANCHOR_COLORS[name]
        marker = "*" if name == "Retrained" else "D"
        size   = 380 if name == "Retrained" else 220
        ax.scatter(c[0], c[1], c=color, marker=marker, s=size,
                   edgecolors="black", linewidths=1.5, zorder=8, label=name)
        ax.annotate(name, xy=(c[0], c[1]),
                    xytext=(0, 12), textcoords="offset points",
                    fontsize=9, ha="center", fontweight="bold",
                    color=color, zorder=9)

    var = pca.explained_variance_ratio_
    layer_str = f" — Layer {layer_label}" if layer_label else ""
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=11)
    ax.set_title(
        f"Activation Trajectory over Training{layer_str}\n"
        "LatentRMU: light→dark = early→late  ·  ◆ Finetuned (bottom-left)  ·  ★ Retrained oracle",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",    required=True)
    p.add_argument("--retrained",     required=True)
    p.add_argument("--runs",   nargs="*", default=[],
                   help="name:run_dir  — trajectory methods with checkpoint dirs.")
    p.add_argument("--models", nargs="*", default=[],
                   help="name:path     — single-checkpoint methods.")
    p.add_argument("--forget_data",   default="forget10")
    p.add_argument("--max_samples",   type=int, default=100)
    p.add_argument("--n_checkpoints", type=int, default=10)
    p.add_argument("--latent_layer",  type=int, default=8)
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--output_dir",    default="analysis_out/traj")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",         default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args  = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading forget data …")
    samples = load_samples(args.forget_data, args.max_samples)
    print(f"  {len(samples)} samples")

    _, tok = load_model(args.base_model, "cpu", dtype)
    batches = build_batches(samples, tok, args.batch_size, args.device)
    del tok

    layer = args.latent_layer

    # ── Anchors ───────────────────────────────────────────────────────────
    anchor_acts: Dict[str, np.ndarray] = {}
    for label, path in [("Finetuned", args.base_model), ("Retrained", args.retrained)]:
        print(f"\n[{label}]")
        model, _ = load_model(path, args.device, dtype)
        if layer < 0:
            layer = model.config.num_hidden_layers + 1 + layer
        anchor_acts[label] = extract_samples(model, batches, layer)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ── Trajectory methods ────────────────────────────────────────────────
    traj_acts:  Dict[str, List[Tuple[int, np.ndarray]]] = {}
    traj_cmaps: Dict[str, str] = {}
    for i, item in enumerate(args.runs):
        name, run_dir_str = item.split(":", 1)
        run_dir = Path(run_dir_str)
        traj_cmaps[name] = TRAJ_CMAPS[i % len(TRAJ_CMAPS)]
        print(f"\n[{name}]  run dir: {run_dir}")
        ckpts = find_checkpoints(run_dir, args.n_checkpoints)
        if not ckpts:
            print(f"  WARNING: no checkpoints with model weights found — skipping.")
            continue
        print(f"  {len(ckpts)} checkpoints: steps {[s for s, _ in ckpts]}")
        traj = []
        for step, ckpt_path in ckpts:
            print(f"  checkpoint-{step}")
            model, _ = load_model(str(ckpt_path), args.device, dtype)
            traj.append((step, extract_samples(model, batches, layer)))
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        traj_acts[name] = traj

    # ── Single-point methods ──────────────────────────────────────────────
    single_acts:   Dict[str, np.ndarray] = {}
    single_colors: Dict[str, str] = {}
    for i, item in enumerate(args.models):
        name, path = item.split(":", 1)
        single_colors[name] = SINGLE_COLORS[i % len(SINGLE_COLORS)]
        print(f"\n[{name}]  (single checkpoint)")
        model, _ = load_model(path, args.device, dtype)
        single_acts[name] = extract_samples(model, batches, layer)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print("\nGenerating trajectory plot …")
    plot_trajectory(
        anchor_acts, traj_acts, single_acts,
        traj_cmaps, single_colors,
        out_dir / "trajectory_pca.png",
        layer_label=str(layer),
    )
    print(f"\nDone. Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
