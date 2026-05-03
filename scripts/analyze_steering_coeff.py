"""
Steering-coefficient representation analysis for MUSE LatentRMU.

For each unlearned model in a coeff sweep:
  1. Run forget data through both the unlearned model and the target.
  2. Hook layer 7 (or whatever module_regex was used) and pool over tokens.
  3. Measure how far the unlearned activations are from the target.

Outputs (under analysis_out/steering_coeff_<DATA_SPLIT>/):
  - displacement.csv          per-coeff scalar metrics
  - displacement_curve.png    L2 / cosine vs coeff
  - pca_scatter.png           PCA of pooled activations across coeffs

Usage:
  python scripts/analyze_steering_coeff.py \
      --data_split News \
      --model Llama-3.2-3B-Instruct \
      --coeffs "0 5 10 20 40 80" \
      --n_samples 64
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def collect_pooled_activations(model_path, tokenizer, texts, layer_idx, device, batch_size=4, max_length=1024):
    """Returns (N, hidden) pooled (mean over tokens) activations from layer_idx."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device).eval()

    cache = []
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        cache.append(h.float().cpu())
    handle = model.model.layers[layer_idx].register_forward_hook(hook)

    pooled = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        cache.clear()
        model(**enc)
        h = cache[0]                                # (B, S, H)
        mask = enc.attention_mask.float().cpu()     # (B, S)
        masked = h * mask.unsqueeze(-1)
        avg = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled.append(avg)

    handle.remove()
    del model
    torch.cuda.empty_cache()
    return torch.cat(pooled, dim=0).numpy()         # (N, H)


def displacement_metrics(target_act, unlearned_act):
    """Per-sample distances between target and unlearned representations."""
    delta = unlearned_act - target_act                              # (N, H)
    l2 = np.linalg.norm(delta, axis=1)                              # (N,)
    target_norm = np.linalg.norm(target_act, axis=1)                # (N,)
    rel_l2 = l2 / np.maximum(target_norm, 1e-8)                     # relative shift
    cos = (target_act * unlearned_act).sum(1) / (
        np.linalg.norm(target_act, axis=1) * np.linalg.norm(unlearned_act, axis=1) + 1e-8
    )
    return {
        "l2_mean": float(l2.mean()),
        "l2_std": float(l2.std()),
        "rel_l2_mean": float(rel_l2.mean()),
        "cos_sim_mean": float(cos.mean()),
        "cos_sim_std": float(cos.std()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_split", default="News", choices=["News", "Books"])
    p.add_argument("--model", default="Llama-3.2-3B-Instruct")
    p.add_argument("--coeffs", default="0 5 10 20 40 80",
                   help="space-separated steering_coeff values that were swept")
    p.add_argument("--layer", type=int, default=7, help="hidden layer to hook (matches module_regex)")
    p.add_argument("--n_samples", type=int, default=64, help="forget samples to evaluate")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    coeffs = [int(c) for c in args.coeffs.split()]
    out_dir = Path(args.out_dir or f"analysis_out/steering_coeff_{args.data_split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output → {out_dir}")

    target_path = f"saves/finetune/muse_{args.model}_{args.data_split}_target/last"
    unlearn_paths = {
        c: f"saves/unlearn/muse_{args.model}_{args.data_split}_LatentRMU_sweep_coeff_{c}/last"
        for c in coeffs
    }

    # Verify paths.
    missing = [target_path] + [p for p in unlearn_paths.values() if not Path(p).exists()]
    missing = [m for m in missing if not Path(m).exists()]
    if missing:
        print("ERROR: missing checkpoints:")
        for m in missing:
            print(f"  {m}")
        return 1

    # Load forget data.
    print(f"loading muse-bench/MUSE-{args.data_split} forget split...")
    ds = load_dataset("muse-bench/MUSE-" + args.data_split, name="raw", split="forget")
    texts = ds["text"][: args.n_samples]
    print(f"using {len(texts)} forget samples")

    tokenizer = AutoTokenizer.from_pretrained(target_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pool target activations once.
    print(f"collecting target activations from {target_path}")
    target_act = collect_pooled_activations(
        target_path, tokenizer, texts, args.layer, args.device, args.batch_size, args.max_length
    )
    print(f"  target shape: {target_act.shape}")

    # For each coeff, collect activations and measure displacement.
    rows = []
    all_acts = {"target": target_act}
    for c in coeffs:
        path = unlearn_paths[c]
        print(f"\ncollecting unlearned activations for coeff={c}")
        u_act = collect_pooled_activations(
            path, tokenizer, texts, args.layer, args.device, args.batch_size, args.max_length
        )
        all_acts[f"coeff_{c}"] = u_act
        m = displacement_metrics(target_act, u_act)
        m["coeff"] = c
        rows.append(m)
        print(f"  coeff={c}: l2={m['l2_mean']:.3f}  rel_l2={m['rel_l2_mean']:.3f}  cos={m['cos_sim_mean']:.3f}")

    # Write CSV.
    csv_path = out_dir / "displacement.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["coeff", "l2_mean", "l2_std", "rel_l2_mean", "cos_sim_mean", "cos_sim_std"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # Plot displacement curves.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cs = [r["coeff"] for r in rows]
    ax[0].errorbar(cs, [r["l2_mean"] for r in rows], yerr=[r["l2_std"] for r in rows],
                   marker="o", capsize=3)
    ax[0].set_xlabel("steering_coeff")
    ax[0].set_ylabel("L2 distance from target")
    ax[0].set_title("Activation displacement vs steering coeff")
    ax[0].grid(True, alpha=0.3)

    ax[1].errorbar(cs, [r["cos_sim_mean"] for r in rows], yerr=[r["cos_sim_std"] for r in rows],
                   marker="s", capsize=3, color="C1")
    ax[1].axhline(1.0, ls="--", c="gray", alpha=0.5, label="target = 1.0")
    ax[1].set_xlabel("steering_coeff")
    ax[1].set_ylabel("cosine similarity to target")
    ax[1].set_title("Direction preservation")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = out_dir / "displacement_curve.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {plot_path}")

    # PCA of all activations to visualize the sweep.
    stacked = np.concatenate(list(all_acts.values()), axis=0)        # (n_coeffs+1)*N, H
    pca = PCA(n_components=2).fit(stacked)
    fig, ax = plt.subplots(figsize=(7, 6))
    n = target_act.shape[0]
    cmap = plt.cm.viridis
    for i, (label, act) in enumerate(all_acts.items()):
        proj = pca.transform(act)
        if label == "target":
            ax.scatter(proj[:, 0], proj[:, 1], c="black", marker="x", s=40, label="target", zorder=5)
        else:
            color = cmap(i / max(1, len(coeffs)))
            ax.scatter(proj[:, 0], proj[:, 1], c=[color], s=18, alpha=0.6, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Layer-{args.layer} activations: forget samples ({args.data_split})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pca_path = out_dir / "pca_scatter.png"
    fig.savefig(pca_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pca_path}")

    # Save raw activations for further analysis.
    np.savez(out_dir / "activations.npz", **all_acts)
    print(f"wrote {out_dir / 'activations.npz'}")


if __name__ == "__main__":
    raise SystemExit(main())
