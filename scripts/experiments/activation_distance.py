"""Experiment 1 — Layer-wise distance to Retrain in latent space.

For each unlearning method (Baseline, FLOUR, RMU, NPO, SimNPO, GradAscent, Retrain),
captures per-layer mean residual-stream activations on a forget-set batch, then
computes 1 − cosine_similarity(method_layer, Retrain_layer) at every layer.

Output:
  analysis_out/experiment1_layer_distance.json   — raw distances per (method, layer)
  analysis_out/experiment1_layer_distance.png    — line plot

Hypothesis: FLOUR's curve stays below RMU/NPO/GradAscent at every layer, indicating
its hidden-state shift more closely tracks the retrain oracle.

Usage:
    python scripts/experiments/activation_distance.py
    python scripts/experiments/activation_distance.py --split forget05    # other splits
    python scripts/experiments/activation_distance.py --device cuda:0
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


_TEMPLATE_ARGS = {
    "apply_chat_template": True,
    "system_prompt": "You are a helpful assistant.",
    "system_prompt_with_special_tokens":
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>",
    "user_start_tag": "<|start_header_id|>user<|end_header_id|>\n\n",
    "user_end_tag": "<|eot_id|>",
    "asst_start_tag": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "asst_end_tag": "<|eot_id|>",
    "date_string": "10 Apr 2025",
}


def load_tofu_batch(hf_split_name, n_examples, tokenizer):
    """Load a tokenized batch from any TOFU split (forget01/forget05/forget10/retain99/...)."""
    from data.qa import QADataset
    from data.collators import DataCollatorForSupervisedDataset

    ds = QADataset(
        hf_args={"name": hf_split_name, "split": "train", "path": "locuslab/TOFU"},
        template_args=_TEMPLATE_ARGS,
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=512,
    )
    n = min(n_examples, len(ds))
    items = [ds[i] for i in range(n)]
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, padding_side="right")
    batch = collator(items)
    return batch


def get_tokenizer(model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@torch.no_grad()
def per_layer_mean_activations(model, batch, device):
    """Forward + collect per-layer mean residual-stream activations.

    Returns: tensor of shape [n_layers + 1, hidden_size] — mean over (batch, label_tokens)
             of `output_hidden_states`. Index 0 is embedding output, last is final layer.
             Padding/non-label tokens are masked out using `labels != -100`.
    """
    inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
    labels = batch.get("labels")
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple of [B, T, H], one per layer (incl. embeddings)

    if labels is not None:
        mask = (labels != -100).to(device).unsqueeze(-1).float()  # [B, T, 1]
    else:
        mask = inputs["attention_mask"].unsqueeze(-1).float()
    denom = mask.sum().clamp_min(1.0)

    means = []
    for h in hs:
        # h: [B, T, H] — average over batch+token positions where mask=1
        masked = h * mask
        m = masked.sum(dim=(0, 1)) / denom
        means.append(m.float().cpu())
    return torch.stack(means, dim=0)  # [n_layers+1, H]


def load_model(path, device, dtype=torch.bfloat16):
    """Load a model checkpoint by local path or HF repo id."""
    from transformers import AutoModelForCausalLM
    print(f"  loading {path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forget01")
    ap.add_argument("--n_examples", type=int, default=40,
                    help="number of forget-set examples to use for the mean")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default="analysis_out")
    args = ap.parse_args()

    holdout_split = "holdout" + args.split.replace("forget", "")

    # Method → model path. Local paths preferred; HF repo fallback for Baseline / Retrain.
    METHODS = {
        "Baseline":   "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "Retrain":    "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99",
        "FLOUR":      "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_LatentRMU_v4.8_sweep_g0.50/checkpoint-50",
        "RMU":        "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_RMU",
        "NPO":        "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_NPO",
        "SimNPO":     "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_SimNPO",
        "GradAscent": "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradAscent",
    }
    if args.split == "forget05":
        METHODS["Retrain"] = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95"
        METHODS["FLOUR"]   = "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_LatentRMU_v4.8_sweep_g0.50/checkpoint-50"
        for m in ("RMU","NPO","SimNPO","GradAscent"):
            METHODS[m] = METHODS[m].replace("forget01", "forget05")
        if not Path(METHODS["SimNPO"]).is_dir() or not (Path(METHODS["SimNPO"])/"model.safetensors").exists():
            # SimNPO forget05 weights live in best/
            METHODS["SimNPO"] = METHODS["SimNPO"] + "/best"
    elif args.split == "forget10":
        METHODS["Retrain"] = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
        METHODS["FLOUR"]   = "saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_LatentRMU_v4.8_sweep_g0.50/checkpoint-150"
        for m in ("RMU","NPO","SimNPO","GradAscent"):
            METHODS[m] = METHODS[m].replace("forget01", "forget10")

    device = args.device
    retain_split = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}[args.split]

    # Tokenize both forget and retain batches (shared tokenizer — all methods use same vocab)
    print("== Tokenizing batches ==")
    tokenizer = get_tokenizer(METHODS["Baseline"])
    forget_batch = load_tofu_batch(args.split, args.n_examples, tokenizer)
    retain_batch = load_tofu_batch(retain_split, args.n_examples, tokenizer)
    print(f"  forget batch ({args.split}): input_ids {tuple(forget_batch['input_ids'].shape)}, "
          f"label tokens = {(forget_batch['labels'] != -100).sum().item()}")
    print(f"  retain batch ({retain_split}): input_ids {tuple(retain_batch['input_ids'].shape)}, "
          f"label tokens = {(retain_batch['labels'] != -100).sum().item()}")

    # Capture per-layer means for each method on BOTH forget and retain batches
    activations = {"forget": {}, "retain": {}}
    for name, path in METHODS.items():
        print(f"\n== {name} ==")
        try:
            model = load_model(path, device)
            activations["forget"][name] = per_layer_mean_activations(model, forget_batch, device)
            activations["retain"][name] = per_layer_mean_activations(model, retain_batch, device)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    if "Retrain" not in activations["forget"]:
        raise SystemExit("No Retrain reference — cannot compute distances.")

    # Distance to Retrain at each layer for both batches (1 − cosine_similarity)
    distances = {"forget": {}, "retain": {}}
    n_layers = activations["forget"]["Retrain"].shape[0]
    for batch_name in ("forget", "retain"):
        retrain_means = activations[batch_name]["Retrain"]
        for name, means in activations[batch_name].items():
            per_layer = []
            for li in range(n_layers):
                cos = F.cosine_similarity(
                    means[li].unsqueeze(0), retrain_means[li].unsqueeze(0)
                ).item()
                per_layer.append(1.0 - cos)
            distances[batch_name][name] = per_layer

    # Persist results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"experiment1_layer_distance_{args.split}.json"
    json_path.write_text(json.dumps({
        "split": args.split,
        "retain_split": retain_split,
        "n_examples": args.n_examples,
        "n_layers": n_layers,
        "distances_forget": distances["forget"],
        "distances_retain": distances["retain"],
        "methods": list(METHODS.keys()),
    }, indent=2))
    print(f"\nSaved distances → {json_path}")

    # Plot — single 2D scatter showing forget vs retain selectivity tradeoff.
    # x = retain shift, y = forget shift, one trajectory per method (across layers).
    # Methods above the y=x diagonal are SELECTIVE (shift forget more than retain).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        colors = {
            "Baseline":   "#888888",
            "Retrain":    "#000000",
            "FLOUR":      "#e91e63",   # pink (matches paper highlight)
            "RMU":        "#1f77b4",
            "NPO":        "#2ca02c",
            "SimNPO":     "#9467bd",
            "GradAscent": "#ff7f0e",
        }
        # Trajectory per method: connect (retain_l, forget_l) across layers
        for name in METHODS:
            if name not in distances["forget"]: continue
            xs = distances["retain"][name]
            ys = distances["forget"][name]
            is_flour = name == "FLOUR"
            ax.plot(xs, ys,
                    color=colors.get(name), label=None,
                    linewidth=2.5 if is_flour else 1.2,
                    alpha=0.85 if is_flour else 0.55,
                    zorder=3 if is_flour else 2)
            ax.scatter(xs, ys,
                       color=colors.get(name), s=20, alpha=0.6,
                       edgecolor="white", linewidth=0.5, zorder=4)
            # Annotate at peak forget layer
            pk_li = max(range(len(ys)), key=lambda i: ys[i])
            ax.annotate(
                f" {name} (L{pk_li})",
                xy=(xs[pk_li], ys[pk_li]),
                fontsize=9, color=colors.get(name),
                fontweight="bold" if is_flour else "normal",
            )

        # y = x diagonal: methods above this line are "selective" (forget shift > retain shift)
        lo, hi = 0.0, max(
            max(max(d) for d in distances["forget"].values()),
            max(max(d) for d in distances["retain"].values()),
        ) * 1.05
        ax.plot([lo, hi], [lo, hi], "--", color="#aaaaaa", linewidth=1, label="y = x  (no selectivity)")
        ax.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.05, color="green",
                        label="Selective: forget shifted > retain")
        ax.fill_between([lo, hi], [lo, lo], [lo, hi], alpha=0.05, color="red",
                        label="Non-selective: retain shifted ≥ forget")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"Retain-set distance:  $1-\cos(\mathbf{h}_{\text{method}}, \mathbf{h}_{\text{Retrain}})$")
        ax.set_ylabel(r"Forget-set distance:  $1-\cos(\mathbf{h}_{\text{method}}, \mathbf{h}_{\text{Retrain}})$")
        ax.set_title(f"Forget/Retain selectivity tradeoff — {args.split}\n"
                     f"(each line = method's per-layer trajectory; label marks peak forget layer)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", frameon=True, fontsize=9)
        ax.set_aspect("equal")
        fig.tight_layout()
        png_path = out_dir / f"experiment1_layer_distance_{args.split}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Saved merged tradeoff plot → {png_path}")
    except ImportError:
        print("  matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
