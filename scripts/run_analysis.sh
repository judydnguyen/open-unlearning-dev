#!/bin/bash
# Distributional shift analysis: Base → Finetuned → Unlearned (RMU)
#
# Model:        Llama-3.2-1B-Instruct
# Forget split: forget10  (10 % of TOFU authors)
# Retain split: retain90
# Holdout:      holdout10
# Method:       RMU
#
# Three model states compared on the forget set:
#   BASE      = retain90-finetuned  (no exposure to forget set; proxy for retrained)
#   FINETUNED = full-finetuned      (trained on forget + retain; memorisation baseline)
#   UNLEARNED = RMU checkpoint      (unlearned from full model)
#
# Outputs written to:  analysis_out/RMU_forget10/

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"
METHOD="SimpleNPO"

SAVES="saves"

# Base model: retain90-finetuned = no exposure to the forget set
BASE_MODEL="${SAVES}/finetune/tofu_${MODEL}_${RETAIN_SPLIT}"

# Finetuned: trained on the full TOFU dataset (forget + retain)
FINETUNED_MODEL="${SAVES}/finetune/tofu_${MODEL}_full"

# Unlearned: RMU checkpoint for forget10
UNLEARNED_MODEL="${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_${METHOD}"

# Output directory
OUTPUT_DIR="analysis_out/${METHOD}_${FORGET_SPLIT}"

# ── Settings ─────────────────────────────────────────────────────────────────
BATCH_SIZE=4
MAX_NEW_TOKENS=200
LAYER=-1          # -1 = last hidden layer
DEVICE="cuda:3"
DTYPE="bfloat16"
SEED=42

# ── Sanity checks ─────────────────────────────────────────────────────────────
for dir in "$BASE_MODEL" "$FINETUNED_MODEL" "$UNLEARNED_MODEL"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: checkpoint not found: $dir"
        exit 1
    fi
done

echo "============================================================"
echo " Analysis: ${MODEL} | ${FORGET_SPLIT} | ${METHOD}"
echo "============================================================"
echo "  Base      : ${BASE_MODEL}"
echo "  Finetuned : ${FINETUNED_MODEL}"
echo "  Unlearned : ${UNLEARNED_MODEL}"
echo "  Output    : ${OUTPUT_DIR}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

# ── Run analysis ──────────────────────────────────────────────────────────────
# --forget_data forget10  → loads locuslab/TOFU forget10 split from HuggingFace
#                           (requires HF_HOME or internet access)
# Add --tsne to also produce a t-SNE projection plot.
# Add --load_sequential if all three models don't fit on GPU simultaneously.

python scripts/analysis.py \
    --base_model      "${BASE_MODEL}"      \
    --finetuned_model "${FINETUNED_MODEL}" \
    --unlearned_model "${UNLEARNED_MODEL}" \
    --forget_data     "${FORGET_SPLIT}"    \
    --output_dir      "${OUTPUT_DIR}"      \
    --batch_size      "${BATCH_SIZE}"      \
    --max_new_tokens  "${MAX_NEW_TOKENS}"  \
    --layer           "${LAYER}"           \
    --device          "${DEVICE}"          \
    --dtype           "${DTYPE}"           \
    --seed            "${SEED}"

echo ""
echo "Done. Results in: ${OUTPUT_DIR}/"
echo "  metrics_summary.json  — mean ROUGE + pairwise distances"
echo "  rouge_scores.csv      — per-sample ROUGE for all three models"
echo "  rouge_comparison.png  — boxplots"
echo "  rouge_histograms.png  — overlapping histograms with mean lines"
echo "  activation_pca.png    — PCA scatter of hidden states"
echo "  distance_heatmap.png  — cosine / L2 / Wasserstein heatmap"
