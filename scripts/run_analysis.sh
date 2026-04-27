#!/bin/bash
# Distributional shift analysis: Base → Finetuned → RMU / SimNPO
#
# Model:        Llama-3.2-1B-Instruct
# Forget split: forget10  (10 % of TOFU authors)
# Retain split: retain90
#
# Model states compared on the forget set:
#   BASE      = retain90-finetuned  (no exposure to forget set; proxy for retrained)
#   FINETUNED = full-finetuned      (trained on forget + retain; memorisation baseline)
#   RMU       = RMU unlearned checkpoint
#   SimNPO    = SimNPO unlearned checkpoint
#
# Outputs written to:  analysis_out/RMU_forget10/

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"

SAVES="saves"

# Base model: retain90-finetuned = no exposure to the forget set
BASE_MODEL="${SAVES}/finetune/tofu_${MODEL}_${RETAIN_SPLIT}"

# Finetuned: trained on the full TOFU dataset (forget + retain)
FINETUNED_MODEL="${SAVES}/finetune/tofu_${MODEL}_full"

# Output directory (kept as RMU_forget10 for continuity with existing artifacts)
OUTPUT_DIR="analysis_out/RMU_forget10"

# ── Settings ─────────────────────────────────────────────────────────────────
BATCH_SIZE=4
MAX_NEW_TOKENS=200
LAYER=-1          # -1 = last hidden layer
DEVICE="cuda:1"
DTYPE="bfloat16"
SEED=42

# ── Sanity checks ─────────────────────────────────────────────────────────────
for dir in "$BASE_MODEL" "$FINETUNED_MODEL" \
           "${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_GradAscent" \
           "${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_GradDiff"   \
           "${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_NPO"        \
           "${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_RMU"        \
           "${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_SimNPO"     \
           "${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_LatentRMU_v4.8"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: checkpoint not found: $dir"
        exit 1
    fi
done

echo "============================================================"
echo " Analysis: ${MODEL} | ${FORGET_SPLIT} | all baselines"
echo "============================================================"
echo "  Base      : ${BASE_MODEL}"
echo "  Finetuned : ${FINETUNED_MODEL}"
echo "  Methods   : GradAscent  GradDiff  NPO  RMU  SimNPO  LatentRMU"
echo "  Output    : ${OUTPUT_DIR}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

# ── Run analysis ──────────────────────────────────────────────────────────────
# --methods passes name:path pairs for each unlearned method.
# Add --tsne to also produce a t-SNE projection plot.

python scripts/analysis.py \
    --base_model      "${BASE_MODEL}"      \
    --finetuned_model "${FINETUNED_MODEL}" \
    --methods \
        "GradAscent:${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_GradAscent" \
        "GradDiff:${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_GradDiff"     \
        "NPO:${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_NPO"               \
        "RMU:${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_RMU"               \
        "SimNPO:${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_SimNPO"         \
        "LatentRMU:${SAVES}/unlearn/tofu_${MODEL}_${FORGET_SPLIT}_LatentRMU_v4.8" \
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
echo "  rouge_scores.csv      — per-sample ROUGE for all models"
echo "  rouge_comparison.png  — boxplots"
echo "  rouge_histograms.png  — overlapping histograms with mean lines"
echo "  activation_pca.png    — PCA scatter: Base / Finetuned / RMU / SimNPO"
echo "  distance_heatmap.png  — cosine / L2 / Wasserstein heatmap"
