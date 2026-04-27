#!/usr/bin/env bash
# Latent representation analysis — runs all configured experiments.
# Usage:
#   bash scripts/run_repr_analysis.sh                   # all configs
#   bash scripts/run_repr_analysis.sh forget10          # only configs matching "forget10"
#   bash scripts/run_repr_analysis.sh forget10 --tsne   # with t-SNE plots
#
# Activate the conda env first: conda activate unlearning
# Or override the interpreter: PYTHON=/path/to/python bash scripts/run_repr_analysis.sh

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tank/home/judy/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

PYTHON="${PYTHON:-python}"
SCRIPT="scripts/repr_analysis.py"
CONFIG_DIR="analysis_configs"
FILTER="${1:-}"
EXTRA_ARGS="${@:2}"

if [ -z "$FILTER" ]; then
  CONFIGS=("$CONFIG_DIR"/*.json)
else
  CONFIGS=("$CONFIG_DIR"/*"$FILTER"*.json)
fi

echo "==========================================================="
echo "Representation Analysis"
echo "Configs: ${CONFIGS[*]}"
echo "==========================================================="

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "-----------------------------------------------------------"
  echo "Config: $cfg"
  echo "-----------------------------------------------------------"
  $PYTHON "$SCRIPT" --config "$cfg" $EXTRA_ARGS
done

echo ""
echo "==========================================================="
echo "Done. Outputs are under analysis_out/"
echo "==========================================================="
