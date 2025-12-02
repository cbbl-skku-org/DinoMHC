#!/bin/bash
#
# Hyperparameter Tuning Script for DinoMHC
#
# Usage:
#   bash run_tuning.sh                    # Use default GPUs (0,1)
#   bash run_tuning.sh 2,3                # Use GPUs 2 and 3
#   bash run_tuning.sh 0,1,2,3            # Use all 4 GPUs
#   bash run_tuning.sh 1                  # Use only GPU 1
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Parse GPU selection from command line
if [ -n "$1" ]; then
    GPU_IDS="$1"
    # Count number of GPUs specified
    N_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
else
    # Default: use GPUs 0-7
    GPU_IDS="0,1,2,3"
    N_GPUS=4
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/configs/test_5pct.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/optuna_tuning_5pct"
PYTHON_BIN="/home/duongtt/conda_env/DINO-MHC/bin/python"

# Tuning parameters
N_TRIALS=100              # Number of hyperparameter combinations to try
N_FOLDS=5                # Number of cross-validation folds
MAX_EPOCHS=15            # Max epochs per fold (keep low for fast tuning)
SEED=69                  # Random seed for reproducibility

# Study settings
STUDY_NAME="dinomhc_tuning_$(date +%Y%m%d_%H%M%S)"
STORAGE="sqlite:///${OUTPUT_DIR}/optuna_study.db"

# ============================================================================
# Environment Setup
# ============================================================================

echo "============================================================"
echo "DinoMHC Hyperparameter Tuning"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Config file:     ${CONFIG_FILE}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "  Study name:      ${STUDY_NAME}"
echo "  N trials:        ${N_TRIALS}"
echo "  N folds:         ${N_FOLDS}"
echo "  Max epochs:      ${MAX_EPOCHS}"
echo "  GPU IDs:         ${GPU_IDS}"
echo "  Number of GPUs:  ${N_GPUS}"
echo "  Parallel folds:  Yes (trains ${N_FOLDS} folds in parallel)"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Check if Python exists
if [ ! -f "${PYTHON_BIN}" ]; then
    echo "Error: Python binary not found: ${PYTHON_BIN}"
    exit 1
fi

# ============================================================================
# Run Hyperparameter Tuning
# ============================================================================

echo "Starting hyperparameter tuning..."
echo ""

# Set CUDA_VISIBLE_DEVICES to restrict to specified GPUs
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# Run tuning with parallel fold training
"${PYTHON_BIN}" tune.py \
    --config "${CONFIG_FILE}" \
    --n_trials ${N_TRIALS} \
    --n_folds ${N_FOLDS} \
    --max_epochs ${MAX_EPOCHS} \
    --study_name "${STUDY_NAME}" \
    --storage "${STORAGE}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed ${SEED} \
    --n_startup_trials 4 \
    --pruning \
    --parallel_folds \
    --n_gpus ${N_GPUS} \
    --n_jobs 5

# ============================================================================
# Post-Tuning Summary
# ============================================================================

echo ""
echo "============================================================"
echo "Tuning Complete!"
echo "============================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - Best config:    ${OUTPUT_DIR}/best_config.yaml"
echo "  - Study stats:    ${OUTPUT_DIR}/study_stats.yaml"
echo "  - Study database: ${OUTPUT_DIR}/optuna_study.db"
echo ""
echo "To visualize results with Optuna dashboard:"
echo "  optuna-dashboard ${STORAGE}"
echo ""
echo "To resume tuning (add more trials):"
echo "  bash run_tuning.sh  # Will automatically resume from saved study"
echo ""
