#!/bin/bash
"""
Sequential Continual Learning Pipeline
Train: KDD Cup -> M4 -> Pedestrian -> Solar -> Uber
Evaluate backward transfer after each step
"""

set -e  # Exit on error

# Configuration
BASE_DIR="./continual_learning_dropout_03_pipeline"
CONFIGS_DIR="./configs/train_tsdiff"
EVAL_CONFIG="${CONFIGS_DIR}/eval_continual.yaml"

# Dataset sequence and their corresponding config files
declare -A DATASETS=(
    [1]="pedestrian_counts" 
    [2]="solar_nips"
    [3]="uber_tlc_hourly"
)

declare -A CONFIG_FILES=(
    [1]="train_pedestrian_counts.yaml"
    [2]="train_solar.yaml" 
    [3]="train_uber_tlc.yaml"
)

declare -A EVAL_NAMES=(
    [0]="kdd_cup_2018_without_missing"
    [1]="pedestrian_counts"
    [2]="solar_nips"
    [3]="uber_tlc_hourly"
)

# Create base directory
mkdir -p "${BASE_DIR}"

echo "=========================================="
echo "SEQUENTIAL CONTINUAL LEARNING PIPELINE"
echo "=========================================="
echo "Sequence: KDD Cup -> M4 -> Pedestrian -> Solar -> Uber"
echo "Base directory: ${BASE_DIR}"
echo

# Initialize with KDD Cup checkpoint (already trained)
# CURRENT_CHECKPOINT="./01_kdd_cup/kdd_cup_2018_without_missing_checkpoint_best.pth"
CURRENT_CHECKPOINT="./test_dropout_03/kdd_cup_2018_without_missing_checkpoint_best.pth"
echo "Starting with KDD Cup checkpoint: ${CURRENT_CHECKPOINT}"

# Verify starting checkpoint exists
if [ ! -f "${CURRENT_CHECKPOINT}" ]; then
    echo "ERROR: Starting checkpoint not found: ${CURRENT_CHECKPOINT}"
    echo "Available checkpoints in 01_kdd_cup/:"
    ls -la ./01_kdd_cup/*.pth 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

# Sequential training and evaluation
for step in {1..4}; do
    CURRENT_DATASET="${DATASETS[$step]}"
    CONFIG_FILE="${CONFIG_FILES[$step]}"
    
    echo
    echo "=========================================="
    echo "STEP $((step + 1)): Training on ${CURRENT_DATASET}"
    echo "=========================================="
    
    # Setup directories
    STEP_DIR="${BASE_DIR}/step_$((step + 1))_${CURRENT_DATASET}"
    TRAIN_DIR="${STEP_DIR}/training"
    EVAL_DIR="${STEP_DIR}/evaluation"
    
    mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}"
    
    # Verify config file exists
    if [ ! -f "${CONFIGS_DIR}/${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIGS_DIR}/${CONFIG_FILE}"
        exit 1
    fi
    
    # Train on current dataset
    echo "Training ${CURRENT_DATASET} from checkpoint..."
    echo "Using config: ${CONFIGS_DIR}/${CONFIG_FILE}"
    
    python bin/train_model_no_lightning.py \
        -c "${CONFIGS_DIR}/${CONFIG_FILE}" \
        --out_dir "${TRAIN_DIR}" \
        --resume_from_checkpoint "${CURRENT_CHECKPOINT}"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for ${CURRENT_DATASET}"
        exit 1
    fi
    
    # Update checkpoint path
    NEW_CHECKPOINT="${TRAIN_DIR}/${CURRENT_DATASET}_checkpoint_best.pth"
    
    # Verify new checkpoint exists
    if [ ! -f "${NEW_CHECKPOINT}" ]; then
        echo "WARNING: Best checkpoint not found, using last checkpoint"
        NEW_CHECKPOINT="${TRAIN_DIR}/${CURRENT_DATASET}_checkpoint_last.pth"
    fi
    
    if [ ! -f "${NEW_CHECKPOINT}" ]; then
        echo "ERROR: No checkpoint found after training ${CURRENT_DATASET}"
        echo "Available files in ${TRAIN_DIR}:"
        ls -la "${TRAIN_DIR}/"
        exit 1
    fi
    
    echo "Training completed. New checkpoint: ${NEW_CHECKPOINT}"
    
    # Evaluate on all previous datasets (backward transfer)
    echo
    echo "Evaluating backward transfer..."
    echo "Testing ${CURRENT_DATASET}-trained model on previous datasets:"
    
    for eval_step in $(seq 0 $step); do
        EVAL_DATASET="${EVAL_NAMES[$eval_step]}"
        EVAL_OUTPUT="${EVAL_DIR}/eval_on_${EVAL_DATASET}"
        
        echo "  - Evaluating on ${EVAL_DATASET}"
        
        python bin/eval_continual_learning.py \
            -c "${EVAL_CONFIG}" \
            --ckpt "${NEW_CHECKPOINT}" \
            --eval_datasets "${EVAL_DATASET}" \
            --out_dir "${EVAL_OUTPUT}" \
            --num_samples 100
        
        if [ $? -ne 0 ]; then
            echo "WARNING: Evaluation failed on ${EVAL_DATASET}"
        else
            echo "    ✓ Evaluation completed: ${EVAL_OUTPUT}"
        fi
    done
    
    # Update current checkpoint for next iteration
    CURRENT_CHECKPOINT="${NEW_CHECKPOINT}"
    
    echo "Step $((step + 1)) completed: ${CURRENT_DATASET}"
done

# Final comprehensive evaluation
echo
echo "=========================================="
echo "FINAL EVALUATION: Testing final model on all datasets"
echo "=========================================="

FINAL_EVAL_DIR="${BASE_DIR}/final_evaluation"
mkdir -p "${FINAL_EVAL_DIR}"

echo "Final model: ${CURRENT_CHECKPOINT}"
echo "Evaluating on all datasets:"

for eval_step in {0..4}; do
    EVAL_DATASET="${EVAL_NAMES[$eval_step]}"
    EVAL_OUTPUT="${FINAL_EVAL_DIR}/final_eval_${EVAL_DATASET}"
    
    echo "  - Evaluating on ${EVAL_DATASET}"
    
    python bin/eval_continual_learning.py \
        -c "${EVAL_CONFIG}" \
        --ckpt "${CURRENT_CHECKPOINT}" \
        --eval_datasets "${EVAL_DATASET}" \
        --out_dir "${EVAL_OUTPUT}" \
        --num_samples 100
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Final evaluation completed: ${EVAL_OUTPUT}"
    else
        echo "    ✗ Final evaluation failed on ${EVAL_DATASET}"
    fi
done

# Generate summary report
echo
echo "=========================================="
echo "GENERATING SUMMARY REPORT"
echo "=========================================="

SUMMARY_FILE="${BASE_DIR}/continual_learning_summary.txt"

cat > "${SUMMARY_FILE}" << EOF
CONTINUAL LEARNING EXPERIMENT SUMMARY
=====================================
Date: $(date)
Sequence: KDD Cup -> M4 -> Pedestrian -> Solar -> Uber

CONFIG FILES USED:
- KDD Cup: Pre-trained model (./01_kdd_cup/)
- M4: ${CONFIGS_DIR}/train_m4.yaml
- Pedestrian: ${CONFIGS_DIR}/train_pedestrian_counts.yaml  
- Solar: ${CONFIGS_DIR}/train_solar.yaml
- Uber: ${CONFIGS_DIR}/train_uber_tlc.yaml
- Evaluation: ${CONFIGS_DIR}/eval_continual.yaml

TRAINING STEPS:
1. Started with: KDD Cup (pre-trained)
2. Step 2: Trained on M4 Hourly
3. Step 3: Trained on Pedestrian Counts 
4. Step 4: Trained on Solar NIPS
5. Step 5: Trained on Uber TLC Hourly

EVALUATION STRUCTURE:
- After each training step, evaluated on all previous datasets
- Final evaluation on all 5 datasets using final model

DIRECTORIES:
- Base: ${BASE_DIR}
- Step results: ${BASE_DIR}/step_*/
- Final evaluation: ${BASE_DIR}/final_evaluation/

CHECKPOINTS:
- Started: ./01_kdd_cup/kdd_cup_2018_without_missing_checkpoint_best.pth
- Final: ${CURRENT_CHECKPOINT}

DATASET EVAL NAMES:
- KDD Cup: kdd_cup_2018_without_missing
- M4: m4_hourly  
- Pedestrian: pedestrian_counts
- Solar: solar_nips
- Uber: uber_tlc_hourly

ANALYSIS:
To analyze results, check:
1. Training histories: */training/results.yaml
2. Evaluation metrics: */evaluation/*/continual_evaluation_results.yaml
3. Final metrics: final_evaluation/*/continual_evaluation_results.yaml
EOF

echo "Summary saved to: ${SUMMARY_FILE}"

echo
echo "=========================================="
echo "CONTINUAL LEARNING PIPELINE COMPLETED"
echo "=========================================="
echo "Results directory: ${BASE_DIR}"
echo "Summary report: ${SUMMARY_FILE}"
echo
echo "To analyze catastrophic forgetting:"
echo "1. Check evaluation results in each step_* directory"
echo "2. Compare performance degradation across datasets"
echo "3. Look for backward/forward transfer patterns"
