#!/bin/bash

# Batch evaluation script for multiple pickle files
# Phase 1: Render all pickle files first
# Phase 2: Run all evaluations for each pickle file

# Define the base directory
BASE_DIR="/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/"

# Array of pickle files to evaluate
PKL_FILES=(
    "$BASE_DIR/outputs/2025-10-16/10-37-29/sampled_scenes_results.pkl"  # DiffuScene, jfgw3io6, ddpm, True, 1000
    "$BASE_DIR/outputs/2025-10-16/10-43-55/sampled_scenes_results.pkl"  # DiffuScene, jfgw3io6, ddim, True, 150
    "$BASE_DIR/outputs/2025-10-16/10-45-35/sampled_scenes_results.pkl"  # Continuous MI, pfksynuz, ddpm, True, 1000
    "$BASE_DIR/outputs/2025-10-16/10-48-55/sampled_scenes_results.pkl"  # Continuous MI, pfksynuz, ddim, True, 150
)

# Get total number of files
TOTAL=${#PKL_FILES[@]}

echo "==============================================="
echo "Batch Evaluation Script (2-Phase)"
echo "Total files to process: $TOTAL"
echo "==============================================="

# Log file for summary
LOG_FILE="batch_eval_$(date +%Y%m%d_%H%M%S).log"
echo "Batch Evaluation Log - $(date)" > "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Activate conda environment
echo ""
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate midiffusion

# ========================================
# PHASE 1: Render all pickle files
# ========================================
echo ""
echo "==============================================="
echo "PHASE 1: Rendering All Scenes"
echo "==============================================="
echo "" >> "$LOG_FILE"
echo "PHASE 1: RENDERING" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"

COUNTER=1
for PKL_FILE in "${PKL_FILES[@]}"; do
    echo ""
    echo "[$COUNTER/$TOTAL] Rendering: $(basename $(dirname $PKL_FILE))"
    
    if [ ! -f "$PKL_FILE" ]; then
        echo "WARNING: File not found, skipping: $PKL_FILE"
        echo "SKIPPED (not found): $(basename $(dirname $PKL_FILE))" >> "$LOG_FILE"
        ((COUNTER++))
        continue
    fi
    
    START_TIME=$(date +%s)
    
    if python scripts/render_results.py "$PKL_FILE" --no_texture --without_floor 2>&1 | tee -a "$LOG_FILE"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "✅ Rendered in ${DURATION}s"
        echo "RENDERED: $(basename $(dirname $PKL_FILE)) - ${DURATION}s" >> "$LOG_FILE"
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "❌ Rendering FAILED"
        echo "RENDER FAILED: $(basename $(dirname $PKL_FILE))" >> "$LOG_FILE"
    fi
    
    ((COUNTER++))
done

# ========================================
# PHASE 2: Run all evaluations
# ========================================
echo ""
echo "==============================================="
echo "PHASE 2: Running All Evaluations"
echo "==============================================="
echo "" >> "$LOG_FILE"
echo "PHASE 2: EVALUATIONS" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"

COUNTER=1
for PKL_FILE in "${PKL_FILES[@]}"; do
    echo ""
    echo "==============================================="
    echo "Evaluating [$COUNTER/$TOTAL]: $(basename $(dirname $PKL_FILE))"
    echo "File: $PKL_FILE"
    echo "==============================================="
    
    if [ ! -f "$PKL_FILE" ]; then
        echo "SKIPPED: File not found"
        ((COUNTER++))
        continue
    fi
    
    START_TIME=$(date +%s)
    
    # Run all evaluations
    EVAL_SUCCESS=true
    
    # 1. FID Score
    echo ""
    echo "[1/6] Computing FID scores..."
    if ! python scripts/compute_fid_scores.py "$PKL_FILE" \
        --output_directory ./fid_tmps \
        --no_texture \
        --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ \
        --no_floor 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi
    
    # 2. KID Score
    echo ""
    echo "[2/6] Computing KID scores..."
    if ! python scripts/compute_fid_scores.py "$PKL_FILE" \
        --compute_kid \
        --output_directory ./fid_tmps \
        --no_texture \
        --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ \
        --no_floor 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi
    
    # 3. Bbox Analysis
    echo ""
    echo "[3/6] Running bbox analysis..."
    if ! python scripts/bbox_analysis.py "$PKL_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi
    
    # 4. KL Divergence
    echo ""
    echo "[4/6] Computing KL divergence..."
    if ! python scripts/evaluate_kl_divergence_object_category.py "$PKL_FILE" \
        --output_directory ./kl_tmps 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi
    
    # 5. Object count
    echo ""
    echo "[5/6] Calculating object count..."
    if ! python scripts/calculate_num_obj.py "$PKL_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi
    
    # 6. Classifier
    echo ""
    echo "[6/6] Running classifier..."
    if ! python scripts/synthetic_vs_real_classifier.py "$PKL_FILE" \
        --output_directory ./classifier_tmps \
        --no_texture \
        --no_floor \
        --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ "$EVAL_SUCCESS" = true ]; then
        echo ""
        echo "✅ SUCCESS: All evaluations completed in ${DURATION}s - $(basename $(dirname $PKL_FILE))"
        echo "SUCCESS: $(basename $(dirname $PKL_FILE)) - Duration: ${DURATION}s" >> "$LOG_FILE"
    else
        echo ""
        echo "❌ FAILED: Some evaluations failed - $(basename $(dirname $PKL_FILE))"
        echo "FAILED: $(basename $(dirname $PKL_FILE)) - Duration: ${DURATION}s" >> "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
    ((COUNTER++))
done

echo ""
echo "==============================================="
echo "Batch Evaluation Complete!"
echo "==============================================="
echo "Processed: $TOTAL files"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Summary:"
echo "--------"
grep -E "SUCCESS|FAILED|SKIPPED|RENDERED" "$LOG_FILE" | grep -v "PHASE"
echo ""
