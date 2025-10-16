#!/bin/bash

# Evaluation script for 3D-FRONT scene generation
# Usage: ./eval.sh <path_to_pkl_file>

set -e  # Exit on error

# Check if pickle file path is provided
if [ -z "$1" ]; then
    echo "Error: No pickle file provided"
    echo "Usage: ./eval.sh <path_to_pkl_file>"
    exit 1
fi

PKL_FILE="$1"

# Check if file exists
if [ ! -f "$PKL_FILE" ]; then
    echo "Error: File not found: $PKL_FILE"
    exit 1
fi

echo "==============================================="
echo "Starting Evaluation Pipeline"
echo "Pickle file: $PKL_FILE"
echo "==============================================="

# Activate conda environment
echo ""
echo "[1/8] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate midiffusion

# 1. Render results
# echo ""
# echo "[2/8] Rendering results..."
# python scripts/render_results.py "$PKL_FILE" --no_texture --without_floor

# 2. FID Score
echo ""
echo "[3/8] Computing FID scores..."
python scripts/compute_fid_scores.py "$PKL_FILE" \
    --output_directory ./fid_tmps \
    --no_texture \
    --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ \
    --no_floor

# 3. KID Score
echo ""
echo "[4/8] Computing KID scores..."
python scripts/compute_fid_scores.py "$PKL_FILE" \
    --compute_kid \
    --output_directory ./fid_tmps \
    --no_texture \
    --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ \
    --no_floor

# 4. Bbox Analysis (OOB and MBL)
echo ""
echo "[5/8] Running bbox analysis (OOB & MBL)..."
python scripts/bbox_analysis.py "$PKL_FILE"

# 5. KL Divergence
echo ""
echo "[6/8] Computing KL divergence..."
python scripts/evaluate_kl_divergence_object_category.py "$PKL_FILE" \
    --output_directory ./kl_tmps

# 6. Object count statistics
echo ""
echo "[7/8] Calculating object count statistics..."
python scripts/calculate_num_obj.py "$PKL_FILE"

# 7. Synthetic vs Real Classifier
echo ""
echo "[8/8] Running synthetic vs real classifier..."
python scripts/synthetic_vs_real_classifier.py "$PKL_FILE" \
    --output_directory ./classifier_tmps \
    --no_texture \
    --no_floor \
    --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

echo ""
echo "==============================================="
echo "Evaluation Complete!"
echo "==============================================="
echo ""
echo "Results saved in:"
echo "  - FID/KID: ./fid_tmps/"
echo "  - KL Divergence: ./kl_tmps/"
echo "  - Classifier: ./classifier_tmps/"
echo ""
