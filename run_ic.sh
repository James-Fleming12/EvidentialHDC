#!/bin/bash

# Diagnostic Panel for Inter/Intra-Class Balancing
# Evaluates IC1, IC4, and XC2 against the Bayesian Momentum baseline.

# METHODS=("none" "ic1" "ic4" "xc2")

# for method in "${METHODS[@]}"; do
#     echo "=========================================="
#     echo "Running IC Diagnostic: ${method}"
#     echo "=========================================="
#     PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 python unsup_kitti-c.py \
#         --method evidential_hdc_tta \
#         --ic_method ${method} \
#         --chunked \
#         --reset_per_corruption \
#         --log_dir logs/ic_diagnostics_${method}
# done

# echo "IC Diagnostics Complete."

# ----

# Resuming Diagnostic Panel for Inter/Intra-Class Balancing
METHODS=("xc2")

for method in "${METHODS[@]}"; do
    echo "=========================================="
    echo "Running IC Diagnostic: ${method}"
    echo "=========================================="
    PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 python unsup_kitti-c.py \
        --method evidential_hdc_tta \
        --ic_method ${method} \
        --chunked \
        --reset_per_corruption \
        --log_dir logs/ic_diagnostics_${method}
done

echo "IC Diagnostics Complete."