#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Clean old logs
rm -f ../Logs/week2_results.log
mkdir -p ../Logs

echo "==========================================" > ../Logs/week2_results.log
echo "Week 2: Multi-View TTA Consensus" >> ../Logs/week2_results.log
echo "==========================================" >> ../Logs/week2_results.log
echo "Testing Spatial Augmentations (Base, Yaw-Shifted, Depth-Scaled)" >> ../Logs/week2_results.log
echo "Using IC4 (Epistemic Weighting) + Calibrated tau=-1.0 as the base configuration." >> ../Logs/week2_results.log
echo "" >> ../Logs/week2_results.log

for tta in "bundle" "min_uncert" "mean_uncert"; do
    echo "------------------------------------------" >> ../Logs/week2_results.log
    echo "Testing Multi-View Strategy: $tta" >> ../Logs/week2_results.log
    echo "------------------------------------------" >> ../Logs/week2_results.log
    
    python unsup_kitti-c.py \
        --method evidential_hdc_tta \
        --chunked \
        --reset_per_corruption \
        --ic_method ic4 \
        --tau -1.0 \
        --kappa 15.0 \
        --mv_tta $tta \
        >> ../Logs/week2_results.log 2>&1
done

echo "Week 2 Multi-View TTA tests complete!"
