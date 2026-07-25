#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Clean old logs
rm -f /home/james/Research/SEE/Logs/week2_results.log
mkdir -p /home/james/Research/SEE/Logs

echo "==========================================" > /home/james/Research/SEE/Logs/week2_results.log
echo "Week 2: Multi-View TTA Consensus" >> /home/james/Research/SEE/Logs/week2_results.log
echo "==========================================" >> /home/james/Research/SEE/Logs/week2_results.log
echo "Testing Spatial Augmentations (Base, Yaw-Shifted, Depth-Scaled)" >> /home/james/Research/SEE/Logs/week2_results.log
echo "Using IC4 (Epistemic Weighting) + Calibrated tau=-1.0 as the base configuration." >> /home/james/Research/SEE/Logs/week2_results.log
echo "" >> /home/james/Research/SEE/Logs/week2_results.log

for tta in "bundle" "min_uncert" "mean_uncert"; do
    echo "------------------------------------------" >> /home/james/Research/SEE/Logs/week2_results.log
    echo "Testing Multi-View Strategy: $tta" >> /home/james/Research/SEE/Logs/week2_results.log
    echo "------------------------------------------" >> /home/james/Research/SEE/Logs/week2_results.log
    
    python unsup_kitti-c.py \
        --method evidential_hdc_tta \
        --chunked \
        --reset_per_corruption \
        --ic_method ic4 \
        --tau -1.0 \
        --kappa 15.0 \
        --mv_tta $tta \
        >> /home/james/Research/SEE/Logs/week2_results.log 2>&1
done

echo "Week 2 Multi-View TTA tests complete!"
