#!/bin/bash

# Convert the nuScenes mini dataset to the SemanticKITTI format.
python export_semantickitti.py nuscenes_gt_to_semantickitti \
--nusc_dir /mnt/data/nuscenes \
--nusc_skitti_dir /mnt/data/nuscenes_semantickitti

# Convert the nuScenes full dataset to the SemanticKITTI format.
python export_semantickitti.py nuscenes_gt_to_semantickitti \
--nusc_dir /mnt/data/nuscenes \
--nusc_skitti_dir /mnt/data/nuscenes_semantickitti --nusc_version v1.0-trainval