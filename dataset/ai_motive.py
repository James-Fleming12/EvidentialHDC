import os
import glob
import json
import numpy as np
import torch
import laspy
from torch.utils.data import Dataset
from tqdm import tqdm

from modules.HDC_utils import UQModel

CLASS_MAP = {
    "car": 0,
    "van": 1,
    "pickup": 2,
    "size_vehicle_m": 3,

    "truck": 4,
    "bus": 5,
    "truck/bus": 4, # ambiguous, map to truck
    "trailer": 6,
    "train": 7,
    "size_vehicle_xl": 4,# ambiguous, map to truck

    "motorcycle": 8,
    "bicycle": 9,
    "bike": 9, # same as bicycle

    "pedestrian": 10,
    "person": 10,

    "traffic_cone": 11,
    "barrier": 12,
    "misc": 13,
}
NUM_CLASSES = 14
NUM_CLASSES = len(set(CLASS_MAP.values()))

ALL_CONDITIONS = ["highway", "urban", "night", "rain"]
NORMAL_CONDITION = "highway"
ADVERSE_CONDITIONS = ["urban", "night", "rain"]

PC_RANGE = [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0]
BEV_SHAPE = (512, 512)
VOXEL_SIZE = [(PC_RANGE[3] - PC_RANGE[0]) / BEV_SHAPE[1], (PC_RANGE[4] - PC_RANGE[1]) / BEV_SHAPE[0], PC_RANGE[5] - PC_RANGE[2]]
MAX_POINTS_PER_VOXEL = 32
MAX_VOXELS = 10000

class AiMotiveDataset(Dataset):
    def __init__(self, root, split="train", conditions=None, val_fraction=0.2):
        self.root = root
        self.split = split
        self.conditions = conditions or ["highway", "night", "rain", "urban"]
        self.frames = []

        for cond in self.conditions:
            cond_path = os.path.join(root, split, cond)
            if not os.path.exists(cond_path):
                continue
                
            sequences = sorted(os.listdir(cond_path))

            split_idx = int(len(sequences) * (1 - val_fraction))
            target_seqs = sequences[:split_idx] if split == "train" else sequences[split_idx:]
            
            for seq in target_seqs:
                seq_path = os.path.join(cond_path, seq)

                lidar_dir = os.path.join(seq_path, "dynamic", "raw-revolutions")
                label_dir = os.path.join(seq_path, "dynamic", "box", "3d_body")
                
                if os.path.exists(lidar_dir):
                    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.laz")))
                    for lf in lidar_files:
                        json_name = os.path.basename(lf).replace('.laz', '.json')
                        self.frames.append({"lidar_path": lf, "label_path": os.path.join(label_dir, json_name), "condition": cond})

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]

        las = laspy.read(frame["lidar_path"])

        intensity = np.array(las.intensity, dtype=np.float32)
        if intensity.max() > 255.0:
            intensity = (intensity / 65535.0) * 255.0
            
        points = np.vstack((las.x, las.y, las.z, intensity)).transpose()

        labels = []
        if os.path.exists(frame["label_path"]):
            with open(frame["label_path"], 'r') as f:
                data = json.load(f)
                
            objects = data.get("CapturedObjects", []) 
            for obj in objects:
                actor_name = obj.get("ActorName", "").lower()
                cls_name = actor_name.split(" ")[0] if actor_name else ""
                
                if cls_name in CLASS_MAP:
                    labels.append(CLASS_MAP[cls_name])

        if len(labels) == 0:
            scene_label = -1
        else:
            scene_label = int(np.bincount(labels).argmax())
            
        return points, scene_label

def voxelize(points):
    """
    Groups (N, 4) points into pillars for the PointPillarEncoder.
    Returns: voxel_features (M, 32, 4), voxel_coords (M, 3)
    """
    mask = (
        (points[:, 0] >= PC_RANGE[0]) & (points[:, 0] <= PC_RANGE[3]) &
        (points[:, 1] >= PC_RANGE[1]) & (points[:, 1] <= PC_RANGE[4]) &
        (points[:, 2] >= PC_RANGE[2]) & (points[:, 2] <= PC_RANGE[5])
    )
    points = points[mask]

    voxel_coords = np.floor(
        (points[:, [0, 1, 2]] - np.array([PC_RANGE[0], PC_RANGE[1], PC_RANGE[2]])) / np.array(VOXEL_SIZE)
    ).astype(np.int32)

    voxel_coords = voxel_coords[:, [2, 1, 0]] 
    
    unique_coords, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
    
    num_voxels = min(len(unique_coords), MAX_VOXELS)
    
    voxel_features = np.zeros((num_voxels, MAX_POINTS_PER_VOXEL, 4), dtype=np.float32)
    final_coords = np.zeros((num_voxels, 3), dtype=np.int32)
    
    # 4. Populate voxels
    voxel_point_counts = np.zeros(num_voxels, dtype=np.int32)
    for i, voxel_idx in enumerate(inverse_indices):
        if voxel_idx >= num_voxels:
            continue
            
        count = voxel_point_counts[voxel_idx]
        if count < MAX_POINTS_PER_VOXEL:
            voxel_features[voxel_idx, count, :] = points[i]
            final_coords[voxel_idx, :] = unique_coords[voxel_idx]
            voxel_point_counts[voxel_idx] += 1

    return voxel_features, final_coords

def _parser_collate(batch):
    """
    Takes batch of (points, labels), applies voxelization, and 
    formats them for PointPillarEncoder.
    """
    batched_voxel_features = []
    batched_voxel_coords = []
    batched_labels = []
    
    for batch_idx, (points, scene_label) in enumerate(batch):
        v_feats, v_coords = voxelize(points)

        v_feats_tensor = torch.tensor(v_feats, dtype=torch.float32)
        v_coords_tensor = torch.tensor(v_coords, dtype=torch.long)

        batch_idx_tensor = torch.full((v_coords_tensor.shape[0], 1), batch_idx, dtype=torch.long)
        coords_with_batch = torch.cat([batch_idx_tensor, v_coords_tensor], dim=1)
        
        batched_voxel_features.append(v_feats_tensor)
        batched_voxel_coords.append(coords_with_batch)
        batched_labels.append(torch.tensor([scene_label], dtype=torch.long))

    final_voxel_features = torch.cat(batched_voxel_features, dim=0)
    final_voxel_coords = torch.cat(batched_voxel_coords, dim=0)

    proj_in = {
        "voxel_features": final_voxel_features,
        "voxel_coords": final_voxel_coords,
        "batch_size": len(batch)
    }

    proj_labels = torch.cat(batched_labels, dim=0)

    return proj_in, None, proj_labels, None, None, None, None, None, None, None, None, None, None, None, None

class AiMotiveEllipsoidTrainer:
    """
    Minimal stand-in for EllipsoidTrainer that works with aiMotive DataLoaders.
    Avoids the KITTI Parser entirely while reusing the HDC train/retrain logic.
    """
    def __init__(self, model: UQModel, num_classes: int, device: torch.device, bipolar_prototypes: bool = False):
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.gpu = device.type == "cuda"
        self.bipolar_prototypes = bipolar_prototypes
        self.is_wrong_list = []
        self.mask = None
        self.logger = None

    def reaccumulate_prototypes(self, train_loader):
        """Re-accumulate HDC class prototypes from scratch."""
        import torch.nn.functional as F
        print("Reaccumulating HDC prototypes...")
        self.model.eval()
        self.is_wrong_list = [None] * len(train_loader)

        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            self.model.classify_weights.data.fill_(0.0)
            self.model.classify.weight.data.fill_(0.0)

            for i, (proj_in, _, proj_labels, *_) in enumerate(
                    tqdm(train_loader, desc="Reaccumulating prototypes")):
                if isinstance(proj_in, dict):
                    proj_in = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in proj_in.items()}
                else:
                    proj_in = proj_in.to(self.device)

                proj_labels = proj_labels.to(self.device).flatten()

                samples_hv, _, _ = self.model.encode(proj_in, self.mask)
                samples_hv = samples_hv.to(self.model.classify_weights.dtype)

                valid = proj_labels >= 0
                if valid.any():
                    self.model.classify_weights.index_add_(
                        0, proj_labels[valid], samples_hv[valid])

                predictions = self.model.get_predictions(samples_hv)
                argmax = predictions.argmax(dim=1)
                self.is_wrong_list[i] = proj_labels != argmax

            if self.bipolar_prototypes:
                self.model.classify_weights.data = torch.sign(self.model.classify_weights.data)
                zero_mask = self.model.classify_weights.data == 0
                if zero_mask.any():
                    self.model.classify_weights.data[zero_mask] = -1.0
                self.model.classify.weight.data = self.model.classify_weights.data.clone()
            else:
                self.model.classify.weight[:] = F.normalize(self.model.classify_weights)

        print("Prototype reaccumulation complete.")

    def retrain(self, train_loader, model, epoch, logger):
        """One epoch of HDC retraining (mistake-driven weight correction)."""
        import torch.nn.functional as F
        total_miss = 0

        if len(self.is_wrong_list) != len(train_loader):
            self.is_wrong_list = [None] * len(train_loader)

        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            if self.bipolar_prototypes:
                model.classify_weights.data = torch.sign(model.classify_weights.data)
                zero_mask = model.classify_weights.data == 0
                if zero_mask.any():
                    model.classify_weights.data[zero_mask] = -1.0
                model.classify.weight.data = model.classify_weights.data.clone()
            else:
                model.classify.weight[:] = F.normalize(model.classify_weights)

            for i, (proj_in, _, proj_labels, *_) in enumerate(
                    tqdm(train_loader, desc=f"Retraining epoch {epoch}")):
                if isinstance(proj_in, dict):
                    proj_in = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                               for k, v in proj_in.items()}
                else:
                    proj_in = proj_in.to(self.device)

                proj_labels = proj_labels.to(self.device).flatten()

                samples_hv, _, _ = self.model.encode(proj_in, self.mask)
                samples_hv = samples_hv.to(model.classify_weights.dtype)

                predictions = self.model.get_predictions(samples_hv)
                argmax = predictions.argmax(dim=1)

                is_wrong = proj_labels != argmax
                valid = proj_labels >= 0
                is_wrong = is_wrong & valid

                if is_wrong.sum().item() == 0:
                    continue

                total_miss += is_wrong.sum().item()
                wrong_labels = proj_labels[is_wrong]
                wrong_preds = argmax[is_wrong]
                wrong_hvs = samples_hv[is_wrong].to(model.classify_weights.dtype)

                valid_pred_mask = (wrong_preds >= 0) & (wrong_preds < self.num_classes)
                wrong_labels = wrong_labels[valid_pred_mask]
                wrong_preds = wrong_preds[valid_pred_mask]
                wrong_hvs = wrong_hvs[valid_pred_mask]

                if len(wrong_labels) == 0:
                    continue

                model.classify_weights.index_add_(0, wrong_labels,  wrong_hvs)
                model.classify_weights.index_add_(0, wrong_preds,  -wrong_hvs)

                self.is_wrong_list[i] = is_wrong

        print(f"  Retrain epoch {epoch} — total misses: {total_miss}")