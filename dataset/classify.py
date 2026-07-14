from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data as torchdata

def v2xr_collate(batch):
    """
    Stack voxel dicts, labels, and conditions from a V2XR batch.
    Because each sample has a different number of voxels we keep them as a
    list inside the dict — PointPillarEncoder's scatter step handles this
    via batch_idx in voxel_coords.
    """
    voxel_feats_list = []
    voxel_coords_list = []
    labels = []
    conditions = []

    for b_idx, (vdict, lbl, cond) in enumerate(batch):
        vf = vdict["voxel_features"]
        vc = vdict["voxel_coords"].clone()
        vc[:, 0] = b_idx
        voxel_feats_list.append(vf)
        voxel_coords_list.append(vc)
        labels.append(lbl)
        conditions.append(cond)

    voxel_batch = {
        "voxel_features": torch.cat(voxel_feats_list, dim=0),
        "voxel_coords": torch.cat(voxel_coords_list, dim=0),
        "batch_size": len(batch),
    }
    return voxel_batch, torch.stack(labels), conditions


def kradar_collate(batch):
    """Stack (radar_tensor, label, condition) tuples."""
    radars = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    conditions = [b[2] for b in batch]
    return radars, labels, conditions

class V2XRDataset(torchdata.Dataset):
    """
    Wrapper for V2X-R (Vehicle-to-Everything with Radar).

    Expected directory layout
    -------------------------
    data_dir/
      {split}/          e.g. train/, val/, test/
        {condition}/    e.g. sunny/, rain/, night/
          {seq_id}/
            lidar/      *.bin  (N x 4  float32  [x, y, z, intensity])
            label/      *.txt  (KITTI-style per-object labels)

    Parameters
    ----------
    data_dir: root of the V2X-R dataset
    split: 'train' | 'val' | 'test'
    conditions: list of weather strings to include, e.g. ['sunny']
    label_map: dict mapping raw string labels → integer class IDs
    voxel_cfg: dict with keys 'voxel_size', 'point_cloud_range', 'max_points'
    """

    # Map V2X-R string labels to integer class indices.
    # Adjust to match your actual annotation vocabulary.
    DEFAULT_LABEL_MAP: Dict[str, int] = {
        "Car": 0,
        "Truck": 1,
        "Bus": 2,
        "Cyclist": 3,
        "Pedestrian": 4,
        "Motorcycle": 5,
        "Background": 6,
    }

    ALL_CONDITIONS = ["sunny", "rain", "night"]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        conditions: Optional[List[str]] = None,
        label_map: Optional[Dict[str, int]] = None,
        voxel_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.conditions = conditions or self.ALL_CONDITIONS
        self.label_map = label_map or self.DEFAULT_LABEL_MAP
        self.num_classes = max(self.label_map.values()) + 1

        self.voxel_size = (voxel_cfg or {}).get("voxel_size", (0.1, 0.1, 0.2))
        self.point_cloud_range = (voxel_cfg or {}).get("point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        self.max_points = (voxel_cfg or {}).get("max_points", 35000)

        self.samples: List[dict] = self._index_samples()

    def _index_samples(self) -> List[dict]:
        """Walk the directory tree and collect (lidar_path, label_path, condition)."""
        samples = []
        for cond in self.conditions:
            cond_root = os.path.join(self.data_dir, self.split, cond)
            if not os.path.isdir(cond_root):
                continue
            for seq_id in sorted(os.listdir(cond_root)):
                lidar_dir = os.path.join(cond_root, seq_id, "lidar")
                label_dir = os.path.join(cond_root, seq_id, "label")
                if not os.path.isdir(lidar_dir):
                    continue
                for fname in sorted(os.listdir(lidar_dir)):
                    if not fname.endswith(".bin"):
                        continue
                    stem = fname[:-4]
                    label_path = os.path.join(label_dir, stem + ".txt")
                    samples.append({
                        "lidar": os.path.join(lidar_dir, fname),
                        "label": label_path,
                        "condition": cond,
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        voxel_batch : dict with keys
            'voxel_features'  (N_voxels, max_pts_per_voxel, 4)
            'voxel_coords'    (N_voxels, 4)   [0, z, y, x]
            'batch_size'      1
        label       : torch.LongTensor  shape ()  – scalar class ID
        condition   : str
        """
        s = self.samples[idx]

        points = np.fromfile(s["lidar"], dtype=np.float32).reshape(-1, 4)

        voxel_features, voxel_coords = self._voxelise(points)

        voxel_batch = {
            "voxel_features": torch.from_numpy(voxel_features),
            "voxel_coords": torch.from_numpy(voxel_coords),
            "batch_size": 1,
        }

        label = self._load_label(s["label"])

        return voxel_batch, torch.tensor(label, dtype=torch.long), s["condition"]

    def _voxelise(self, points: np.ndarray, max_voxels: int = 20000, max_pts: int = 32):
        """
        Simple numpy voxelisation — replace with your preferred implementation
        (e.g. OpenPCDet's voxel generator) for production use.
        """
        pc_min = np.array(self.point_cloud_range[:3], dtype=np.float32)
        pc_max = np.array(self.point_cloud_range[3:], dtype=np.float32)
        vsize = np.array(self.voxel_size, dtype=np.float32)

        mask = np.all((points[:, :3] >= pc_min) & (points[:, :3] < pc_max), axis=1)
        points = points[mask]

        if len(points) == 0:
            vf = np.zeros((1, max_pts, 4), dtype=np.float32)
            vc = np.zeros((1, 4), dtype=np.int32)
            return vf, vc

        voxel_idx = np.floor((points[:, :3] - pc_min) / vsize).astype(np.int32)
        keys = voxel_idx[:, 0] * 1_000_000 + voxel_idx[:, 1] * 1_000 + voxel_idx[:, 2]

        unique_keys, inverse = np.unique(keys, return_inverse=True)
        if len(unique_keys) > max_voxels:
            unique_keys = unique_keys[:max_voxels]
            keep = inverse < max_voxels
            points = points[keep]
            inverse = inverse[keep]

        n_vox = len(unique_keys)
        vf = np.zeros((n_vox, max_pts, 4), dtype=np.float32)
        cnt = np.zeros(n_vox, dtype=np.int32)

        for pi, vi in enumerate(inverse):
            if cnt[vi] < max_pts:
                vf[vi, cnt[vi]] = points[pi]
                cnt[vi] += 1

        # Build coords [batch=0, z, y, x]
        grid_idx = np.stack([
            unique_keys // 1_000_000,
            (unique_keys % 1_000_000) // 1_000,
            unique_keys % 1_000,
        ], axis=1).astype(np.int32)
        vc = np.concatenate(
            [np.zeros((n_vox, 1), dtype=np.int32), grid_idx[:, [2, 1, 0]]],
            axis=1,
        )
        return vf, vc

    def _load_label(self, label_path: str) -> int:
        """
        Read KITTI-style label file and return the most frequent class ID.
        Falls back to 0 if the file is missing.
        """
        if not os.path.isfile(label_path):
            return 0

        class_counts: Dict[int, int] = {}
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_str = parts[0]
                cls_id  = self.label_map.get(cls_str, -1)
                if cls_id >= 0:
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        if not class_counts:
            return 0
        return max(class_counts, key=class_counts.__getitem__)

class KRadarDataset(torchdata.Dataset):
    """
    Wrapper for K-Radar (4-D Radar Dataset for Adverse Weather Conditions).

    K-Radar provides a (Range × Azimuth × Elevation × Doppler) radar power
    spectrum together with per-frame 3-D bounding boxes and class labels.

    Expected directory layout
    -------------------------
    data_dir/
      {sequence_id}/
        radar_zyx_cube/   *.npy  (R × A × E × D  float32 power cube)
        info/             *.pkl  per-frame metadata including weather tag
        label/            *.txt  per-frame object labels

    Weather tags (adjust to match your version of K-Radar):
        'normal'  → maps to 'clear' (treated as the normal condition)
        'rain'    → adverse
        'sleet'   → adverse
        'snow'    → adverse
        'fog'     → adverse

    Parameters
    ----------
    data_dir      : root of the K-Radar dataset
    split         : 'train' | 'val' | 'test'
    conditions    : list of weather strings to include
    label_map     : dict mapping raw string labels → integer class IDs
    """

    DEFAULT_LABEL_MAP: Dict[str, int] = {
        "Sedan":         0,
        "Bus or Truck":  1,
        "Motorcycle":    2,
        "Bicycle":       3,
        "Pedestrian":    4,
        "Background":    5,
    }

    WEATHER_TAG_MAP: Dict[str, str] = {
        "normal": "clear",
        "rain":   "rain",
        "sleet":  "sleet",
        "snow":   "snow",
        "fog":    "fog",
    }

    ALL_CONDITIONS = ["clear", "rain", "sleet", "snow"]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        conditions: Optional[List[str]] = None,
        label_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.conditions = conditions or self.ALL_CONDITIONS
        self.label_map = label_map or self.DEFAULT_LABEL_MAP
        self.num_classes = max(self.label_map.values()) + 1

        self.samples: List[dict] = self._index_samples()

    def _index_samples(self) -> List[dict]:
        import pickle
        samples = []
        split_file = os.path.join(self.data_dir, f"{self.split}.txt")

        if os.path.isfile(split_file):
            with open(split_file) as f:
                seq_ids = [l.strip() for l in f if l.strip()]
        else:
            seq_ids = sorted(os.listdir(self.data_dir))

        for seq_id in seq_ids:
            radar_dir = os.path.join(self.data_dir, seq_id, "radar_zyx_cube")
            info_dir = os.path.join(self.data_dir, seq_id, "info")
            label_dir = os.path.join(self.data_dir, seq_id, "label")
            if not os.path.isdir(radar_dir):
                continue

            for fname in sorted(os.listdir(radar_dir)):
                if not fname.endswith(".npy"):
                    continue
                stem = fname[:-4]
                info_path  = os.path.join(info_dir,  stem + ".pkl")
                label_path = os.path.join(label_dir, stem + ".txt")

                # Read weather tag from info file
                weather = "clear"
                if os.path.isfile(info_path):
                    with open(info_path, "rb") as f:
                        info = pickle.load(f)
                    raw_tag = info.get("weather", "normal")
                    weather = self.WEATHER_TAG_MAP.get(raw_tag, "clear")

                if weather not in self.conditions:
                    continue

                samples.append({
                    "radar": os.path.join(radar_dir, fname),
                    "label": label_path,
                    "condition": weather,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        radar_tensor : (1, R, A)  float32  – range-azimuth power (collapsed)
        label        : torch.LongTensor  shape ()
        condition    : str
        """
        s = self.samples[idx]

        # Load and collapse the 4-D radar cube  (R, A, E, D) → (1, R, A)
        cube = np.load(s["radar"]).astype(np.float32)  # (R, A, E, D)
        ra   = cube.max(axis=-1).max(axis=-1)           # (R, A)
        # Normalise to [0, 1]
        ra   = (ra - ra.min()) / (ra.max() - ra.min() + 1e-8)
        radar_tensor = torch.from_numpy(ra).unsqueeze(0)  # (1, R, A)

        label = self._load_label(s["label"])
        return radar_tensor, torch.tensor(label, dtype=torch.long), s["condition"]

    def _load_label(self, label_path: str) -> int:
        if not os.path.isfile(label_path):
            return 0

        class_counts: Dict[int, int] = {}
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_str = parts[0]
                cls_id  = self.label_map.get(cls_str, -1)
                if cls_id >= 0:
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        if not class_counts:
            return 0
        return max(class_counts, key=class_counts.__getitem__)