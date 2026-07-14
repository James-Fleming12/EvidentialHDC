import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from abc import ABC, abstractmethod

# DGLSS unified label mapping (20 classes)
DGLSS_LEARNING_MAP = {
    0: 0,   # "unlabeled"
    1: 1,   # "car"
    2: 2,   # "bicycle"
    3: 3,   # "motorcycle"
    4: 4,   # "truck"
    5: 5,   # "other-vehicle"
    6: 6,   # "person"
    7: 7,   # "bicyclist"
    8: 8,   # "motorcyclist"
    9: 9,   # "road"
    10: 10, # "parking"
    11: 11, # "sidewalk"
    12: 12, # "other-ground"
    13: 13, # "building"
    14: 14, # "fence"
    15: 15, # "vegetation"
    16: 16, # "trunk"
    17: 17, # "terrain"
    18: 18, # "pole"
    19: 19, # "traffic-sign"
}

DGLSS_LABELS = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",
}


class BaseLiDARDataset(ABC, Dataset):
    """Base class for all LiDAR datasets"""
    
    def __init__(self, root, sequences, max_points=150000, gt=True, transform=False):
        self.root = root
        self.sequences = sequences
        self.max_points = max_points
        self.gt = gt
        self.transform = transform
        
        # Unified parameters
        self.nclasses = 20  # DGLSS standard
        self.learning_map = DGLSS_LEARNING_MAP
        
        # Dataset-specific parameters (to be set by subclasses)
        self.sensor_img_H = None
        self.sensor_img_W = None
        self.sensor_img_means = None
        self.sensor_img_stds = None
        self.sensor_fov_up = None
        self.sensor_fov_down = None
        
        # File lists
        self.scan_files = []
        self.label_files = []
        
    @abstractmethod
    def _load_scan(self, scan_file):
        """Load raw point cloud data"""
        pass
    
    @abstractmethod
    def _load_label(self, label_file):
        """Load semantic labels"""
        pass
    
    @abstractmethod
    def _get_dataset_to_dglss_map(self):
        """Return mapping from dataset labels to DGLSS labels"""
        pass
    
    @abstractmethod
    def _project_points(self, points, remission):
        """Project 3D points to 2D range image"""
        pass
    
    def _apply_augmentation(self):
        """Determine augmentation flags"""
        if not self.transform:
            return False, False, False, 0.0
        
        if random.random() > 0.5:
            DA = random.random() > 0.5
            flip_sign = random.random() > 0.5
            rot = random.random() > 0.5
            drop_points = random.uniform(0, 0.5)
            return DA, flip_sign, rot, drop_points
        return False, False, False, 0.0
    
    def _map_labels(self, labels):
        """Map dataset labels to DGLSS labels"""
        dataset_map = self._get_dataset_to_dglss_map()
        maxkey = max(dataset_map.keys())
        lut = np.zeros(maxkey + 100, dtype=np.int32)
        for key, value in dataset_map.items():
            lut[key] = value
        return lut[labels]
    
    def __len__(self):
        return len(self.scan_files)


class SemanticKittiDataset(BaseLiDARDataset):
    """SemanticKITTI dataset with DGLSS mapping"""
    
    # SemanticKITTI to DGLSS mapping
    KITTI_TO_DGLSS = {
        0: 0,   # "unlabeled" -> unlabeled
        1: 0,   # "outlier" -> unlabeled
        10: 1,  # "car" -> car
        11: 2,  # "bicycle" -> bicycle
        15: 3,  # "motorcycle" -> motorcycle
        13: 3,  # "other-vehicle" -> motorcycle (approximate)
        18: 4,  # "truck" -> truck
        20: 5,  # "other-vehicle" -> other-vehicle
        30: 6,  # "person" -> person
        31: 7,  # "bicyclist" -> bicyclist
        32: 8,  # "motorcyclist" -> motorcyclist
        40: 9,  # "road" -> road
        44: 10, # "parking" -> parking
        48: 11, # "sidewalk" -> sidewalk
        49: 12, # "other-ground" -> other-ground
        50: 13, # "building" -> building
        51: 14, # "fence" -> fence
        70: 15, # "vegetation" -> vegetation
        71: 16, # "trunk" -> trunk
        72: 17, # "terrain" -> terrain
        80: 18, # "pole" -> pole
        81: 19, # "traffic-sign" -> traffic-sign
    }
    
    def __init__(self, root, sequences, max_points=150000, gt=True, transform=False):
        super().__init__(root, sequences, max_points, gt, transform)
        
        # SemanticKITTI sensor parameters
        self.sensor_img_H = 64
        self.sensor_img_W = 2048
        self.sensor_img_means = torch.tensor([12.12, 10.88, 0.23, -1.04, 0.21])
        self.sensor_img_stds = torch.tensor([12.32, 11.47, 6.91, 0.86, 0.16])
        self.sensor_fov_up = 3.0
        self.sensor_fov_down = -25.0
        
        # Load file lists
        self._load_file_lists()
    
    def _get_dataset_to_dglss_map(self):
        return self.KITTI_TO_DGLSS
    
    def _load_file_lists(self):
        root_sequences = os.path.join(self.root, "sequences")
        for seq in self.sequences:
            seq_str = '{0:02d}'.format(int(seq))
            scan_path = os.path.join(root_sequences, seq_str, "velodyne")
            label_path = os.path.join(root_sequences, seq_str, "labels")
            
            scan_files = sorted([os.path.join(scan_path, f) for f in os.listdir(scan_path) if f.endswith('.bin')])
            label_files = sorted([os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.label')])
            
            if self.gt:
                assert len(scan_files) == len(label_files)
            
            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)
        
        print(f"SemanticKITTI: Loaded {len(self.scan_files)} scans from sequences {self.sequences}")
    
    def _load_scan(self, scan_file):
        scan = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))
        points = scan[:, :3]
        remission = scan[:, 3]
        return points, remission
    
    def _load_label(self, label_file):
        label = np.fromfile(label_file, dtype=np.int32).reshape((-1))
        sem_label = label & 0xFFFF  # Lower 16 bits
        return sem_label
    
    def _project_points(self, points, remission):
        """Project to spherical coordinates"""
        depth = np.linalg.norm(points, 2, axis=1)
        
        # Get angles
        yaw = -np.arctan2(points[:, 1], points[:, 0])
        pitch = np.arcsin(points[:, 2] / (depth + 1e-8))
        
        # Get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(self.sensor_fov_down * np.pi / 180)) / (
            abs(self.sensor_fov_down * np.pi / 180) + abs(self.sensor_fov_up * np.pi / 180))
        
        # Scale to image size
        proj_x = np.floor(proj_x * self.sensor_img_W).astype(np.int32)
        proj_y = np.floor(proj_y * self.sensor_img_H).astype(np.int32)
        
        # Clamp
        proj_x = np.clip(proj_x, 0, self.sensor_img_W - 1)
        proj_y = np.clip(proj_y, 0, self.sensor_img_H - 1)
        
        # Create range image
        proj_range = np.full((self.sensor_img_H, self.sensor_img_W), -1.0, dtype=np.float32)
        proj_xyz = np.full((self.sensor_img_H, self.sensor_img_W, 3), -1.0, dtype=np.float32)
        proj_remission = np.full((self.sensor_img_H, self.sensor_img_W), -1.0, dtype=np.float32)
        proj_idx = np.full((self.sensor_img_H, self.sensor_img_W), -1, dtype=np.int32)
        proj_mask = np.zeros((self.sensor_img_H, self.sensor_img_W), dtype=np.int32)
        
        # Fill in
        order = np.argsort(depth)[::-1]
        for i in order:
            proj_range[proj_y[i], proj_x[i]] = depth[i]
            proj_xyz[proj_y[i], proj_x[i]] = points[i]
            proj_remission[proj_y[i], proj_x[i]] = remission[i]
            proj_idx[proj_y[i], proj_x[i]] = i
            proj_mask[proj_y[i], proj_x[i]] = 1
        
        return proj_range, proj_xyz, proj_remission, proj_mask, proj_x, proj_y, proj_idx
    
    def __getitem__(self, index):
        scan_file = self.scan_files[index]
        
        # Load data
        points, remission = self._load_scan(scan_file)
        
        # Apply augmentation (simplified - full implementation would need LaserScan class)
        # DA, flip_sign, rot, drop_points = self._apply_augmentation()
        
        # Project to range image
        proj_range, proj_xyz, proj_remission, proj_mask, proj_x, proj_y, proj_idx = self._project_points(points, remission)
        
        # Load and map labels
        if self.gt:
            label_file = self.label_files[index]
            sem_label = self._load_label(label_file)
            sem_label_mapped = self._map_labels(sem_label)
            
            # Project labels
            proj_sem_label = np.zeros((self.sensor_img_H, self.sensor_img_W), dtype=np.int32)
            for i in range(len(points)):
                proj_sem_label[proj_y[i], proj_x[i]] = sem_label_mapped[i]
            proj_labels = proj_sem_label * proj_mask
        else:
            sem_label_mapped = np.array([])
            proj_labels = np.array([])
        
        # Create unproj tensors
        unproj_n_points = points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(np.linalg.norm(points, 2, axis=1))
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(remission)
        
        if self.gt:
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(sem_label_mapped)
        else:
            unproj_labels = torch.tensor([])
        
        # Create proj tensors
        proj_x_tensor = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x_tensor[:unproj_n_points] = torch.from_numpy(proj_x)
        proj_y_tensor = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y_tensor[:unproj_n_points] = torch.from_numpy(proj_y)
        
        # Concatenate channels
        proj = torch.cat([
            torch.from_numpy(proj_range).unsqueeze(0),
            torch.from_numpy(proj_xyz).permute(2, 0, 1),
            torch.from_numpy(proj_remission).unsqueeze(0)
        ])
        
        # Normalize
        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]
        proj = proj * torch.from_numpy(proj_mask).float()
        
        # Get metadata
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")
        
        return (proj, torch.from_numpy(proj_mask), torch.from_numpy(proj_labels).long(),
                unproj_labels, path_seq, path_name, proj_x_tensor, proj_y_tensor,
                torch.from_numpy(proj_range), unproj_range, torch.from_numpy(proj_xyz),
                unproj_xyz, torch.from_numpy(proj_remission), unproj_remissions, unproj_n_points)


class WaymoDataset(BaseLiDARDataset):
    """Waymo Open Dataset with DGLSS mapping"""
    
    WAYMO_TO_DGLSS = {
        0: 0,   # UNDEFINED -> unlabeled
        1: 1,   # CAR -> car
        2: 4,   # TRUCK -> truck
        3: 5,   # BUS -> other-vehicle
        4: 6,   # OTHER_VEHICLE -> other-vehicle  
        5: 2,   # MOTORCYCLIST -> bicycle (approximate)
        6: 7,   # BICYCLIST -> bicyclist
        7: 6,   # PEDESTRIAN -> person
        8: 19,  # SIGN -> traffic-sign
        9: 19,  # TRAFFIC_LIGHT -> traffic-sign
        10: 18, # POLE -> pole
        11: 13, # CONSTRUCTION_CONE -> building (approximate)
        12: 2,  # BICYCLE -> bicycle
        13: 3,  # MOTORCYCLE -> motorcycle
        14: 13, # BUILDING -> building
        15: 15, # VEGETATION -> vegetation
        16: 16, # TREE_TRUNK -> trunk
        17: 0,  # CURB -> unlabeled (or could map to other-ground)
        18: 9,  # ROAD -> road
        19: 0,  # LANE_MARKER -> unlabeled
        20: 0,  # OTHER_GROUND -> other-ground
        21: 11, # WALKABLE -> sidewalk
        22: 11, # SIDEWALK -> sidewalk
    }
    
    def __init__(self, root, sequences, max_points=150000, gt=True, transform=False):
        super().__init__(root, sequences, max_points, gt, transform)
        
        # Waymo sensor parameters (top lidar)
        self.sensor_img_H = 64
        self.sensor_img_W = 2650
        self.sensor_img_means = torch.tensor([11.5, 9.2, 0.3, -0.9, 0.19])
        self.sensor_img_stds = torch.tensor([11.8, 10.5, 6.5, 0.82, 0.15])
        self.sensor_fov_up = 2.4
        self.sensor_fov_down = -17.6
        
        self._load_file_lists()
    
    def _get_dataset_to_dglss_map(self):
        return self.WAYMO_TO_DGLSS
    
    def _load_file_lists(self):
        # Waymo dataset structure: root/sequence/velodyne/*.bin and labels/*.label
        for seq in self.sequences:
            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")
            
            if os.path.exists(scan_path):
                scan_files = sorted([os.path.join(scan_path, f) for f in os.listdir(scan_path) if f.endswith('.bin')])
                self.scan_files.extend(scan_files)
                
                if self.gt and os.path.exists(label_path):
                    label_files = sorted([os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.label')])
                    self.label_files.extend(label_files)
        
        print(f"Waymo: Loaded {len(self.scan_files)} scans from sequences {self.sequences}")
    
    def _load_scan(self, scan_file):
        scan = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))
        points = scan[:, :3]
        remission = scan[:, 3]
        return points, remission
    
    def _load_label(self, label_file):
        label = np.fromfile(label_file, dtype=np.uint8).reshape((-1))
        return label
    
    def _project_points(self, points, remission):
        # Same projection as SemanticKITTI
        return SemanticKittiDataset._project_points(self, points, remission)
    
    def __getitem__(self, index):
        # Same as SemanticKITTI but with Waymo-specific processing
        return SemanticKittiDataset.__getitem__(self, index)


class NuScenesDataset(BaseLiDARDataset):
    """nuScenes dataset with DGLSS mapping"""
    
    NUSCENES_TO_DGLSS = {
        0: 0,   # noise -> unlabeled
        1: 13,  # barrier -> building (approximate)
        2: 2,   # bicycle -> bicycle
        3: 5,   # bus -> other-vehicle
        4: 1,   # car -> car
        5: 5,   # construction_vehicle -> other-vehicle
        6: 3,   # motorcycle -> motorcycle
        7: 6,   # pedestrian -> person
        8: 19,  # traffic_cone -> traffic-sign
        9: 5,   # trailer -> other-vehicle
        10: 4,  # truck -> truck
        11: 9,  # driveable_surface -> road
        12: 12, # other_flat -> other-ground
        13: 11, # sidewalk -> sidewalk
        14: 17, # terrain -> terrain
        15: 13, # manmade -> building
        16: 15, # vegetation -> vegetation
    }
    
    def __init__(self, root, sequences, max_points=150000, gt=True, transform=False):
        super().__init__(root, sequences, max_points, gt, transform)

        self.sensor_img_H = 32
        self.sensor_img_W = 1024
        self.sensor_img_means = torch.tensor([10.8, 8.9, 0.25, -0.95, 0.22])
        self.sensor_img_stds = torch.tensor([11.2, 10.1, 6.3, 0.79, 0.17])
        self.sensor_fov_up = 10.0
        self.sensor_fov_down = -30.0
        
        self._load_file_lists()
    
    def _get_dataset_to_dglss_map(self):
        return self.NUSCENES_TO_DGLSS
    
    def _load_file_lists(self):
        for seq in self.sequences:
            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")
            
            if os.path.exists(scan_path):
                scan_files = sorted([os.path.join(scan_path, f) for f in os.listdir(scan_path) if f.endswith('.bin')])
                self.scan_files.extend(scan_files)
                
                if self.gt and os.path.exists(label_path):
                    label_files = sorted([os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.label')])
                    self.label_files.extend(label_files)
        
        print(f"nuScenes: Loaded {len(self.scan_files)} scans from sequences {self.sequences}")
    
    def _load_scan(self, scan_file):
        scan = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 5))
        points = scan[:, :3]
        remission = scan[:, 3]
        return points, remission
    
    def _load_label(self, label_file):
        label = np.fromfile(label_file, dtype=np.uint8).reshape((-1))
        return label
    
    def _project_points(self, points, remission):
        return SemanticKittiDataset._project_points(self, points, remission)
    
    def __getitem__(self, index):
        return SemanticKittiDataset.__getitem__(self, index)


class SemanticPOSSDataset(BaseLiDARDataset):
    """SemanticPOSS dataset with DGLSS mapping"""
    POSS_TO_DGLSS = {
        0: 0,   # unlabeled -> unlabeled
        1: 6,   # people -> person
        2: 7,   # rider -> bicyclist
        3: 1,   # car -> car
        4: 5,   # trunk -> other-vehicle (approximate)
        5: 5,   # plants -> vegetation (approximate, may need adjustment)
        6: 19,  # traffic_sign -> traffic-sign
        7: 18,  # pole -> pole
        8: 0,   # garbage_can -> unlabeled
        9: 13,  # building -> building
        10: 14, # cone/stone -> fence (approximate)
        11: 14, # fence -> fence
        12: 2,  # bike -> bicycle
        13: 9,  # ground -> road
    }
    
    def __init__(self, root, sequences, max_points=150000, gt=True, transform=False):
        super().__init__(root, sequences, max_points, gt, transform)
        
        # SemanticPOSS sensor parameters
        self.sensor_img_H = 40
        self.sensor_img_W = 1800
        self.sensor_img_means = torch.tensor([11.3, 9.5, 0.28, -0.88, 0.21])
        self.sensor_img_stds = torch.tensor([11.6, 10.3, 6.7, 0.84, 0.16])
        self.sensor_fov_up = 7.0
        self.sensor_fov_down = -16.0
        
        self._load_file_lists()
    
    def _get_dataset_to_dglss_map(self):
        return self.POSS_TO_DGLSS
    
    def _load_file_lists(self):
        for seq in self.sequences:
            seq_str = '{0:02d}'.format(int(seq))
            scan_path = os.path.join(self.root, seq_str, "velodyne")
            label_path = os.path.join(self.root, seq_str, "labels")
            
            if os.path.exists(scan_path):
                scan_files = sorted([os.path.join(scan_path, f) for f in os.listdir(scan_path) if f.endswith('.bin')])
                self.scan_files.extend(scan_files)
                
                if self.gt and os.path.exists(label_path):
                    label_files = sorted([os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.label')])
                    self.label_files.extend(label_files)
        
        print(f"SemanticPOSS: Loaded {len(self.scan_files)} scans from sequences {self.sequences}")
    
    def _load_scan(self, scan_file):
        scan = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))
        points = scan[:, :3]
        remission = scan[:, 3]
        return points, remission
    
    def _load_label(self, label_file):
        label = np.fromfile(label_file, dtype=np.uint8).reshape((-1))
        return label
    
    def _project_points(self, points, remission):
        return SemanticKittiDataset._project_points(self, points, remission)
    
    def __getitem__(self, index):
        return SemanticKittiDataset.__getitem__(self, index)


class UniversalLiDARParser:
    """Universal parser for multiple LiDAR datasets with DGLSS mapping"""
    
    DATASET_CLASSES = {
        'semantickitti': SemanticKittiDataset,
        'waymo': WaymoDataset,
        'nuscenes': NuScenesDataset,
        'semanticposs': SemanticPOSSDataset,
    }

    def __init__(self,
                 dataset_type,      # 'semantickitti', 'waymo', 'nuscenes', 'semanticposs'
                 root,              # root directory
                 train_sequences,   # sequences to train
                 valid_sequences,   # sequences to validate
                 test_sequences=None,    # sequences to test
                 max_points=150000, # max points in scan
                 batch_size=2,      # batch size
                 workers=4,         # num workers
                 gt=True,           # get ground truth?
                 shuffle_train=True):
        
        self.dataset_type = dataset_type.lower()
        if self.dataset_type not in self.DATASET_CLASSES:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Must be one of {list(self.DATASET_CLASSES.keys())}")
        
        self.root = root
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.max_points = max_points
        self.batch_size = batch_size
        self.workers = workers
        self.gt = gt
        self.shuffle_train = shuffle_train

        self.nclasses = 20
        self.labels = DGLSS_LABELS

        DatasetClass = self.DATASET_CLASSES[self.dataset_type]

        self.train_dataset = DatasetClass(
            root=self.root,
            sequences=self.train_sequences,
            max_points=self.max_points,
            gt=self.gt,
            transform=True
        )
        
        self.valid_dataset = DatasetClass(
            root=self.root,
            sequences=self.valid_sequences,
            max_points=self.max_points,
            gt=self.gt,
            transform=False
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        g = torch.Generator()
        g.manual_seed(1024)
        
        self.trainloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.workers,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True
        )
        
        self.validloader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            drop_last=True
        )

        if self.test_sequences:
            self.test_dataset = DatasetClass(
                root=self.root,
                sequences=self.test_sequences,
                max_points=self.max_points,
                gt=False,
                transform=False
            )
            
            self.testloader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                drop_last=True
            )
        
        print(f"Initialized {self.dataset_type} parser with DGLSS mapping (20 classes)")
        print(f"Train: {len(self.train_dataset)} samples, Valid: {len(self.valid_dataset)} samples")
        if self.test_sequences:
            print(f"Test: {len(self.test_dataset)} samples")
    
    def get_train_set(self):
        return self.trainloader
    
    def get_valid_set(self):
        return self.validloader
    
    def get_test_set(self):
        if self.test_sequences:
            return self.testloader
        return None
    
    def get_train_size(self):
        return len(self.trainloader)
    
    def get_valid_size(self):
        return len(self.validloader)
    
    def get_test_size(self):
        if self.test_sequences:
            return len(self.testloader)
        return 0
    
    def get_n_classes(self):
        return self.nclasses
    
    def get_class_string(self, idx):
        """Get DGLSS class name"""
        return self.labels.get(idx, "unknown")

def main():
    pass

if __name__ == "__main__":
    # Example: Create parser for SemanticKITTI
    # kitti_parser = UniversalLiDARParser(
    #     dataset_type='semantickitti',
    #     root='/path/to/semantickitti',
    #     train_sequences=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    #     valid_sequences=[8],
    #     batch_size=2,
    #     workers=4
    # )

    main()