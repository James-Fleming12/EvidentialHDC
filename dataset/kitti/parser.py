import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan
import torchvision

import torch
import math
import random
from PIL import Image

from dataset.waymo_data import WaymoDataset
import torch.utils.data as data
import numpy as np
import numbers
import types
from collections.abc import Sequence, Iterable
import warnings

CURRICULUM_PHASES = {
    0: ["sunny"],
    1: ["rain", "fog", "night"],
    2: ["sunny", "rain", "fog", "night"],   # i.e. no filter = all
}

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

def _make_seed_worker():
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker

def _make_loader(dataset, batch_size, shuffle, workers, drop_last=True, generator=None):
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        worker_init_fn=_make_seed_worker(),
        generator=generator,
        drop_last=drop_last,
    )

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def my_collate(batch):
    data = [item[0] for item in batch]
    project_mask = [item[1] for item in batch]
    proj_labels = [item[2] for item in batch]
    data = torch.stack(data,dim=0)
    project_mask = torch.stack(project_mask,dim=0)
    proj_labels = torch.stack(proj_labels, dim=0)

    to_augment =(proj_labels == 12).nonzero()
    to_augment_unique_12 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 5).nonzero()
    to_augment_unique_5 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 8).nonzero()
    to_augment_unique_8 = torch.unique(to_augment[:, 0])

    to_augment_unique = torch.cat((to_augment_unique_5,to_augment_unique_8,to_augment_unique_12),dim=0)
    to_augment_unique = torch.unique(to_augment_unique)

    for k in to_augment_unique:
        data = torch.cat((data,torch.flip(data[k.item()], [2]).unsqueeze(0)),dim=0)
        proj_labels = torch.cat((proj_labels,torch.flip(proj_labels[k.item()], [1]).unsqueeze(0)),dim=0)
        project_mask = torch.cat((project_mask,torch.flip(project_mask[k.item()], [1]).unsqueeze(0)),dim=0)

    return data, project_mask,proj_labels

class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               max_points=150000,   # max number of points present in dataset
               gt=True,
               transform=False):            # send ground truth?
    # save deats
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt
    self.transform = transform

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []

    # fill in with names, checking that all sequences are complete
    print("sequences: ", self.sequences)
    for seq_idx in self.sequences:
      # Try 2-digit padding (Official KITTI) first, then 4-digit (NuScenes-Kitti)
      seq_2 = '{0:02d}'.format(int(seq_idx))
      seq_4 = '{0:04d}'.format(int(seq_idx))
      
      # Determine which path exists
      if os.path.exists(os.path.join(self.root, seq_2)):
          seq = seq_2
      elif os.path.exists(os.path.join(self.root, seq_4)):
          seq = seq_4
      else:
          print(f"Warning: Could not find sequence folder for {seq_idx} (tried {seq_2} and {seq_4})")
          continue

      # get paths for each
      scan_path = os.path.join(self.root, seq, "velodyne")
      label_path = os.path.join(self.root, seq, "labels")
      # print(f"scan_path: {scan_path}")

      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]
      print("Found {} scans in sequence {}".format(len(scan_files), seq))
      # check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))

      # extend list
      self.scan_files.extend(scan_files)
      self.label_files.extend(label_files)

    # sort for correspondance
    self.scan_files.sort()
    self.label_files.sort()

    print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                    self.sequences))

  def __getitem__(self, index):
    # get item in tensor shape
    scan_file = self.scan_files[index]
    if self.gt:
      label_file = self.label_files[index]

    # open a semantic laserscan
    DA = False
    flip_sign = False
    rot = False
    drop_points = False
    if self.transform:
        if random.random() > 0.5:
            if random.random() > 0.5:
                DA = True
            if random.random() > 0.5:
                flip_sign = True
            if random.random() > 0.5:
                rot = True
            drop_points = random.uniform(0, 0.5)

    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down,
                          DA=DA,
                          flip_sign=flip_sign,
                          rot=rot,
                          drop_points=drop_points)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down,
                       DA=DA,
                       flip_sign=flip_sign,
                       rot=rot,
                       drop_points=drop_points)

    # open and obtain scan
    scan.open_scan(scan_file)
    if self.gt:
      scan.open_label(label_file)
      # map unused classes to used classes (also for projection)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    # make a tensor of the uncompressed data (with the max num points)
    unproj_n_points = scan.points.shape[0]
    unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
    unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
    unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
    unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
    if self.gt:
      unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
      unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
    else:
      unproj_labels = []

    # get points and labels
    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()

#     proj_normal = torch.from_numpy(scan.normal_image).clone()

    proj_mask = torch.from_numpy(scan.proj_mask)
    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []
    proj_x = torch.full([self.max_points], -1, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
    proj_y = torch.full([self.max_points], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])

#     proj = torch.cat([proj_range.unsqueeze(0).clone(),
#                       proj_xyz.clone().permute(2, 0, 1),
#                       proj_remission.unsqueeze(0).clone(),
#                       proj_normal.unsqueeze(0).clone()])

    proj = (proj - self.sensor_img_means[:, None, None]
            ) / self.sensor_img_stds[:, None, None]

    proj = proj * proj_mask.float()

    # get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")

    # return

    #debug
    #unique_labels = torch.unique(proj_labels)
    #print(f"[DEBUG] Sample {index} → unique labels: {unique_labels.tolist()}")

    return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

  def __len__(self):
    return len(self.scan_files)

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

class Parser:
    """
    Unified parser that initializes as exactly one of two modes.
 
    mode="kitti"  (default)
        Wraps SemanticKitti. Identical behaviour to the original Parser.
        set_curriculum_phase() is silently ignored.
 
    mode="waymo"
        Wraps WaymoDataset with curriculum weather scheduling.
        root / train_sequences / valid_sequences point at the converted
        Waymo sequences/ directory and its 4-digit sequence IDs.
        Call set_curriculum_phase(0/1/2) from Trainer to advance the schedule.
 
    All constructor arguments are the same in both modes so Trainer only
    needs to change mode="kitti" → mode="waymo" (and update root/sequences).
 
    Args:
        mode              : "kitti" or "waymo"
        root              : Dataset root (contains sequences/).
        train_sequences   : List[int] sequence IDs for training.
        valid_sequences   : List[int] sequence IDs for validation.
        test_sequences    : List[int] sequence IDs for test, or None.
        labels            : {id: name} dict.
        color_map         : {id: [B,G,R]} dict.
        learning_map      : {original_id: xentropy_id} dict.
        learning_map_inv  : {xentropy_id: original_id} dict.
        sensor            : Sensor config dict.
        max_points        : Max points per scan.
        batch_size        : Batch size for all loaders.
        workers           : DataLoader worker threads.
        gt                : Load ground-truth labels.
        shuffle_train     : Shuffle training loader.
        initial_curriculum: Starting curriculum phase (Waymo only, default 0).
    """
    def __init__(self,
                 root: str,
                 train_sequences: list,
                 valid_sequences: list,
                 test_sequences,
                 labels: dict,
                 color_map: dict,
                 learning_map: dict,
                 learning_map_inv: dict,
                 sensor: dict,
                 max_points: int,
                 batch_size: int,
                 workers: int,
                 mode: str = "kitti",
                 gt: bool = True,
                 shuffle_train: bool = True,
                 initial_curriculum: int = 0):
        if mode not in ("kitti", "waymo"):
            raise ValueError(f"mode must be 'kitti' or 'waymo', got '{mode}'")
 
        self._mode             = mode
        self._root             = root
        self._train_seqs       = train_sequences
        self._valid_seqs       = valid_sequences
        self._test_seqs        = test_sequences
        self._labels           = labels
        self._color_map        = color_map
        self._learning_map     = learning_map
        self._learning_map_inv = learning_map_inv
        self._sensor           = sensor
        self._max_points       = max_points
        self._batch_size       = batch_size
        self._workers          = workers
        self._gt               = gt
        self._shuffle_train    = shuffle_train
        self._curriculum_phase = -1
 
        self.nclasses = len(learning_map_inv)
 
        self._g = torch.Generator()
        self._g.manual_seed(1024)
 
        if mode == "kitti":
            self._build_kitti_loaders()
        else:
            self.set_curriculum_phase(initial_curriculum)

    def _build_kitti_loaders(self):
        def _ds(seqs, transform=False, gt=None):
            return SemanticKitti(
                root=self._root,
                sequences=seqs,
                labels=self._labels,
                color_map=self._color_map,
                learning_map=self._learning_map,
                learning_map_inv=self._learning_map_inv,
                sensor=self._sensor,
                max_points=self._max_points,
                transform=transform,
                gt=self._gt if gt is None else gt,
            )
 
        train_ds = _ds(self._train_seqs, transform=True)
        valid_ds = _ds(self._valid_seqs)
 
        self.trainloader = _make_loader(train_ds, self._batch_size, shuffle=self._shuffle_train, workers=self._workers, generator=self._g)
        assert len(self.trainloader) > 0, "KITTI train loader is empty!"
        self.trainiter = iter(self.trainloader)
 
        self.validloader = _make_loader(
            valid_ds, self._batch_size,
            shuffle=False, workers=self._workers)
        assert len(self.validloader) > 0, "KITTI valid loader is empty!"
        self.validiter = iter(self.validloader)
 
        if self._test_seqs:
            test_ds = _ds(self._test_seqs, gt=False)
            self.testloader = _make_loader(test_ds, self._batch_size, shuffle=False, workers=self._workers, drop_last=False)
            self.testiter = iter(self.testloader)
        else:
            self.testloader = None
            self.testiter = None
 
        print(f"[Parser|kitti] train={len(train_ds)}  valid={len(valid_ds)}")

    def set_curriculum_phase(self, phase: int):
        """
        Switch curriculum phase.
 
        In KITTI mode this is a no-op — safe to call unconditionally from
        Trainer regardless of which mode is active.
 
        Phase 0 → sunny frames only           base model warm-up
        Phase 1 → rain / fog / night only     adverse-condition fine-tuning
        Phase 2 → all conditions              full robustness
 
        Rebuilds train and validation loaders in-place.
        Test loader is built once at phase 0 with NO weather filter so the
        held-out evaluation always covers all conditions.
        """
        if self._mode == "kitti":
            return
 
        if phase == self._curriculum_phase:
            return
        if phase not in CURRICULUM_PHASES:
            raise ValueError(f"phase must be 0, 1, or 2; got {phase}")
 
        self._curriculum_phase = phase
        weather_filter = CURRICULUM_PHASES[phase]
 
        print(f"[Parser|waymo] Curriculum phase {phase} → weather filter: {weather_filter if weather_filter else 'ALL'}")
 
        def _ds(seqs, transform=False, gt=None, override_filter=None):
            return WaymoDataset(
                root=self._root,
                sequences=seqs,
                labels=self._labels,
                color_map=self._color_map,
                learning_map=self._learning_map,
                learning_map_inv=self._learning_map_inv,
                sensor=self._sensor,
                max_points=self._max_points,
                transform=transform,
                gt=self._gt if gt is None else gt,
                weather_filter=override_filter if override_filter is not None else weather_filter,
            )
 
        train_ds = _ds(self._train_seqs, transform=True)
        valid_ds = _ds(self._valid_seqs)
 
        if len(train_ds) == 0:
            raise RuntimeError(f"Waymo train dataset is empty for phase {phase} (filter={weather_filter}). Check weather.txt files or sequence IDs.")
        if len(valid_ds) == 0:
            raise RuntimeError(
                f"Waymo valid dataset is empty for phase {phase} (filter={weather_filter}).")
 
        self.trainloader = _make_loader(train_ds, self._batch_size, shuffle=self._shuffle_train, workers=self._workers, generator=self._g)
        self.trainiter = iter(self.trainloader)
 
        self.validloader = _make_loader(valid_ds, self._batch_size, shuffle=False, workers=self._workers)
        self.validiter = iter(self.validloader)

        if not hasattr(self, "testloader"):
            if self._test_seqs:
                test_ds = WaymoDataset(
                    root=self._root,
                    sequences=self._test_seqs,
                    labels=self._labels,
                    color_map=self._color_map,
                    learning_map=self._learning_map,
                    learning_map_inv=self._learning_map_inv,
                    sensor=self._sensor,
                    max_points=self._max_points,
                    transform=False,
                    gt=False,
                    weather_filter=None,
                )
                self.testloader = _make_loader(test_ds, self._batch_size, shuffle=False, workers=self._workers, drop_last=False)
                self.testiter = iter(self.testloader)
            else:
                self.testloader = None
                self.testiter   = None
 
        print(f"[Parser|waymo] phase={phase}  train={len(train_ds)}  valid={len(valid_ds)}")

    def get_train_batch(self):
        return next(self.trainiter)
 
    def get_train_set(self):
        return self.trainloader
 
    def get_valid_batch(self):
        return next(self.validiter)
 
    def get_valid_set(self):
        return self.validloader
 
    def get_test_batch(self):
        return next(self.testiter)
 
    def get_test_set(self):
        return self.testloader
 
    def get_train_size(self):
        return len(self.trainloader)
 
    def get_valid_size(self):
        return len(self.validloader)
 
    def get_test_size(self):
        return len(self.testloader) if self.testloader else 0
 
    def get_n_classes(self):
        return self.nclasses
 
    def get_original_class_string(self, idx):
        return self._labels[idx]
 
    def get_xentropy_class_string(self, idx):
        return self._labels[self._learning_map_inv[idx]]
 
    def to_original(self, label):
        return SemanticKitti.map(label, self._learning_map_inv)
 
    def to_xentropy(self, label):
        return SemanticKitti.map(label, self._learning_map)
 
    def to_color(self, label):
        label = SemanticKitti.map(label, self._learning_map_inv)
        return SemanticKitti.map(label, self._color_map)