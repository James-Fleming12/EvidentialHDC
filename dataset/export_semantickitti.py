import json
import os
from typing import List, Dict, Any
from pprint import pprint

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from nuscenes.utils.data_io import load_bin_file
from nuscenes.eval.lidarseg.utils import LidarsegClassMapper

class KittiConverter:
    def __init__(self,
                 nusc_dir: str,
                 nusc_skitti_dir: str,
                 lidar_name: str = 'LIDAR_TOP',
                 nusc_version: str = 'v1.0-mini',
                #  split: str = 'mini_train',
                 ):
        """
        :param nusc_skitti_dir: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export. Note that only one camera is allowed in KITTI.
        :param lidar_name: Name of the lidar sensor.
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """
        self.nusc_skitti_dir = os.path.expanduser(nusc_skitti_dir)
        self.lidar_name = lidar_name
        self.nusc_version = nusc_version
        # self.split = split

        # Create nusc_skitti_dir.
        if not os.path.isdir(self.nusc_skitti_dir):
            os.makedirs(self.nusc_skitti_dir)

        # Select subset of the data to look at.
        self.nusc = NuScenes(version=nusc_version, dataroot=nusc_dir)

    def nuscenes_gt_to_semantickitti(self):
        """
        Converts nuScenes GT panoptic annotations to SemanticKITTI format.
        """
        nu_to_kitti_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        nu_to_kitti_lidar_inv = nu_to_kitti_lidar.inverse

        ### Target directory structure
        seqs_folder = os.path.join(self.nusc_skitti_dir, 'sequences')
        ### In each seq, there are 2 folders: 'velodyne' and 'labels', 3 files: 'calib.txt', 'poses.txt', 'times.txt'

        # ### Load categories and their indices
        # categories = self.nusc.category
        # class_mapper = LidarsegClassMapper(self.nusc)
        # fine2coarse = class_mapper.fine_idx_2_coarse_idx_mapping
        # pprint(fine2coarse)
        
        # ### Print the colormap
        # for key in class_mapper.fine_name_2_coarse_name_mapping:
        #     find_idx = self.nusc.lidarseg_name2idx_mapping[key]
        #     print(find_idx, ':', list(class_mapper.coarse_colormap[class_mapper.fine_name_2_coarse_name_mapping[key]])[::-1] )

        # ### Print the scene splits
        # scene_splits = create_splits_scenes(verbose=False)
        # for key in scene_splits:
        #     scene_splits[key] = [int(x[6:]) for x in scene_splits[key]]
        # print(scene_splits)

        ### Iterate over scenes
        for scene_idx, scene in enumerate(self.nusc.scene):
            print(f'Converting scene {scene_idx} out of {len(self.nusc.scene)}: {scene["name"]}')

            name_idx = int(scene['name'][6:])
            ### Create sequence folder
            seq_folder = os.path.join(seqs_folder, f'{name_idx:04d}')
            if not os.path.exists(seq_folder):
                os.makedirs(seq_folder)
            ### Create subfolders
            velo_folder = os.path.join(seq_folder, 'velodyne')
            label_folder = os.path.join(seq_folder, 'labels')
            if not os.path.exists(velo_folder):
                os.makedirs(velo_folder)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
            ### Create files
            calib_file = os.path.join(seq_folder, 'calib.txt')
            pose_file = os.path.join(seq_folder, 'poses.txt')
            times_file = os.path.join(seq_folder, 'times.txt')
            calib_f = open(calib_file, 'w')
            pose_f = open(pose_file, 'w')
            times_f = open(times_file, 'w')

            ### Locate the first sample of the scene
            sample_token = scene['first_sample_token']

            ### Write calibration file
            ### Get the calibrated sensor pose relative to the ego vehicle
            sample = self.nusc.get('sample', sample_token)
            lidar_data_token = sample['data'][self.lidar_name]
            lidar_data = self.nusc.get('sample_data', lidar_data_token)
            cali_lidar = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            ego_to_lidar = transform_matrix(cali_lidar['translation'], Quaternion(cali_lidar['rotation']), inverse=False)

            ego_to_lidar_kitti = np.dot(ego_to_lidar, nu_to_kitti_lidar.transformation_matrix )
            ego_to_lidar_kitti_flat = ego_to_lidar_kitti[:3].reshape(-1)

            calib_f.write('P0: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            calib_f.write('P1: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            calib_f.write('P2: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            calib_f.write('P3: 0 0 0 0 0 0 0 0 0 0 0 0\n')
            calib_f.write('Tr: ' + ' '.join([str(x) for x in ego_to_lidar_kitti_flat]) + '\n')
            calib_f.close()

            ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            ### {'token', 'timestamp', 'rotation', 'translation'}
            ego_pose_kitti_first = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
                

            token_idx = 0  # Start tokens from 0.
            while True:
                sample = self.nusc.get('sample', sample_token)
                ### {'token', 'timestamp', 'prev', 'next', 'scene_token', 
                # 'data': {'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT':
                # 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP', 'CAM_FRONT', 
                # 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 
                # 'CAM_FRONT_LEFT'}, 'anns': []}

                lidar_data_token = sample['data'][self.lidar_name]

                ################ 1. Load panoptic annotation ################
                lidar_panoptic = self.nusc.get('lidarseg', lidar_data_token)
                ### {'token', 'sample_data_token', 'filename'}

                lidar_panoptic_anno = load_bin_file(os.path.join(self.nusc.dataroot, lidar_panoptic['filename']), type='lidarseg')
                semantic_anno = np.uint32(lidar_panoptic_anno)  # 不做除法，不提instance
                print("semantic_anno unique:", np.unique(semantic_anno))  # 加一句调试
                semantic_anno.tofile(os.path.join(label_folder, f'{token_idx:06}.label'))

                #cat_idx = lidar_panoptic_anno // 1000
                #print("cat_idx unique:", np.unique(cat_idx))

                #ins_idx = lidar_panoptic_anno % 1000
                # ### convert to categories in evaluation
                # coarse_idx = np.array([fine2coarse[c] for c in cat_idx])
                ### convert to uint32, with lower 16 as semantic label, upper 16 as instance id
                #panoptic_anno = (np.uint32(ins_idx) << 16) | np.uint32(cat_idx)
                #semantic_anno = np.uint32(cat_idx)

                # cat_idx = panoptic_anno & 0xFFFF  # semantic label in lower half
                # ins_idx = panoptic_anno >> 16
                ### save to label file
                #panoptic_anno.tofile(os.path.join(label_folder, f'{token_idx:06}.label'))
                #semantic_anno.tofile(os.path.join(label_folder, f'{token_idx:06}.label'))

                ################ 2. Load lidar point cloud ################
                lidar_data = self.nusc.get('sample_data', lidar_data_token)
                ### {'token', 'sampe_token', 'ego_pose_token', 'calibrated_sensor_token', 'timestamp', 
                # 'fileformat', 'is_key_frame', 'height', 'width', 'filename', 'prev', 'next', 'sensor_modality', 'channel'}
                
                lidar_pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_data['filename']))
                ### convert lidar point cloud to semantickitti frame
                lidar_pc.rotate(nu_to_kitti_lidar_inv.rotation_matrix)

                # fixed? old was lidar_pc.points.T.tofile(os.path.join(velo_folder, f'{token_idx:06}.bin'))
                lidar_pc.points[:4, :].T.astype(np.float32).tofile(os.path.join(velo_folder, f'{token_idx:06}.bin'))

                ################ 3. Load ego pose ################
                ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
                ### {'token', 'timestamp', 'rotation', 'translation'}
                ego_pose_kitti = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
                ego_pose_kitti_first_to_curr = np.dot(np.linalg.inv(ego_pose_kitti_first), ego_pose_kitti)
                ego_pose_kitti_flat = ego_pose_kitti_first_to_curr[:3].reshape(-1)
                pose_f.write(' '.join([str(x) for x in ego_pose_kitti_flat]) + '\n')

                ################ 4. Load timestamp ################
                ### convert microsecond to second
                time_second = lidar_data['timestamp']/1e6
                if token_idx == 0:
                    time_start = time_second
                ### write to times.txt in scientific notation
                times_f.write('{:.6e}\n'.format(time_second-time_start))

                ################ 5. Update sample token ################
                token_idx += 1
                if sample['next'] == '':
                    break
                else:
                    sample_token = sample['next']
                    
            pose_f.close()
            times_f.close()
            print(f'Finish processing scene {scene_idx} with {token_idx} samples.', flush=True)
            
        print('Finish processing all scenes.')
        return

### run the script
if __name__ == '__main__':
    fire.Fire(KittiConverter)