import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_io import load_bin_file

# 设置你的 nuScenes 数据根目录
NUSC_DIR = "/mnt/data/nuscenes"
NUSC_VERSION = "v1.0-trainval"

def check_lidarseg_label():
    print("🚀 正在加载 nuScenes...")
    nusc = NuScenes(version=NUSC_VERSION, dataroot=NUSC_DIR)

    for i, sample in enumerate(nusc.sample[:5]):  # 检查前5个 sample
        lidar_token = sample['data']['LIDAR_TOP']

        # 检查是否有 lidarseg 标注
        lidarseg_record = nusc.get('lidarseg', lidar_token)
        if not lidarseg_record:
            print(f"第 {i} 个 sample 没有 lidarseg 标签")
            continue

        label_rel_path = lidarseg_record['filename']
        label_abs_path = os.path.join(NUSC_DIR, label_rel_path)

        if not os.path.exists(label_abs_path):
            print(f"标签文件不存在: {label_abs_path}")
            continue

        # 读取标签并打印唯一值
        label = load_bin_file(label_abs_path, type='lidarseg')
        unique_vals = np.unique(label)
        print(f"Sample {i} 标签唯一值: {unique_vals}")

        if len(unique_vals) <= 1:
            print("⚠️ 标签文件存在，但只有一个类别，可能无效")
        else:
            print("标签文件包含多个类别，正常")
        break  # 检查一个成功即可

if __name__ == '__main__':
    check_lidarseg_label()
