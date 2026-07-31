import torch
import torch.nn.functional as F
import os
import argparse
import sys
sys.path.append('.')

from dataset.kitti.parser import Parser
from torch.utils.data import DataLoader
load_hdc_model = __import__("unsup_kitti-c").load_hdc_model
import yaml

args = argparse.Namespace(pretrained_path='logs/kitti_pretrain/hdc_sub.pth', kitti_c_dir='/mnt/alpha/jmfleming/KITTI-C', tau=0.0)
with open('config/arch/senet-2048p.yml', 'r') as f:
    ARCH = yaml.safe_load(f)
with open('config/labels/semantic-kitti-all.yaml', 'r') as f:
    DATA = yaml.safe_load(f)

model = load_hdc_model(args.pretrained_path, num_classes=17)
model = model.cuda()
model.eval()

target_dataset = Parser(
    root=os.path.join(args.kitti_c_dir, 'fog', '3'),
    train_sequences=DATA["split"]["valid"],
    valid_sequences=DATA["split"]["valid"],
    test_sequences=None,
    labels=DATA["labels"],
    color_map=DATA.get("color_map", {}),
    learning_map=DATA["learning_map"],
    learning_map_inv=DATA["learning_map_inv"],
    sensor=ARCH["dataset"]["sensor"],
    max_volume_space=ARCH["dataset"]["max_volume_space"],
    min_volume_space=ARCH["dataset"]["min_volume_space"],
    dynamic_geom=False
)
loader = target_dataset.get_valid_set()
proj_in, proj_labels, _, _ = next(iter(loader))
proj_in = proj_in.cuda()

with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=True):
        latent_x = model.net(proj_in, only_feat=True)
    raw_enc, _, _ = model.encode(proj_in)
    norm_enc = F.normalize(raw_enc, dim=1).cuda().to(model.classify.weight.dtype)
    
    fallback_logits = model.classify(norm_enc)
    
    # Try different temperatures
    for temp in [1.0, 0.5, 0.1, 0.05, 0.01]:
        conf = F.softmax(fallback_logits / temp, dim=1).max(dim=1)[0]
        above_09 = (conf >= 0.9).sum().item()
        print(f"Temp: {temp} -> Max Conf: {conf.max().item():.4f}, Mean Conf: {conf.mean().item():.4f}, Points >= 0.9: {above_09} out of {conf.shape[0]}")
