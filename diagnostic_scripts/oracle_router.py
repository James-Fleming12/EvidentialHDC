import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm

from dataset.kitti.parser import Parser
from modules.HDC_utils import set_uq_model

NUM_CLASSES = 17
CORRUPTIONS = [
    'fog', 'snow', 'wet_ground', 'motion_blur',
    'beam_missing', 'crosstalk', 'incomplete_echo', 'cross_sensor'
]

def get_parser(root, data_cfg, arch_cfg):
    return Parser(root=root,
                  train_sequences=['08'],
                  valid_sequences=['08'],
                  test_sequences=['08'],
                  labels=data_cfg["labels"],
                  color_map=data_cfg["color_map"],
                  learning_map=data_cfg["learning_map"],
                  learning_map_inv=data_cfg["learning_map_inv"],
                  sensor=arch_cfg["dataset"]["sensor"],
                  max_points=arch_cfg["dataset"]["max_points"],
                  batch_size=1,
                  workers=4,
                  gt=True,
                  shuffle_train=False)

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def calculate_iou(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    iou[np.isnan(iou)] = 0.0
    return iou

def load_hdc_model(path, num_classes=NUM_CLASSES, arch_cfg=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modeldir = os.path.dirname(path)
    model = set_uq_model(arch_cfg, modeldir, 'rp', 0, 0, num_classes, device)
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

def get_corruption_type(ctype):
    if ctype in ['incomplete_echo']:
        return 'Type A'
    elif ctype in ['snow', 'wet_ground', 'motion_blur', 'beam_missing']:
        return 'Type B'
    else:
        return 'Type C'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--pretrained_path", type=str, default="logs/kitti_pretrain/hdc_sub.pth")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--out_dir", type=str, default="logs/atlas")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        DATA = yaml.safe_load(f)
    with open(args.arch, 'r') as f:
        ARCH = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    
    print("Loading HDC model and Backbone...")
    base_model = load_hdc_model(args.pretrained_path, arch_cfg=ARCH)
    # Save clean prototype weights to reset between corruptions
    clean_weights = base_model.classify.weight.data.clone()
    
    results = {}
    
    for ctype in CORRUPTIONS:
        print(f"\n--- Processing {ctype} ({get_corruption_type(ctype)}) ---")
        
        corrupt_root = os.path.join(args.kittic_dir, ctype, 'heavy')
        if not os.path.exists(corrupt_root):
            corrupt_root = os.path.join(args.kittic_dir, ctype, 'moderate')
            if not os.path.exists(corrupt_root):
                print(f"Warning: {corrupt_root} missing, skipping {ctype}")
                continue
                
        c_parser = get_parser(corrupt_root, DATA, ARCH)
        c_loader = c_parser.validloader
        
        # Reset model weights for independent evaluation
        base_model.classify.weight.data = clean_weights.clone()
        
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        max_frames = 50
        
        # EMA alpha for Type B
        alpha = 0.05
        
        for i, batch in enumerate(tqdm(c_loader, total=max_frames)):
            if i >= max_frames: break
            
            proj_in = batch[0].to(device)
            proj_labels = batch[2].to(device)
            mask = (batch[1].to(device) > 0).view(-1)
            
            with torch.no_grad():
                feat, _, _ = base_model.encode(proj_in)
                feat = feat[mask].to(base_model.classify.weight.dtype)
                labels = proj_labels.view(-1)[mask]
                
                # Forward pass
                logits = base_model.classify(feat)
                preds = logits.argmax(dim=1)
                
                # Router Logic
                ctype_class = get_corruption_type(ctype)
                
                if ctype_class == 'Type A':
                    # Temperature Scaling (Simulated by just using raw frozen predictions)
                    # We do not update prototypes because geometry is perfect.
                    pass
                    
                elif ctype_class == 'Type B':
                    # Prototype EMA Adaptation
                    probs = F.softmax(logits, dim=1)
                    conf, pseudo_labels = probs.max(dim=1)
                    
                    # High confidence threshold for pseudo-labeling
                    hc_mask = conf > 0.90
                    
                    if hc_mask.sum() > 0:
                        feat_norm = F.normalize(feat, p=2, dim=1)
                        for c in range(NUM_CLASSES):
                            c_mask = hc_mask & (pseudo_labels == c)
                            if c_mask.sum() > 5:
                                # Update prototype using EMA
                                new_center = feat_norm[c_mask].mean(dim=0)
                                current_proto = base_model.classify.weight.data[c]
                                updated_proto = (1 - alpha) * current_proto + alpha * new_center
                                base_model.classify.weight.data[c] = F.normalize(updated_proto, p=2, dim=0)
                                
                elif ctype_class == 'Type C':
                    # Frozen Generative Gate (Simulated by falling back to frozen state)
                    # Geometry is collapsed, so updating would cause catastrophic drift.
                    pass
                
                hist += fast_hist(preds.cpu().numpy(), labels.cpu().numpy(), NUM_CLASSES)
                
        miou = np.nanmean(calculate_iou(hist))
        results[ctype] = float(miou)
        print(f"Result for {ctype}: {miou:.4f}")
        
    out_path = os.path.join(args.out_dir, "oracle_router_results.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved Oracle Router Results to {out_path}")

if __name__ == "__main__":
    main()
