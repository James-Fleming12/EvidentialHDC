import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import numpy as np
import json
from tqdm import tqdm

from dataset.kitti.parser import Parser
from modules.HDC_utils import set_uq_model

NUM_CLASSES = 17
SEVERITY_MAP = {1: 'light', 2: 'moderate', 3: 'heavy'}
CORRUPTIONS = [
    'fog', 'snow', 'wet_ground', 'motion_blur',
    'beam_missing', 'crosstalk', 'incomplete_echo', 'cross_sensor'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Corruption Robustness Atlas")
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI", help="Clean KITTI data")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C", help="KITTI-C data")
    parser.add_argument("--pretrained_path", type=str, default="logs/kitti_pretrain/hdc_sub.pth", help="HDC model")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--out_dir", type=str, default="logs/atlas", help="Output dir")
    return parser.parse_args()

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

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    with open(args.config, 'r') as f:
        DATA = yaml.safe_load(f)
    with open(args.arch, 'r') as f:
        ARCH = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    
    print("Loading HDC model and Backbone...")
    model = load_hdc_model(args.pretrained_path, arch_cfg=ARCH)
    
    print("Loading clean dataset...")
    clean_parser = get_parser(args.kitti_dir, DATA, ARCH)
    clean_loader = clean_parser.validloader
    
    print("Loading corrupted datasets...")
    corrupt_loaders = {}
    for ctype in CORRUPTIONS:
        corrupt_root = os.path.join(args.kittic_dir, ctype, 'heavy')
        if not os.path.exists(corrupt_root):
            corrupt_root = os.path.join(args.kittic_dir, ctype, 'moderate')
            if not os.path.exists(corrupt_root):
                print(f"Warning: {corrupt_root} missing, skipping {ctype}")
                continue
        c_parser = get_parser(corrupt_root, DATA, ARCH)
        corrupt_loaders[ctype] = c_parser.validloader
        
    atlas = {}
    
    for ctype, c_loader in corrupt_loaders.items():
        print(f"\n--- Processing {ctype} ---")
        
        # Accumulators for Pass 1 (Prototypes & Metrics)
        c_proto_sum = torch.zeros((NUM_CLASSES, model.classify.weight.shape[1]), dtype=torch.float32, device=device)
        c_proto_count = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)
        
        x_proto_sum = torch.zeros((NUM_CLASSES, model.classify.weight.shape[1]), dtype=torch.float32, device=device)
        x_proto_count = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)
        
        cosine_shifts = []
        euclidean_shifts = []
        
        clean_purity_list = []
        corrupt_purity_list = []
        
        hist_baseline = np.zeros((NUM_CLASSES, NUM_CLASSES))
        hist_oracle_input = np.zeros((NUM_CLASSES, NUM_CLASSES))
        
        max_frames = 50
        
        # -----------------------------------------
        # PASS 1: Build Prototypes & compute inline metrics
        # -----------------------------------------
        print("  -> Pass 1: Extracting global prototypes & inline metrics")
        for i, (clean_batch, corrupt_batch) in enumerate(tqdm(zip(clean_loader, c_loader), total=max_frames)):
            if i >= max_frames: break
            
            clean_in = clean_batch[0].to(device)
            clean_proj = clean_batch[1].to(device)
            clean_labels = clean_batch[2].to(device)
            corr_in = corrupt_batch[0].to(device)
            clean_mask = (clean_proj > 0).view(-1)
            
            with torch.no_grad():
                c_feat, _, _ = model.encode(clean_in)
                c_feat = c_feat[clean_mask].to(model.classify.weight.dtype)
                
                c_logits = model.classify(c_feat)
                c_preds = torch.argmax(c_logits, dim=1).cpu().numpy()
                
                x_feat, _, _ = model.encode(corr_in)
                x_feat = x_feat[clean_mask].to(model.classify.weight.dtype)
                
                x_logits = model.classify(x_feat)
                x_preds = torch.argmax(x_logits, dim=1).cpu().numpy()
                
                valid_labels = clean_labels.view(-1)[clean_mask]
                
                # Oracle Input (Perfect Diffusion)
                hist_baseline += fast_hist(x_preds, valid_labels.cpu().numpy(), NUM_CLASSES)
                hist_oracle_input += fast_hist(c_preds, valid_labels.cpu().numpy(), NUM_CLASSES)
                
                # Representation Sensitivity
                c_feat_norm = torch.nn.functional.normalize(c_feat, p=2, dim=1)
                x_feat_norm = torch.nn.functional.normalize(x_feat, p=2, dim=1)
                
                cos_shift = (1.0 - torch.cosine_similarity(c_feat_norm, x_feat_norm, dim=1)).mean().item()
                euc_shift = torch.norm(c_feat_norm - x_feat_norm, p=2, dim=1).mean().item()
                cosine_shifts.append(cos_shift)
                euclidean_shifts.append(euc_shift)
                
                # Accumulate prototypes
                for cls_idx in range(NUM_CLASSES):
                    mask = (valid_labels == cls_idx)
                    if mask.sum() > 0:
                        c_proto_sum[cls_idx] += c_feat_norm[mask].sum(dim=0).to(torch.float32)
                        c_proto_count[cls_idx] += mask.sum().float()
                        x_proto_sum[cls_idx] += x_feat_norm[mask].sum(dim=0).to(torch.float32)
                        x_proto_count[cls_idx] += mask.sum().float()
                        
                # Oracle Memory (Neighborhood Purity via intra-frame 1-NN)
                # Downsample to max 5000 points to avoid OOM in pairwise distance
                if c_feat_norm.shape[0] > 5000:
                    idx = torch.randperm(c_feat_norm.shape[0])[:5000]
                    c_sub = c_feat_norm[idx]
                    x_sub = x_feat_norm[idx]
                    y_sub = valid_labels[idx]
                else:
                    c_sub = c_feat_norm
                    x_sub = x_feat_norm
                    y_sub = valid_labels
                    
                c_sim = torch.mm(c_sub, c_sub.T)
                c_sim.fill_diagonal_(-1)
                c_nn = c_sim.argmax(dim=1)
                clean_purity_list.append((y_sub[c_nn] == y_sub).float().mean().item())
                
                x_sim = torch.mm(x_sub, x_sub.T)
                x_sim.fill_diagonal_(-1)
                x_nn = x_sim.argmax(dim=1)
                corrupt_purity_list.append((y_sub[x_nn] == y_sub).float().mean().item())

        # Normalize prototypes
        c_protos = torch.nn.functional.normalize(c_proto_sum / c_proto_count.unsqueeze(1).clamp(min=1), p=2, dim=1).to(model.classify.weight.dtype)
        x_protos = torch.nn.functional.normalize(x_proto_sum / x_proto_count.unsqueeze(1).clamp(min=1), p=2, dim=1).to(model.classify.weight.dtype)
        
        # Prototype Drift (Global angle between true classes)
        valid_proto_mask = (c_proto_count > 0) & (x_proto_count > 0)
        proto_drifts = 1.0 - torch.cosine_similarity(c_protos[valid_proto_mask], x_protos[valid_proto_mask], dim=1)
        mean_proto_drift = proto_drifts.mean().item()
        
        # -----------------------------------------
        # PASS 2: Oracle Prototype evaluation
        # -----------------------------------------
        print("  -> Pass 2: Oracle Prototype evaluation")
        hist_oracle_proto = np.zeros((NUM_CLASSES, NUM_CLASSES))
        
        for i, (clean_batch, corrupt_batch) in enumerate(tqdm(zip(clean_loader, c_loader), total=max_frames)):
            if i >= max_frames: break
            
            clean_in = clean_batch[0].to(device)
            clean_proj = clean_batch[1].to(device)
            clean_labels = clean_batch[2].to(device)
            corr_in = corrupt_batch[0].to(device)
            clean_mask = (clean_proj > 0).view(-1)
            
            with torch.no_grad():
                x_feat, _, _ = model.encode(corr_in)
                x_feat = x_feat[clean_mask].to(model.classify.weight.dtype)
                x_feat_norm = torch.nn.functional.normalize(x_feat, p=2, dim=1)
                valid_labels = clean_labels.view(-1)[clean_mask].cpu().numpy()
                
                # Classify using Oracle Corrupted Prototypes
                # Only consider valid prototypes
                logits = torch.mm(x_feat_norm, x_protos.T)
                # Mask out unseen prototypes to -inf
                logits[:, ~valid_proto_mask] = -1e9
                preds = logits.argmax(dim=1).cpu().numpy()
                
                hist_oracle_proto += fast_hist(preds, valid_labels, NUM_CLASSES)
                
        baseline_miou = np.nanmean(calculate_iou(hist_baseline))
        oracle_input_miou = np.nanmean(calculate_iou(hist_oracle_input))
        oracle_proto_miou = np.nanmean(calculate_iou(hist_oracle_proto))
        
        atlas[ctype] = {
            "Representation Sensitivity": {
                "Average Cosine Shift": float(np.mean(cosine_shifts)),
                "Average Euclidean Shift": float(np.mean(euclidean_shifts)),
                "Global Prototype Drift": float(mean_proto_drift)
            },
            "Oracle Family (mIoU)": {
                "Baseline (Corrupted)": float(baseline_miou),
                "Oracle Input (Perfect Diffusion)": float(oracle_input_miou),
                "Oracle Prototype (Perfect Graph)": float(oracle_proto_miou)
            },
            "Neighborhood Purity": {
                "Clean 1-NN Purity": float(np.mean(clean_purity_list)),
                "Corrupted 1-NN Purity": float(np.mean(corrupt_purity_list))
            }
        }
        
    out_path = os.path.join(args.out_dir, "atlas.json")
    with open(out_path, 'w') as f:
        json.dump(atlas, f, indent=4)
    print(f"\nSaved Corruption Atlas to {out_path}")

if __name__ == "__main__":
    main()
