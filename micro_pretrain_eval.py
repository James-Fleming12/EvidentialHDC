import argparse
import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.HDC_utils import set_uq_model

NUM_CLASSES = 17

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def calculate_iou(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    iou[np.isnan(iou)] = 0.0
    return iou

def evaluate_headroom(model, clean_loader, corrupt_loader, device, num_frames=50):
    model.eval()
    
    clean_feats = []
    clean_lbls = []
    
    fog_feats = []
    fog_lbls = []
    
    print("  -> Extracting Clean Latents...")
    for i, batch in enumerate(tqdm(clean_loader, total=num_frames)):
        if i >= num_frames: break
        in_vol = batch[0].to(device)
        labels = batch[2].to(device).view(-1)
        mask = (batch[1].to(device) > 0).view(-1)
        with torch.no_grad():
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            clean_feats.append(z_flat.cpu())
            clean_lbls.append(labels[mask].cpu())
            
    print("  -> Extracting Fog Latents...")
    for i, batch in enumerate(tqdm(corrupt_loader, total=num_frames)):
        if i >= num_frames: break
        in_vol = batch[0].to(device)
        labels = batch[2].to(device).view(-1)
        mask = (batch[1].to(device) > 0).view(-1)
        with torch.no_grad():
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            fog_feats.append(z_flat.cpu())
            fog_lbls.append(labels[mask].cpu())
            
    clean_feats = torch.cat(clean_feats, dim=0)
    clean_lbls = torch.cat(clean_lbls, dim=0)
    fog_feats = torch.cat(fog_feats, dim=0)
    fog_lbls = torch.cat(fog_lbls, dim=0)
    
    # 1. Cosine Shift and Euclidean Shift
    print("  -> Calculating Shifts...")
    shifts = []
    euc_shifts = []
    for c in range(NUM_CLASSES):
        c_mask = clean_lbls == c
        f_mask = fog_lbls == c
        if c_mask.sum() > 0 and f_mask.sum() > 0:
            c_center_unnorm = clean_feats[c_mask].mean(dim=0).unsqueeze(0)
            f_center_unnorm = fog_feats[f_mask].mean(dim=0).unsqueeze(0)
            
            c_center = F.normalize(c_center_unnorm, p=2, dim=1)
            f_center = F.normalize(f_center_unnorm, p=2, dim=1)
            shift = 1.0 - torch.cosine_similarity(c_center, f_center).item()
            shifts.append(shift)
            
            euc_shift = torch.norm(c_center_unnorm - f_center_unnorm).item()
            euc_shifts.append(euc_shift)
    avg_cosine_shift = np.mean(shifts)
    avg_euc_shift = np.mean(euc_shifts)
    
    # 2. Neighborhood Purity (1-NN) in Fog space
    print("  -> Calculating 1-NN Purity...")
    # Subsample to avoid OOM
    if len(fog_feats) > 10000:
        idx = torch.randperm(len(fog_feats))[:10000]
        sub_feats = fog_feats[idx].to(device)
        sub_lbls = fog_lbls[idx].to(device)
    else:
        sub_feats = fog_feats.to(device)
        sub_lbls = fog_lbls.to(device)
        
    sub_feats_norm = F.normalize(sub_feats, p=2, dim=1)
    sim = torch.mm(sub_feats_norm, sub_feats_norm.T)
    sim.fill_diagonal_(-1)
    nearest_idx = sim.argmax(dim=1)
    purity = (sub_lbls[nearest_idx] == sub_lbls).float().mean().item()
    
    # 3. Magnitude Segregation
    # Just calculate average magnitude of clean vs corrupted
    print("  -> Calculating Magnitude Segregation...")
    clean_mag = torch.norm(clean_feats, p=2, dim=1).mean().item()
    fog_mag = torch.norm(fog_feats, p=2, dim=1).mean().item()
    
    return {
        "Cosine Shift": avg_cosine_shift,
        "Euclidean Shift": avg_euc_shift,
        "Target Neighborhood Purity": purity,
        "Average L2 Norm (Clean)": clean_mag,
        "Average L2 Norm (Fog)": fog_mag
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--out_dir", type=str, default="logs/micro_pretrain")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        DATA = yaml.safe_load(f)
    with open(args.arch, 'r') as f:
        ARCH = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    
    methods = ['baseline', 'supcon', 'vib', 'smoothness']
    results = {}
    
    # We need a validation parser for Fog-3
    fog_dir = os.path.join(args.kittic_dir, 'fog', 'heavy')
    if not os.path.exists(fog_dir):
        fog_dir = os.path.join(args.kittic_dir, 'fog', 'moderate')
        
    clean_parser = Parser(root=args.kitti_dir, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                          labels=DATA["labels"], color_map=DATA["color_map"], learning_map=DATA["learning_map"],
                          learning_map_inv=DATA["learning_map_inv"], sensor=ARCH["dataset"]["sensor"],
                          max_points=ARCH["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)
                          
    fog_parser = Parser(root=fog_dir, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                        labels=DATA["labels"], color_map=DATA["color_map"], learning_map=DATA["learning_map"],
                        learning_map_inv=DATA["learning_map_inv"], sensor=ARCH["dataset"]["sensor"],
                        max_points=ARCH["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)

    for method in methods:
        print(f"\n{'='*50}")
        print(f" Starting Micro-Pretrain: {method.upper()}")
        print(f"{'='*50}")
        
        log_dir = os.path.join(args.out_dir, method)
        os.makedirs(log_dir, exist_ok=True)
        
        # Instantiate GenTrainer
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, path=None, method=method)
        
        # Micro-training for 5 epochs
        # We manually truncate train_epoch in GenTrainer or just run normal train
        # Here we just call trainer.train(epochs=args.epochs)
        # Note: We will modify GenTrainer to only train 10% of the epoch to speed it up.
        trainer.train(epochs=args.epochs)
        
        # Extract features and calculate headroom metrics
        print(f"\n--- Evaluating Headroom for {method.upper()} ---")
        metrics = evaluate_headroom(trainer.model, clean_parser.validloader, fog_parser.validloader, device, num_frames=50)
        
        results[method] = metrics
        
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    out_path = os.path.join(args.out_dir, "micro_pretrain_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved Micro-Pretrain Results to {out_path}")

if __name__ == "__main__":
    main()
