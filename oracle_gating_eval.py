import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from common.parser import Parser
from modules.trainer import Trainer
from modules.gen_trainers import GenTrainer

def evaluate_oracle_gating(clean_feats, clean_lbls, fog_feats, fog_lbls, device='cuda'):
    print("\n" + "="*60)
    print("--- Phase 9: Oracle Gating Validation ---")
    print("="*60)
    
    # 1. Project to HDC Space (D=1000)
    HD_DIM = 1000
    torch.manual_seed(42)
    proj = (torch.rand(clean_feats.shape[1], HD_DIM) > 0.5).float() * 2 - 1
    proj = proj.to(device)
    
    print("-> Projecting to HDC Space...")
    h_clean = torch.matmul(clean_feats.to(device), proj)
    h_fog = torch.matmul(fog_feats.to(device), proj)
    
    c_lbl = clean_lbls.to(device)
    f_lbl = fog_lbls.to(device)
    
    # Binarize (Sign)
    h_clean = torch.sign(h_clean)
    h_fog = torch.sign(h_fog)
    
    # Train Linear Probe on 128D (to use as our confidence oracle)
    print("-> Training Linear Probe Oracle (128D)...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(clean_feats[:20000].numpy(), clean_lbls[:20000].numpy())
    
    # Get pseudo-labels and confidence for Fog points
    print("-> Generating Oracle metrics for candidate pool...")
    fog_probs = clf.predict_proba(fog_feats.numpy())
    fog_pseudo_lbls = torch.tensor(fog_probs.argmax(axis=1)).to(device)
    fog_confidences = torch.tensor(fog_probs.max(axis=1)).to(device)
    
    # 2. Establish Base Prototypes (Zero-Shot)
    print("-> Establishing Zero-Shot Prototypes...")
    protos = []
    proto_lbls = []
    NUM_CLASSES = 17
    for c in range(NUM_CLASSES):
        mask = c_lbl == c
        if mask.sum() > 0:
            protos.append(h_clean[mask].mean(dim=0))
            proto_lbls.append(c)
    
    base_protos = F.normalize(torch.stack(protos), p=2, dim=1)
    proto_lbls = torch.tensor(proto_lbls).to(device)
    
    # Base accuracy on validation set (last 20k points of fog)
    N_total = len(h_fog)
    val_idx = N_total - 20000
    val_feats = h_fog[val_idx:]
    val_labels = f_lbl[val_idx:]
    
    sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), base_protos.T)
    base_preds = proto_lbls[sims.argmax(dim=1)]
    zero_shot_correct = (base_preds == val_labels).sum().item()
    zero_shot_acc = zero_shot_correct / len(val_labels)
    print(f"   Zero-Shot Accuracy: {zero_shot_acc:.4f} ({zero_shot_correct}/{len(val_labels)})")
    
    # Candidate pool for adaptation (first 10k points)
    pool_feats = h_fog[:10000]
    pool_pseudo = fog_pseudo_lbls[:10000]
    pool_conf = fog_confidences[:10000]
    pool_labels = f_lbl[:10000]
    
    # Calculate base prototype distances for the pool
    base_pool_sims = torch.matmul(F.normalize(pool_feats, p=2, dim=1), base_protos.T)
    base_pool_max_sims = base_pool_sims.max(dim=1)[0]
    
    # DIAGNOSTIC 1: ORACLE GATING
    print("\n--- Diagnostic 1: Oracle Gating ---")
    def run_adaptation(mask, name):
        adapted_protos = base_protos.clone()
        alpha = 0.01  # Smaller alpha so 10k updates don't completely overwrite
        updates_applied = 0
        for i in range(len(pool_feats)):
            if mask[i]:
                pl = pool_pseudo[i]
                idx = (proto_lbls == pl).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    idx = idx[0]
                    adapted_protos[idx] = adapted_protos[idx] * (1 - alpha) + pool_feats[i] * alpha
                    adapted_protos[idx] = F.normalize(adapted_protos[idx], p=2, dim=0)
                    updates_applied += 1
                    
        sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), adapted_protos.T)
        preds = proto_lbls[sims.argmax(dim=1)]
        acc = (preds == val_labels).float().mean().item()
        
        drift = torch.norm(adapted_protos - base_protos, dim=1).mean().item()
        print(f"[{name}] Acc: {acc:.4f} | Drift: {drift:.4f} | Updates: {updates_applied}")
        return acc, drift

    # Tests
    mask_100 = torch.ones(len(pool_feats), dtype=torch.bool).to(device)
    run_adaptation(mask_100, "100% Updates (No Gate)")
    
    thresh_90 = torch.quantile(pool_conf, 0.10)
    run_adaptation(pool_conf >= thresh_90, "Top 90% Confidence")
    
    thresh_75 = torch.quantile(pool_conf, 0.25)
    run_adaptation(pool_conf >= thresh_75, "Top 75% Confidence")
    
    thresh_50 = torch.quantile(pool_conf, 0.50)
    run_adaptation(pool_conf >= thresh_50, "Top 50% Confidence")
    
    thresh_dist_50 = torch.quantile(base_pool_max_sims, 0.50) 
    run_adaptation(base_pool_max_sims >= thresh_dist_50, "Top 50% Proto Sim")
    
    mask_perfect = pool_pseudo == pool_labels
    run_adaptation(mask_perfect, "Perfect Oracle (True Lbl)")
    
    # DIAGNOSTIC 2: LEAVE-ONE-UPDATE-OUT
    print("\n--- Diagnostic 2: Leave-One-Update-Out ---")
    helpful, neutral, harmful = [], [], []
    alpha = 0.05 # For a single point update, 5% is good to force a measurable boundary shift
    
    # Limit to 2000 points to keep runtime manageable
    eval_pool_size = min(2000, len(pool_feats))
    
    for i in tqdm(range(eval_pool_size), desc="Evaluating Single Updates"):
        pl = pool_pseudo[i]
        idx = (proto_lbls == pl).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        
        new_protos = base_protos.clone()
        new_protos[idx] = new_protos[idx] * (1 - alpha) + pool_feats[i] * alpha
        new_protos[idx] = F.normalize(new_protos[idx], p=2, dim=0)
        
        sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), new_protos.T)
        preds = proto_lbls[sims.argmax(dim=1)]
        new_correct = (preds == val_labels).sum().item()
        
        delta = new_correct - zero_shot_correct
        
        meta = {
            'conf': pool_conf[i].item(),
            'sim': base_pool_max_sims[i].item(),
            'norm': torch.norm(fog_feats[i]).item(),
            'delta': delta
        }
        
        if delta > 0:
            helpful.append(meta)
        elif delta < 0:
            harmful.append(meta)
        else:
            neutral.append(meta)
            
    print(f"\nFound {len(helpful)} Helpful, {len(neutral)} Neutral, {len(harmful)} Harmful updates.")
    
    if len(helpful) > 0 and len(harmful) > 0:
        print("\nMetric      | Helpful Mean | Harmful Mean")
        print("-" * 40)
        h_conf = np.mean([m['conf'] for m in helpful])
        hm_conf = np.mean([m['conf'] for m in harmful])
        print(f"Probe Conf  | {h_conf:.4f}       | {hm_conf:.4f}")
        
        h_sim = np.mean([m['sim'] for m in helpful])
        hm_sim = np.mean([m['sim'] for m in harmful])
        print(f"Proto Sim   | {h_sim:.4f}       | {hm_sim:.4f}")
        
        h_norm = np.mean([m['norm'] for m in helpful])
        hm_norm = np.mean([m['norm'] for m in harmful])
        print(f"Feat Norm   | {h_norm:.4f}       | {hm_norm:.4f}")

def main():
    parser = argparse.ArgumentParser("./oracle_gating_eval.py")
    parser.add_argument('--kitti_dir', '-d', type=str, default='/home/james/Research/SEE/dataset/', help='Dataset path')
    args, _ = parser.parse_known_args()
    
    cfg_file = "config/kitti_gen.yaml"
    DATA = yaml.safe_load(open(cfg_file, 'r'))
    ARCH = yaml.safe_load(open("config/SENet_VIB.yaml", 'r'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    # We will just load the SupCon+VIB model
    method = 'supcon_vib'
    load_path = f"logs/med_pretrain_{method}"
    
    if not os.path.exists(load_path):
        print(f"Error: {load_path} not found.")
        return
        
    print(f"\nLoading Model: {method}")
    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, load_path, path=load_path, method=method)
    model = trainer.model
    model.eval()
    
    clean_loader = trainer.parser.get_train_set()
    
    clean_feats, clean_lbls = [], []
    fog_feats, fog_lbls = [], []
    
    # We'll extract 10 batches of clean and fog (for ~60,000 points per domain)
    NUM_BATCHES = 10
    
    print("-> Extracting Clean Latents...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(clean_loader, total=NUM_BATCHES)):
            if i >= NUM_BATCHES: break
            in_vol, _, labels, _ = batch
            in_vol = in_vol.to(device)
            mask = labels > 0 
            
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            clean_feats.append(z_flat.cpu())
            clean_lbls.append(labels[mask].cpu())
            
    print("-> Extracting Fog Latents...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(clean_loader, total=NUM_BATCHES)):
            if i >= NUM_BATCHES: break
            in_vol, _, labels, _ = batch
            in_vol = in_vol.to(device)
            # Add fog augmentation
            in_vol = trainer.get_augmented_view(in_vol)
            
            mask = labels > 0 
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
    
    evaluate_oracle_gating(clean_feats, clean_lbls, fog_feats, fog_lbls, device)

if __name__ == '__main__':
    main()
