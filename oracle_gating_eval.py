import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer

CORRUPTIONS = [
    'fog', 'snow', 'wet_ground', 'incomplete_echo', 
    'crosstalk', 'beam_missing', 'motion_blur', 'cross_sensor'
]

def get_hdc_projection(dim_in=128, dim_out=10000, device='cuda'):
    torch.manual_seed(42)
    proj = (torch.rand(dim_in, dim_out) > 0.5).float() * 2 - 1
    return proj.to(device)

def build_hdc_prototypes(feats_128, lbls, proj, num_classes=17, device='cuda', chunk_size=50000):
    protos = torch.zeros(num_classes, proj.shape[1], device=device)
    counts = torch.zeros(num_classes, device=device)
    
    for i in range(0, len(feats_128), chunk_size):
        chunk_f = feats_128[i:i+chunk_size].to(device)
        chunk_l = lbls[i:i+chunk_size].to(device)
        
        h_chunk = torch.sign(torch.matmul(chunk_f, proj))
        
        for c in range(num_classes):
            mask = chunk_l == c
            if mask.sum() > 0:
                protos[c] += h_chunk[mask].sum(dim=0)
                counts[c] += mask.sum()
                
    for c in range(num_classes):
        if counts[c] > 0:
            protos[c] /= counts[c]
            
    base_protos = F.normalize(protos, p=2, dim=1)
    proto_lbls = torch.arange(num_classes, device=device)
    
    # Filter out empty classes
    valid_mask = counts > 0
    return base_protos[valid_mask], proto_lbls[valid_mask]

def evaluate_oracle_gating(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, clf, proj, device='cuda'):
    c_lbl = corrupt_lbls.to(device)
    
    # Get pseudo-labels and confidence for Fog points (in 128D)
    print("      -> Running Probe Inference (128D)...")
    corrupt_probs = clf.predict_proba(corrupt_feats.numpy())
    corrupt_pseudo_lbls = torch.tensor(corrupt_probs.argmax(axis=1)).to(device)
    corrupt_confidences = torch.tensor(corrupt_probs.max(axis=1)).to(device)
    
    # Extract sets to avoid OOM
    # Pool = 20k points for adaptation tests. Val = 100k points for evaluation
    pool_size = 20000
    val_size = 100000
    
    pool_f_128 = corrupt_feats[:pool_size].to(device)
    pool_lbls = c_lbl[:pool_size]
    pool_pseudo = corrupt_pseudo_lbls[:pool_size]
    pool_conf = corrupt_confidences[:pool_size]
    
    val_f_128 = corrupt_feats[-val_size:].to(device)
    val_lbls = c_lbl[-val_size:]
    
    print("      -> Projecting Validation Set to 10kD HDC...")
    val_feats = torch.sign(torch.matmul(val_f_128, proj))
    
    print("      -> Projecting Adaptation Pool to 10kD HDC...")
    pool_feats = torch.sign(torch.matmul(pool_f_128, proj))
    
    # Base accuracy on validation set
    sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), base_protos.T)
    base_preds = proto_lbls[sims.argmax(dim=1)]
    zero_shot_correct = (base_preds == val_lbls).sum().item()
    zero_shot_acc = zero_shot_correct / len(val_lbls)
    
    # Perfect Oracle test
    print("      -> Running Perfect Oracle Test...")
    mask_perfect = pool_pseudo == pool_lbls
    adapted_protos = base_protos.clone()
    alpha = 0.01
    for i in range(len(pool_feats)):
        if mask_perfect[i]:
            pl = pool_pseudo[i]
            idx = (proto_lbls == pl).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                idx = idx[0]
                adapted_protos[idx] = adapted_protos[idx] * (1 - alpha) + pool_feats[i] * alpha
                adapted_protos[idx] = F.normalize(adapted_protos[idx], p=2, dim=0)
                
    sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), adapted_protos.T)
    preds = proto_lbls[sims.argmax(dim=1)]
    perfect_acc = (preds == val_lbls).float().mean().item()
    
    # Leave-One-Update-Out
    print("      -> Running Leave-One-Update-Out Test...")
    helpful, neutral, harmful = [], [], []
    alpha_single = 0.05 
    eval_pool_size = min(5000, len(pool_feats)) # 5000 updates tested
    
    for i in tqdm(range(eval_pool_size), desc="         Updates", leave=False):
        pl = pool_pseudo[i]
        idx = (proto_lbls == pl).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        
        new_protos = base_protos.clone()
        new_protos[idx] = new_protos[idx] * (1 - alpha_single) + pool_feats[i] * alpha_single
        new_protos[idx] = F.normalize(new_protos[idx], p=2, dim=0)
        
        sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), new_protos.T)
        preds = proto_lbls[sims.argmax(dim=1)]
        new_correct = (preds == val_lbls).sum().item()
        
        delta = new_correct - zero_shot_correct
        
        meta = {
            'conf': pool_conf[i].item(),
            'norm': torch.norm(pool_f_128[i]).item(),
            'delta': delta
        }
        
        if delta > 0:
            helpful.append(meta)
        elif delta < 0:
            harmful.append(meta)
        else:
            neutral.append(meta)
            
    res = {
        'zero_shot': zero_shot_acc,
        'perfect_acc': perfect_acc,
        'h_conf': np.mean([m['conf'] for m in helpful]) if helpful else 0.0,
        'hm_conf': np.mean([m['conf'] for m in harmful]) if harmful else 0.0,
        'h_norm': np.mean([m['norm'] for m in helpful]) if helpful else 0.0,
        'hm_norm': np.mean([m['norm'] for m in harmful]) if harmful else 0.0,
        'h_count': len(helpful),
        'hm_count': len(harmful)
    }
    return res

def main():
    parser = argparse.ArgumentParser("./oracle_gating_eval.py")
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    args, _ = parser.parse_known_args()
    
    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    method = 'supcon_vib'
    load_path = f"logs/med_pretrain_{method}"
    
    if not os.path.exists(load_path):
        print(f"Error: {load_path} not found.")
        return
        
    print(f"\nLoading Model: {method}")
    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, load_path, path=load_path, method=method)
    model = trainer.model
    model.eval()
    
    clean_parser = Parser(root=args.kitti_dir, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                          labels=DATA["labels"], color_map=DATA["color_map"], learning_map=DATA["learning_map"],
                          learning_map_inv=DATA["learning_map_inv"], sensor=ARCH["dataset"]["sensor"],
                          max_points=ARCH["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)
                          
    clean_loader = clean_parser.get_train_set()
    
    clean_feats, clean_lbls = [], []
    NUM_BATCHES = 100
    
    print("-> Extracting Clean Latents (100 Frames)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(clean_loader, total=NUM_BATCHES)):
            if i >= NUM_BATCHES: break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            clean_feats.append(z_flat.cpu())
            clean_lbls.append(labels[mask].cpu())
            
    clean_feats = torch.cat(clean_feats, dim=0)
    clean_lbls = torch.cat(clean_lbls, dim=0)
    print(f"   [Total Clean Points Extracted: {len(clean_feats)}]")
    
    # Train Linear Probe on 128D (to use as our confidence oracle)
    print("-> Training Linear Probe Oracle (128D on 100k points)...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    train_size = min(100000, len(clean_feats))
    clf.fit(clean_feats[:train_size].numpy(), clean_lbls[:train_size].numpy())
    
    probe_clean_acc = clf.score(clean_feats[:train_size].numpy(), clean_lbls[:train_size].numpy())
    print(f"   [Base] Linear Probe Accuracy (Clean): {probe_clean_acc:.4f}\n")
    
    # Build robust 10kD HDC base prototypes over all clean points
    print("-> Building 10kD HDC Clean Base Prototypes...")
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    base_protos, proto_lbls = build_hdc_prototypes(clean_feats, clean_lbls, proj, device=device)
    
    # We no longer need the massive clean_feats tensor
    del clean_feats
    del clean_lbls
    
    all_results = {}
    
    for corruption in CORRUPTIONS:
        print(f"\n{'='*60}")
        print(f"Evaluating Corruption: {corruption.upper()}")
        print(f"{'='*60}")
        
        fog_dir = os.path.join(args.kittic_dir, corruption, 'heavy')
        if not os.path.exists(fog_dir):
            fog_dir = os.path.join(args.kittic_dir, corruption, 'moderate')
            
        corrupt_parser = Parser(root=fog_dir, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                            labels=DATA["labels"], color_map=DATA["color_map"], learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"], sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)
        
        corrupt_loader = corrupt_parser.get_train_set()
        
        corrupt_feats, corrupt_lbls = [], []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(corrupt_loader, total=NUM_BATCHES, desc=f"   Ext. {corruption}")):
                if i >= NUM_BATCHES: break
                in_vol = batch[0].to(device)
                labels = batch[2].to(device).view(-1)
                mask = (batch[1].to(device) > 0).view(-1)
                
                out_tuple = model(in_vol)
                if len(out_tuple) == 3:
                    _, _, z8 = out_tuple
                else:
                    _, z8 = out_tuple
                z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
                corrupt_feats.append(z_flat.cpu())
                corrupt_lbls.append(labels[mask].cpu())
                
        corrupt_feats = torch.cat(corrupt_feats, dim=0)
        corrupt_lbls = torch.cat(corrupt_lbls, dim=0)
        
        probe_corrupt_acc = clf.score(corrupt_feats[:train_size].numpy(), corrupt_lbls[:train_size].numpy())
        print(f"   -> 128D Linear Probe Accuracy: {probe_corrupt_acc:.4f}")
        
        res = evaluate_oracle_gating(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, clf, proj, device)
        res['probe_acc'] = probe_corrupt_acc
        all_results[corruption] = res
        
        print(f"   -> Perfect Oracle HDC Acc: {res['perfect_acc']:.4f} (Zero-Shot: {res['zero_shot']:.4f})")
        print(f"   -> Leave-One-Out (5k tests): {res['h_count']} Helpful, {res['hm_count']} Harmful")
        if res['hm_count'] > 0:
            print(f"      Helpful Conf: {res['h_conf']:.4f} | Harmful Conf: {res['hm_conf']:.4f}")
            print(f"      Helpful Norm: {res['h_norm']:.4f} | Harmful Norm: {res['hm_norm']:.4f}")

    print("\n\n" + "="*80)
    print(" UNIVERSAL ORACLE GATING RESULTS ")
    print("="*80)
    print(f"| {'Corruption':<16} | {'Probe Acc':<9} | {'Perf. Oracle':<12} | {'Helpful Conf':<12} | {'Harmful Conf':<12} |")
    print("|" + "-"*18 + "|" + "-"*11 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|")
    for corruption, res in all_results.items():
        print(f"| {corruption:<16} | {res['probe_acc']:<9.4f} | {res['perfect_acc']:<12.4f} | {res['h_conf']:<12.4f} | {res['hm_conf']:<12.4f} |")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
