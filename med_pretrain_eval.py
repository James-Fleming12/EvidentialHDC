import os
import yaml
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer

NUM_CLASSES = 17

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
    
    # 1. Per-Class Cosine Shift and Prototype HDC Accuracy
    print("  -> Calculating Shifts and Prototype Accuracy...")
    shifts = {}
    euc_shifts = {}
    clean_prototypes = {}
    for c in range(NUM_CLASSES):
        c_mask = clean_lbls == c
        f_mask = fog_lbls == c
        if c_mask.sum() > 0:
            c_center_unnorm = clean_feats[c_mask].mean(dim=0)
            clean_prototypes[c] = c_center_unnorm
            
        if c_mask.sum() > 0 and f_mask.sum() > 0:
            f_center_unnorm = fog_feats[f_mask].mean(dim=0)
            c_center = F.normalize(c_center_unnorm.unsqueeze(0), p=2, dim=1)
            f_center = F.normalize(f_center_unnorm.unsqueeze(0), p=2, dim=1)
            shift = 1.0 - torch.cosine_similarity(c_center, f_center).item()
            shifts[f"class_{c}"] = shift
            
            euc_shift = torch.norm(c_center_unnorm - f_center_unnorm).item()
            euc_shifts[f"class_{c}"] = euc_shift
            
    avg_cosine_shift = np.mean(list(shifts.values()))
    avg_euc_shift = np.mean(list(euc_shifts.values()))
    
    # HDC Prototype Accuracy (Nearest Class Mean)
    proto_tensor = torch.stack([clean_prototypes[c] for c in range(NUM_CLASSES) if c in clean_prototypes]).to(device)
    proto_labels = torch.tensor([c for c in range(NUM_CLASSES) if c in clean_prototypes]).to(device)
    
    # Evaluate HDC on Fog
    sub_fog = fog_feats[:50000].to(device)
    sub_fog_lbls = fog_lbls[:50000].to(device)
    dists = torch.cdist(sub_fog.unsqueeze(0), proto_tensor.unsqueeze(0)).squeeze(0)
    proto_preds = proto_labels[dists.argmin(dim=1)]
    hdc_fog_acc = (proto_preds == sub_fog_lbls).float().mean().item()
    
    # 2. Cross-Domain Retrieval (Fog -> Clean)
    print("  -> Calculating Cross-Domain Retrieval...")
    # Subsample to avoid OOM
    sub_clean = clean_feats[:10000].to(device)
    sub_clean_lbls = clean_lbls[:10000].to(device)
    sub_fog_for_retrieval = fog_feats[:10000].to(device)
    sub_fog_lbls_for_retrieval = fog_lbls[:10000].to(device)
        
    # Nearest neighbor of Fog in Clean
    dists_retrieval = torch.cdist(sub_fog_for_retrieval.unsqueeze(0), sub_clean.unsqueeze(0)).squeeze(0)
    retrieval_preds = sub_clean_lbls[dists_retrieval.argmin(dim=1)]
    cross_domain_retrieval = (retrieval_preds == sub_fog_lbls_for_retrieval).float().mean().item()
    
    # 3. Linear Probe (Logistic Regression)
    print("  -> Training Linear Probe...")
    X_train = clean_feats[:50000].numpy()
    y_train = clean_lbls[:50000].numpy()
    X_test = fog_feats[:50000].numpy()
    y_test = fog_lbls[:50000].numpy()
    
    clf = LogisticRegression(max_iter=1000, n_jobs=-1).fit(X_train, y_train)
    probe_clean_acc = clf.score(X_train, y_train)
    probe_fog_acc = clf.score(X_test, y_test)
    
    # 4. Magnitude Segregation
    print("  -> Calculating Magnitude Segregation...")
    clean_mag = torch.norm(clean_feats, p=2, dim=1).mean().item()
    fog_mag = torch.norm(fog_feats, p=2, dim=1).mean().item()
    
    res = {
        "Avg Cosine Shift": avg_cosine_shift,
        "Cross-Domain Retrieval": cross_domain_retrieval,
        "HDC Prototype Accuracy (Fog)": hdc_fog_acc,
        "Linear Probe (Clean)": probe_clean_acc,
        "Linear Probe (Fog)": probe_fog_acc,
        "Linear Robustness Gap": probe_clean_acc - probe_fog_acc,
        "Average L2 Norm (Clean)": clean_mag,
        "Average L2 Norm (Fog)": fog_mag
    }
    
    # Add per-class shifts
    for k, v in shifts.items():
        res[f"Cosine_Shift_{k}"] = v
        
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of medium-scale epochs to run from scratch")
    parser.add_argument("--continue_training", type=int, default=0, help="If > 0, resume from log_dir weights and train for this many extra epochs")
    args = parser.parse_args()

    # Load configurations
    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))

    methods = ['baseline', 'supcon_vib']
    results = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    
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
        print(f"\n{'='*60}")
        print(f" Starting Medium-Scale Pretraining: {method.upper()}")
        print(f" Running {args.epochs} epochs on 100% of data (Scheduler-Safe)")
        print(f"{'='*60}")
        
        # Give each method its own distinct logging and weight-saving directory
        log_dir = os.path.join(args.log_dir, f"med_pretrain_{method}")
        os.makedirs(log_dir, exist_ok=True)
        
        load_path = log_dir if args.continue_training > 0 else None
        epochs_to_run = args.continue_training if args.continue_training > 0 else args.epochs
        
        if load_path:
            print(f" [Resuming training from {load_path} for {epochs_to_run} epochs]")
            
        # Instantiate GenTrainer (100% dataset for safe scheduler convergence)
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, path=load_path, method=method)
        
        # Run the full PyTorch loop for N epochs
        trainer.train(epochs=epochs_to_run)
        
        print(f"Finished {method}. Weights saved to: {log_dir}")
        
        print(f"\n--- Evaluating Headroom for {method.upper()} ---")
        metrics = evaluate_headroom(trainer.model, clean_parser.validloader, fog_parser.validloader, device, num_frames=50)
        results[method] = metrics
        
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    out_path = os.path.join(args.log_dir, "med_pretrain_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved Medium-Pretrain Results to {out_path}")

if __name__ == "__main__":
    main()
