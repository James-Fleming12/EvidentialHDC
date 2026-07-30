import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    import importlib
    ukc = importlib.import_module("unsup_kitti-c")
except ImportError as e:
    print(f"ImportError: {e}\nPlease run this script from the EvidentialHDC root directory.")
    exit(1)

ARCH = ukc.ARCH
DATA = ukc.DATA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_cluster_diagnostics():
    print("Initializing model...")
    ckpt_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = ukc.load_hdc_model(ckpt_path, num_classes=17)
    model.eval()
    
    # Extract Source Features
    print("\nExtracting Source Memory Bank...")
    source_root = "/mnt/alpha/jmfleming/KITTI"
    source_parser = ukc.Parser(root=source_root,
                            train_sequences=['08'],
                            valid_sequences=['08'],
                            test_sequences=['08'],
                            labels=DATA["labels"],
                            color_map=DATA["color_map"],
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"],
                            batch_size=1, workers=4, gt=True, shuffle_train=False)
    
    source_dl = DataLoader(torch.utils.data.Subset(source_parser.validloader.dataset, range(10)), batch_size=1, shuffle=False)
    
    source_feats = []
    source_labels = []
    
    with torch.no_grad():
        for batch in tqdm(source_dl, desc="Source Bank"):
            proj_in = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            if proj_in.shape[1] == 0: continue
            raw_enc, indices, _ = model.encode(proj_in)
            if len(indices) == 0: continue
            
            clean_labels = labels[indices]
            valid = (clean_labels >= 0) & (clean_labels < 17)
            
            source_feats.append(raw_enc[valid].cpu())
            source_labels.append(clean_labels[valid].cpu())
            
    # Subsample X_src on CPU to avoid OOM
    X_src_cpu = torch.cat(source_feats, dim=0)
    Y_src_cpu = torch.cat(source_labels, dim=0)
    if len(X_src_cpu) > 100000:
        perm = torch.randperm(len(X_src_cpu))[:100000]
        X_src_cpu = X_src_cpu[perm]
        Y_src_cpu = Y_src_cpu[perm]
        
    X_src = X_src_cpu.to(device).float()
    X_src = F.normalize(X_src, dim=1)
    Y_src = Y_src_cpu.to(device)
    
    # Compute Source Centroids (Empirical Prototypes)
    src_centroids = torch.zeros((17, X_src.shape[1]), device=device)
    for c in range(17):
        mask = (Y_src == c)
        if mask.sum() > 0:
            src_centroids[c] = F.normalize(X_src[mask].mean(dim=0), dim=0)
            
    corruptions = ['fog', 'crosstalk']
    
    for ct in corruptions:
        print(f"\nEvaluating Diagnostics on {ct}...")
        corruption_root = f"/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/{ct}/heavy"
        
        parser_obj = ukc.Parser(root=corruption_root,
                                train_sequences=['08'],
                                valid_sequences=['08'],
                                test_sequences=['08'],
                                labels=DATA["labels"],
                                color_map=DATA["color_map"],
                                learning_map=DATA["learning_map"],
                                learning_map_inv=DATA["learning_map_inv"],
                                sensor=ARCH["dataset"]["sensor"],
                                max_points=ARCH["dataset"]["max_points"],
                                batch_size=1, workers=4, gt=True, shuffle_train=False)
                                
        dl = DataLoader(torch.utils.data.Subset(parser_obj.validloader.dataset, range(20)), batch_size=1, shuffle=False)
        
        all_feats = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dl, desc=ct):
                proj_in = batch[0].to(device)
                labels = batch[2].to(device).view(-1)
                if proj_in.shape[1] == 0: continue
                raw_enc, indices, _ = model.encode(proj_in)
                if len(indices) == 0: continue
                
                clean_labels = labels[indices]
                valid = (clean_labels >= 0) & (clean_labels < 17)
                
                all_feats.append(raw_enc[valid].cpu())
                all_labels.append(clean_labels[valid].cpu())
                
        # Subsample X_tgt on CPU to avoid OOM
        X_tgt_cpu = torch.cat(all_feats, dim=0)
        Y_tgt_cpu = torch.cat(all_labels, dim=0)
        if len(X_tgt_cpu) > 100000:
            perm = torch.randperm(len(X_tgt_cpu))[:100000]
            X_tgt_cpu = X_tgt_cpu[perm]
            Y_tgt_cpu = Y_tgt_cpu[perm]
            
        X_tgt = X_tgt_cpu.to(device).float()
        X_tgt = F.normalize(X_tgt, dim=1)
        Y_tgt = Y_tgt_cpu.to(device)
        
        # 1. Cluster Drift Diagnostic
        tgt_centroids = torch.zeros((17, X_tgt.shape[1]), device=device)
        print("\n--- Diagnostic 2: Cluster Drift ---")
        print("Class | Cosine Sim (Source Centroid vs Target Centroid)")
        print("-----------------------------------------------------")
        drift_sims = []
        for c in range(17):
            mask = (Y_tgt == c)
            if mask.sum() > 0 and (Y_src == c).sum() > 0:
                tgt_centroids[c] = F.normalize(X_tgt[mask].mean(dim=0), dim=0)
                sim = torch.dot(src_centroids[c], tgt_centroids[c]).item()
                drift_sims.append(sim)
                print(f" {c:4d} | {sim:.4f}")
            else:
                print(f" {c:4d} | N/A (Missing from dataset)")
                
        if len(drift_sims) > 0:
            print(f"\nAverage Centroid Similarity: {sum(drift_sims)/len(drift_sims):.4f}")
            
        # 2. Prototype Voronoi Diagnostic
        # Compare distances using the actual model's learned prototypes
        print("\n--- Diagnostic 5: Prototype Voronoi (Hallucination Depth) ---")
        with torch.no_grad():
            cos_sims = model.classify(X_tgt)
            max_cos, preds = cos_sims.max(dim=1)
            
            # Find hallucinations (wrong predictions)
            is_hallucination = (preds != Y_tgt)
            hal_sims = cos_sims[is_hallucination]
            
            # Sort similarities for hallucinations
            sorted_hal_sims, _ = hal_sims.sort(dim=1, descending=True)
            
            # Margin = Top1_Cos - Top2_Cos
            margins = (sorted_hal_sims[:, 0] - sorted_hal_sims[:, 1]).cpu()
            
            # Average Margin for Hallucinations
            avg_margin = margins.mean().item()
            
            print(f"Average Margin for Hallucinations: {avg_margin:.4f}")
            
            # Bucket margins
            deep_inside = (margins > 0.05).float().mean().item() * 100
            boundary = (margins <= 0.05).float().mean().item() * 100
            print(f"% Hallucinations deep inside a Voronoi Cell (Margin > 0.05): {deep_inside:.1f}%")
            print(f"% Hallucinations near a Decision Boundary (Margin <= 0.05): {boundary:.1f}%")

if __name__ == "__main__":
    run_cluster_diagnostics()
