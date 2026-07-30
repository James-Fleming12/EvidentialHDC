import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import numpy as np
from scipy.stats import spearmanr

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

def compute_participation_ratio(X):
    # X: [N, D]
    X = X - X.mean(dim=0)
    cov = torch.mm(X.t(), X) / (X.shape[0] - 1)
    trace = torch.trace(cov)
    trace_sq = torch.trace(torch.mm(cov, cov))
    if trace_sq == 0:
        return 0.0
    pr = (trace ** 2) / trace_sq
    return pr.item()

def compute_purity(X, y, k=10):
    X_norm = F.normalize(X.float(), dim=1)
    chunk_size = 2000
    purity_scores = []
    
    for i in range(0, len(X_norm), chunk_size):
        chunk = X_norm[i:i+chunk_size]
        sims = torch.mm(chunk, X_norm.t())
        
        topk_sims, topk_idx = sims.topk(k=k+1, dim=1)
        topk_idx = topk_idx[:, 1:] # exclude self
        
        neighbor_labels = y[topk_idx]
        target_labels = y[i:i+chunk_size].unsqueeze(1)
        
        matches = (neighbor_labels == target_labels).float()
        purity_scores.append(matches.mean(dim=1))
        
    return torch.cat(purity_scores).mean().item()

def compute_linear_cka(X, Y):
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)
    
    cross_cov = torch.mm(Y.t(), X)
    num = torch.norm(cross_cov, p='fro') ** 2
    
    cov_X = torch.mm(X.t(), X)
    cov_Y = torch.mm(Y.t(), Y)
    
    den = torch.norm(cov_X, p='fro') * torch.norm(cov_Y, p='fro')
    return (num / den).item()

def compute_neighborhood_rank_preservation(X_src, X_tgt, k=50):
    N = min(1000, len(X_src))
    perm = torch.randperm(len(X_src))[:N]
    
    X_src_subset = F.normalize(X_src[perm].float(), dim=1)
    X_src_all = F.normalize(X_src.float(), dim=1)
    
    X_tgt_subset = F.normalize(X_tgt[perm].float(), dim=1)
    X_tgt_all = F.normalize(X_tgt.float(), dim=1)
    
    sims_src = torch.mm(X_src_subset, X_src_all.t())
    topk_sims_src, topk_idx = sims_src.topk(k=k+1, dim=1)
    topk_idx = topk_idx[:, 1:] 
    
    spearman_scores = []
    
    for i in range(N):
        sims_src_i = torch.mm(X_src_subset[i:i+1], X_src_all[topk_idx[i]].t()).squeeze()
        sims_tgt_i = torch.mm(X_tgt_subset[i:i+1], X_tgt_all[topk_idx[i]].t()).squeeze()
        
        src_ranks = sims_src_i.cpu().numpy()
        tgt_ranks = sims_tgt_i.cpu().numpy()
        
        corr, _ = spearmanr(src_ranks, tgt_ranks)
        if not np.isnan(corr):
            spearman_scores.append(corr)
            
    return np.mean(spearman_scores)

def run_phase2_geometry():
    print("Initializing model for Phase II Geometry Characterization...")
    ckpt_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = ukc.load_hdc_model(ckpt_path, num_classes=17)
    model.eval()
    
    ct = 'fog'
    print(f"\nExtracting Representations on {ct}...")
    corruption_root = f"/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/{ct}/heavy"
    if not os.path.exists(corruption_root):
        corruption_root = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/"
    
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
                            
    subset = torch.utils.data.Subset(parser_obj.validloader.dataset, range(10))
    dl = DataLoader(subset, batch_size=1, shuffle=False)
    
    all_backbone = []
    all_hdc = []
    all_sims = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dl):
            proj_in = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            if proj_in.shape[1] == 0: continue
            
            with torch.amp.autocast('cuda', enabled=True):
                backbone = model.net(proj_in, only_feat=True)
                
            raw_enc, indices, _ = model.encode(proj_in)
            if len(indices) == 0: continue
            
            backbone = backbone.permute(0, 2, 3, 1).reshape(-1, 128)[indices]
            norm_enc = F.normalize(raw_enc, dim=1).to(model.classify.weight.dtype)
            cos_sims = model.classify(norm_enc)
            clean_labels = labels[indices]
            
            valid = (clean_labels >= 0) & (clean_labels < 17)
            valid_idx = valid.nonzero(as_tuple=True)[0]
            if len(valid_idx) > 2000:
                perm = torch.randperm(len(valid_idx))[:2000]
                valid_idx = valid_idx[perm]
            
            all_backbone.append(backbone[valid_idx].cpu())
            all_hdc.append(raw_enc[valid_idx].cpu())
            all_sims.append(cos_sims[valid_idx].cpu())
            all_labels.append(clean_labels[valid_idx].cpu())
            
    X_bb = torch.cat(all_backbone, dim=0).to(device).float()
    X_hdc = torch.cat(all_hdc, dim=0).to(device).float()
    X_sims = torch.cat(all_sims, dim=0).to(device).float()
    Y_sem = torch.cat(all_labels, dim=0).to(device).long()
    
    print("\n--- Phase II: Geometry Characterization ---")
    
    print("Computing Effective Rank...")
    pr_bb = compute_participation_ratio(X_bb)
    pr_hdc = compute_participation_ratio(X_hdc)
    pr_sims = compute_participation_ratio(X_sims)
    
    print("Computing Neighborhood Purity...")
    purity_hdc = compute_purity(X_hdc, Y_sem, k=10)
    purity_sims = compute_purity(X_sims, Y_sem, k=10)
    
    print("Computing Geometry Preservation...")
    cka_score = compute_linear_cka(X_hdc, X_sims)
    spearman = compute_neighborhood_rank_preservation(X_hdc, X_sims, k=50)
    
    print("\n=== GLOBAL GEOMETRY ===")
    print("Effective Rank (Participation Ratio):")
    print(f"  Backbone (128D):       {pr_bb:.1f} dimensions")
    print(f"  HDC Embedding (10k-D): {pr_hdc:.1f} dimensions")
    print(f"  Prototype Sims (17D):  {pr_sims:.1f} dimensions")
    
    print("\n=== LOCAL GEOMETRY ===")
    print("Neighborhood Purity (k=10):")
    print(f"  HDC Embedding (10k-D): {purity_hdc*100:.2f}%")
    print(f"  Prototype Sims (17D):  {purity_sims*100:.2f}%")
    
    print("\n=== GEOMETRY PRESERVATION (HDC -> Prototype Sims) ===")
    print(f"Linear CKA (Global Equivalence): {cka_score:.4f}")
    print(f"Neighborhood Rank Preservation (Spearman, k=50): {spearman:.4f}")
    
    print("\nInterpretation:")
    if spearman < 0.5:
        print("Verdict: The prototype projection severely destroys local geometric topology (ranking collapses).")
    else:
        print("Verdict: Local geometric topology is preserved.")

if __name__ == "__main__":
    run_phase2_geometry()
