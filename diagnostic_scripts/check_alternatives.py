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

def compute_knn(features, bank_features, bank_labels=None, k=10):
    # features: [N, D], bank_features: [M, D]
    # normalize just in case
    f1 = F.normalize(features, dim=1).float()
    f2 = F.normalize(bank_features, dim=1).float()
    
    # Compute cosine similarity matrix: [N, M]
    sim = torch.mm(f1, f2.t())
    topk_sim, topk_idx = sim.topk(k, dim=1)
    
    res = {"density": topk_sim.mean(dim=1)}
    
    if bank_labels is not None:
        # topk_idx has shape [N, k]
        # bank_labels has shape [M]
        topk_labels = bank_labels[topk_idx] # [N, k]
        res["labels"] = topk_labels
        
    return res

def run_diagnostics():
    print("Initializing model...")
    ckpt_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = ukc.load_hdc_model(ckpt_path, num_classes=17)
    model.eval()
    
    # 1. Extract Source Bank (DD-2 / AC-3)
    print("\nExtracting Source Memory Bank...")
    source_root = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/"
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
            
    source_bank = torch.cat(source_feats, dim=0)
    source_bank_labels = torch.cat(source_labels, dim=0)
    
    # Subsample source bank to 50k for memory efficiency during k-NN
    if len(source_bank) > 50000:
        perm = torch.randperm(len(source_bank))[:50000]
        source_bank = source_bank[perm]
        source_bank_labels = source_bank_labels[perm]
        
    source_bank = source_bank.to(device).float()
    source_bank_labels = source_bank_labels.to(device)
    
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
        all_preds = []
        all_logits = []
        all_max_cos = []
        
        with torch.no_grad():
            for batch in tqdm(dl, desc=ct):
                proj_in = batch[0].to(device)
                labels = batch[2].to(device).view(-1)
                if proj_in.shape[1] == 0: continue
                raw_enc, indices, _ = model.encode(proj_in)
                if len(indices) == 0: continue
                
                norm_enc = F.normalize(raw_enc, dim=1).to(model.classify.weight.dtype)
                cos_sims = model.classify(norm_enc)
                max_cos, preds = cos_sims.max(dim=1)
                
                clean_labels = labels[indices]
                valid = (clean_labels >= 0) & (clean_labels < 17)
                
                all_feats.append(raw_enc[valid].cpu())
                all_labels.append(clean_labels[valid].cpu())
                all_preds.append(preds[valid].cpu())
                all_logits.append(cos_sims[valid].cpu())
                all_max_cos.append(max_cos[valid].cpu())
                
        # Aggregate Target
        X_target = torch.cat(all_feats, dim=0).to(device).float()
        Y_target = torch.cat(all_labels, dim=0).to(device)
        Preds = torch.cat(all_preds, dim=0).to(device)
        Logits = torch.cat(all_logits, dim=0).to(device)
        MaxCos = torch.cat(all_max_cos, dim=0).to(device)
        
        # We may need to chunk X_target to avoid OOM during k-NN
        chunk_size = 10000
        
        # 1. PL-1: Top-k Candidate Coverage
        print("\n--- PL-1: Top-k Candidate Coverage ---")
        sorted_logits = torch.argsort(Logits, dim=1, descending=True)
        top1 = (sorted_logits[:, 0] == Y_target).float().mean().item()
        top2 = ((sorted_logits[:, :2] == Y_target.unsqueeze(1)).sum(dim=1) > 0).float().mean().item()
        top3 = ((sorted_logits[:, :3] == Y_target.unsqueeze(1)).sum(dim=1) > 0).float().mean().item()
        top5 = ((sorted_logits[:, :5] == Y_target.unsqueeze(1)).sum(dim=1) > 0).float().mean().item()
        print(f"Top-1 Accuracy: {top1:.4f}")
        print(f"Top-2 Accuracy: {top2:.4f}")
        print(f"Top-3 Accuracy: {top3:.4f}")
        print(f"Top-5 Accuracy: {top5:.4f}")
        
        # 2. DD-2 / AC-3: Oracle Source Projection (1-NN Accuracy)
        print("\n--- DD-2 / MB-3: Oracle Source Projection (k-NN vs Prototype) ---")
        knn_correct = 0
        knn_purity = 0
        target_purity = 0
        
        # Subsample target for self-kNN to avoid OOM
        target_bank = X_target
        target_bank_labels = Y_target
        if len(target_bank) > 50000:
            perm = torch.randperm(len(target_bank))[:50000]
            target_bank = target_bank[perm]
            target_bank_labels = target_bank_labels[perm]
            
        for i in range(0, len(X_target), chunk_size):
            chunk = X_target[i:i+chunk_size]
            y_chunk = Y_target[i:i+chunk_size]
            
            # k-NN against source bank
            res = compute_knn(chunk, source_bank, source_bank_labels, k=10)
            knn_preds = res["labels"][:, 0] # 1-NN prediction
            knn_correct += (knn_preds == y_chunk).sum().item()
            
            # Purity in source bank (k=10)
            purity = (res["labels"] == y_chunk.unsqueeze(1)).float().mean(dim=1).sum().item()
            knn_purity += purity
            
            # k-NN against target itself (MB-2)
            res_target = compute_knn(chunk, target_bank, target_bank_labels, k=10)
            t_purity = (res_target["labels"] == y_chunk.unsqueeze(1)).float().mean(dim=1).sum().item()
            target_purity += t_purity
            
        knn_acc = knn_correct / len(X_target)
        print(f"Prototype Accuracy:    {top1:.4f}")
        print(f"Source 1-NN Accuracy:  {knn_acc:.4f}")
        
        # 3. MB-2: Neighborhood Purity
        print("\n--- MB-2: Neighborhood Purity ---")
        print(f"Avg Source Neighborhood Purity (k=10): {knn_purity / len(X_target):.4f}")
        print(f"Avg Target Neighborhood Purity (k=10): {target_purity / len(X_target):.4f}")
        print(f"Prototype 'Purity' (Top-1 Accuracy):   {top1:.4f}")

if __name__ == "__main__":
    run_diagnostics()
