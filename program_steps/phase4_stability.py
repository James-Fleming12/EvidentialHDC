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

def compute_auroc(probs, labels):
    labels = labels.float()
    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    desc_score_indices = torch.argsort(probs, descending=True)
    labels = labels[desc_score_indices]
    
    tps = torch.cumsum(labels, dim=0).float()
    fps = torch.cumsum(1.0 - labels, dim=0).float()
    
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    fpr_diff = torch.cat([fpr[0:1], fpr[1:] - fpr[:-1]])
    auroc = torch.sum(fpr_diff * tpr).item()
    return auroc

def run_memory_sweep(X_train, y_train, y_corr_train, X_test, y_test, y_corr_test, memory_sizes):
    # Normalize for cosine similarity
    X_train_norm = F.normalize(X_train.float(), dim=1)
    X_test_norm = F.normalize(X_test.float(), dim=1)
    
    results = []
    
    for m_size in memory_sizes:
        if m_size > len(X_train_norm):
            m_size = len(X_train_norm)
            
        # Randomly sample the memory bank
        perm = torch.randperm(len(X_train_norm))[:m_size]
        mem_X = X_train_norm[perm].to(device)
        mem_y = y_train[perm].to(device)
        mem_corr = y_corr_train[perm].to(device)
        
        chunk_size = 2000
        preds_sem = []
        probs_corr = []
        
        # dynamic k based on memory size to ensure stability
        k = min(10, m_size)
        
        for i in range(0, len(X_test_norm), chunk_size):
            chunk = X_test_norm[i:i+chunk_size].to(device)
            sims = torch.mm(chunk, mem_X.t())
            
            topk_sims, topk_idx = sims.topk(k=k, dim=1)
            
            # Semantic
            neighbor_sem = mem_y[topk_idx]
            pred = torch.mode(neighbor_sem, dim=1).values
            preds_sem.append(pred.cpu())
            
            # Correctness (Hallucination detection)
            neighbor_corr = mem_corr[topk_idx]
            prob = neighbor_corr.float().mean(dim=1)
            probs_corr.append(prob.cpu())
            
        preds_sem = torch.cat(preds_sem)
        probs_corr = torch.cat(probs_corr)
        
        sem_acc = (preds_sem == y_test).float().mean().item()
        corr_auroc = compute_auroc(probs_corr, y_corr_test)
        
        results.append((m_size, sem_acc, corr_auroc))
        
    return results

def run_phase4_stability():
    print("Initializing model for Phase IV Stability Analysis...")
    ckpt_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = ukc.load_hdc_model(ckpt_path, num_classes=17)
    model.eval()
    
    ct = 'fog'
    print(f"\nExtracting Representations on {ct} for Memory Sweep...")
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
                            
    # We use 10 frames to collect a large diverse pool of points
    subset = torch.utils.data.Subset(parser_obj.validloader.dataset, range(10))
    dl = DataLoader(subset, batch_size=1, shuffle=False)
    
    all_bb = []
    all_hdc = []
    all_labels = []
    all_correctness = []
    
    with torch.no_grad():
        for batch in tqdm(dl, desc="Collecting points"):
            proj_in = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            if proj_in.shape[1] == 0: continue
            
            with torch.amp.autocast('cuda', enabled=True):
                backbone = model.net(proj_in, only_feat=True)
                raw_enc, indices, _ = model.encode(proj_in)
                
            if len(indices) == 0: continue
            
            backbone = backbone.permute(0, 2, 3, 1).reshape(-1, 128)[indices]
            
            norm_enc = F.normalize(raw_enc, dim=1).to(model.classify.weight.dtype)
            logits = model.classify(norm_enc)
            probs = torch.softmax(logits / 0.05, dim=1)
            conf, pseudo_labels = probs.max(dim=1)
            
            clean_labels = labels[indices]
            valid = (clean_labels >= 0) & (clean_labels < 17)
            
            # Sample 10,000 points per frame to get a massive 100k pool
            valid_idx = valid.nonzero(as_tuple=True)[0]
            if len(valid_idx) > 10000:
                perm = torch.randperm(len(valid_idx))[:10000]
                valid_idx = valid_idx[perm]
                
            all_bb.append(backbone[valid_idx].cpu())
            all_hdc.append(raw_enc[valid_idx].cpu())
            all_labels.append(clean_labels[valid_idx].cpu())
            all_correctness.append((pseudo_labels[valid_idx] == clean_labels[valid_idx]).cpu())
            
    X_bb = torch.cat(all_bb, dim=0)
    X_hdc = torch.cat(all_hdc, dim=0)
    Y_sem = torch.cat(all_labels, dim=0)
    Y_corr = torch.cat(all_correctness, dim=0)
    
    print(f"Total points collected: {len(X_bb)}")
    
    # Split into Train (Memory Bank Pool) and Test (Queries)
    perm = torch.randperm(len(X_bb))
    mid = len(X_bb) // 2
    idx_train = perm[:mid]
    idx_test = perm[mid:]
    
    # We will test using 20,000 queries to save time, against variable sized memory banks
    idx_test = idx_test[:20000]
    
    X_bb_train, X_bb_test = X_bb[idx_train], X_bb[idx_test]
    X_hdc_train, X_hdc_test = X_hdc[idx_train], X_hdc[idx_test]
    Y_sem_train, Y_sem_test = Y_sem[idx_train], Y_sem[idx_test]
    Y_corr_train, Y_corr_test = Y_corr[idx_train], Y_corr[idx_test]
    
    memory_sizes = [50, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    
    print("\n--- Phase IV: Stability Analysis (Memory Capacity Scaling) ---")
    print("\nEvaluating Backbone (128D) Capacity...")
    bb_results = run_memory_sweep(X_bb_train, Y_sem_train, Y_corr_train, 
                                  X_bb_test, Y_sem_test, Y_corr_test, memory_sizes)
                                  
    print("Evaluating HDC Embedding (10,000D) Capacity...")
    hdc_results = run_memory_sweep(X_hdc_train, Y_sem_train, Y_corr_train, 
                                   X_hdc_test, Y_sem_test, Y_corr_test, memory_sizes)
                                   
    print("\n=== MEMORY CAPACITY SCALING REPORT ===")
    print(f"{'Memory Size':<12} | {'BB Sem Acc':<11} | {'HDC Sem Acc':<12} | {'BB Corr AUROC':<14} | {'HDC Corr AUROC':<14}")
    print("-" * 75)
    
    for (size, bb_sem, bb_corr), (_, hdc_sem, hdc_corr) in zip(bb_results, hdc_results):
        print(f"{size:<12} | {bb_sem:.4f}      | {hdc_sem:.4f}       | {bb_corr:.4f}         | {hdc_corr:.4f}")
        
    print("\nVerdict:")
    print("Look for the saturation point where increasing memory size no longer provides significant gains.")
    print("This explicitly dictates the required size of the upcoming Adaptive Memory Bank algorithm.")

if __name__ == "__main__":
    run_phase4_stability()
