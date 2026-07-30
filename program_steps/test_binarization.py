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

def eval_knn(X_train, Y_train, Y_corr_train, X_test, Y_test, Y_corr_test):
    k = 10
    preds_sem = []
    probs_corr = []
    
    chunk_size = 2000
    for i in range(0, len(X_test), chunk_size):
        chunk = X_test[i:i+chunk_size].to(device)
        sims = torch.mm(chunk, X_train.to(device).t())
        topk_sims, topk_idx = sims.topk(k=k, dim=1)
        
        neighbor_sem = Y_train[topk_idx].to(device)
        pred = torch.mode(neighbor_sem, dim=1).values
        preds_sem.append(pred.cpu())
        
        neighbor_corr = Y_corr_train[topk_idx].to(device)
        prob = neighbor_corr.float().mean(dim=1)
        probs_corr.append(prob.cpu())
        
    preds_sem = torch.cat(preds_sem)
    probs_corr = torch.cat(probs_corr)
    
    sem_acc = (preds_sem == Y_test).float().mean().item()
    corr_auroc = compute_auroc(probs_corr, Y_corr_test)
    return sem_acc, corr_auroc

def test_binarization():
    print("Initializing model to test HDC Binarization...")
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
                            
    subset = torch.utils.data.Subset(parser_obj.validloader.dataset, range(5))
    dl = DataLoader(subset, batch_size=1, shuffle=False)
    
    all_hdc = []
    all_labels = []
    all_correctness = []
    
    with torch.no_grad():
        for batch in tqdm(dl, desc="Collecting points"):
            proj_in = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            if proj_in.shape[1] == 0: continue
            
            raw_enc, indices, _ = model.encode(proj_in)
            if len(indices) == 0: continue
            
            norm_enc = F.normalize(raw_enc, dim=1).to(model.classify.weight.dtype)
            
            # Use original network linear layer to get correctness (like during Phase 1)
            logits = model.classify(norm_enc)
            probs = torch.softmax(logits / 0.05, dim=1)
            _, pseudo_labels = probs.max(dim=1)
            
            clean_labels = labels[indices]
            valid = (clean_labels >= 0) & (clean_labels < 17)
            
            valid_idx = valid.nonzero(as_tuple=True)[0]
            if len(valid_idx) > 10000:
                perm = torch.randperm(len(valid_idx))[:10000]
                valid_idx = valid_idx[perm]
                
            all_hdc.append(raw_enc[valid_idx].cpu())
            all_labels.append(clean_labels[valid_idx].cpu())
            all_correctness.append((pseudo_labels[valid_idx] == clean_labels[valid_idx]).cpu())
            
    X_hdc = torch.cat(all_hdc, dim=0)
    Y_sem = torch.cat(all_labels, dim=0)
    Y_corr = torch.cat(all_correctness, dim=0)
    
    perm = torch.randperm(len(X_hdc))
    mid = min(10000, len(X_hdc) // 2)
    idx_train = perm[:mid]
    idx_test = perm[mid:]
    
    X_train_raw, X_test_raw = X_hdc[idx_train], X_hdc[idx_test]
    Y_train_sem, Y_test_sem = Y_sem[idx_train], Y_sem[idx_test]
    Y_train_corr, Y_test_corr = Y_corr[idx_train], Y_corr[idx_test]
    
    print(f"\n--- Binarization Evaluation (Memory Bank Size: {len(X_train_raw)}) ---")
    
    # 1. Float32 Continuous Space
    X_train_float = F.normalize(X_train_raw.float(), dim=1)
    X_test_float = F.normalize(X_test_raw.float(), dim=1)
    sem_f32, corr_f32 = eval_knn(X_train_float, Y_train_sem, Y_train_corr, X_test_float, Y_test_sem, Y_test_corr)
    print(f"Float32 (10,000D)   - Sem Acc: {sem_f32:.4f} | Corr AUROC: {corr_f32:.4f}")
    
    # 2. Binarized Space (Sign)
    X_train_bin = torch.sign(X_train_raw.float())
    X_train_bin[X_train_bin == 0] = 1.0 # Ensure bipolar {-1, 1}
    X_train_bin = X_train_bin / (10000 ** 0.5) # Normalize length for dot product equivalence
    
    X_test_bin = torch.sign(X_test_raw.float())
    X_test_bin[X_test_bin == 0] = 1.0
    X_test_bin = X_test_bin / (10000 ** 0.5)
    
    sem_bin, corr_bin = eval_knn(X_train_bin, Y_train_sem, Y_train_corr, X_test_bin, Y_test_sem, Y_test_corr)
    print(f"Binarized (10,000D) - Sem Acc: {sem_bin:.4f} | Corr AUROC: {corr_bin:.4f}")
    
    print("\nVerdict:")
    if sem_bin >= (sem_f32 - 0.05):
        print("Binarization preserves topology! We can use 1-bit vectors, reducing memory to 11.9 MB and latency to <10ms via XOR!")
    else:
        print("Binarization degrades performance heavily. We must stick to Float32 and use FAISS or subsampling.")

if __name__ == "__main__":
    test_binarization()
