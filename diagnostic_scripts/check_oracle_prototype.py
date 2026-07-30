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

def run_oracle_prototype_diagnostics():
    print("Initializing model...")
    ckpt_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = ukc.load_hdc_model(ckpt_path, num_classes=17)
    model.eval()
    
    corruptions = ['fog', 'crosstalk']
    
    for ct in corruptions:
        print(f"\nEvaluating Diagnostic 16: Oracle Prototype on {ct}...")
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
                
        X_tgt_cpu = torch.cat(all_feats, dim=0)
        Y_tgt_cpu = torch.cat(all_labels, dim=0)
        
        # We need ALL points to compute perfect oracle centroids, but we can compute them on CPU to avoid OOM
        print("Computing Oracle Target Centroids...")
        oracle_centroids = torch.zeros((17, X_tgt_cpu.shape[1]), dtype=torch.float32)
        X_tgt_cpu_norm = F.normalize(X_tgt_cpu.float(), dim=1)
        
        valid_classes = []
        for c in range(17):
            mask = (Y_tgt_cpu == c)
            if mask.sum() > 0:
                oracle_centroids[c] = F.normalize(X_tgt_cpu_norm[mask].mean(dim=0), dim=0)
                valid_classes.append(c)
                
        oracle_centroids = oracle_centroids.to(device)
        
        # Now evaluate accuracy using these PERFECT centroids
        # To avoid OOM, chunk the evaluation
        chunk_size = 10000
        
        correct = 0
        all_correctness = []
        all_max_cos = []
        
        print("Classifying target points using Oracle Target Centroids...")
        for i in range(0, len(X_tgt_cpu_norm), chunk_size):
            chunk = X_tgt_cpu_norm[i:i+chunk_size].to(device)
            y_chunk = Y_tgt_cpu[i:i+chunk_size].to(device)
            
            # [chunk_size, 17]
            sims = torch.mm(chunk, oracle_centroids.t())
            
            # Mask out missing classes by setting their similarity to -1
            for c in range(17):
                if c not in valid_classes:
                    sims[:, c] = -1.0
                    
            max_cos, preds = sims.max(dim=1)
            correct += (preds == y_chunk).sum().item()
            
            all_correctness.append((preds == y_chunk).float())
            all_max_cos.append(max_cos)
            
        oracle_acc = correct / len(X_tgt_cpu_norm)
        
        # Original Baseline Prototype Accuracy
        print("Computing Baseline Prototype Accuracy...")
        baseline_correct = 0
        with torch.no_grad():
            for i in range(0, len(X_tgt_cpu_norm), chunk_size):
                chunk = X_tgt_cpu_norm[i:i+chunk_size].to(device).to(model.classify.weight.dtype)
                y_chunk = Y_tgt_cpu[i:i+chunk_size].to(device)
                sims = model.classify(chunk)
                preds = sims.argmax(dim=1)
                baseline_correct += (preds == y_chunk).sum().item()
        baseline_acc = baseline_correct / len(X_tgt_cpu_norm)
        
        print(f"\n--- Diagnostic 16: Oracle Prototype ---")
        print(f"Baseline (Source) Prototype Accuracy: {baseline_acc:.4f}")
        print(f"Oracle (Target) Prototype Accuracy:   {oracle_acc:.4f}")
        
        if oracle_acc < 0.60:
            print("Verdict: Even with PERFECT oracle placement, 17 prototypes fail spectacularly. The issue is the centroid representation itself, not drift.")
        else:
            print("Verdict: Perfect oracle placement recovers significant accuracy. Drift is the primary culprit.")

if __name__ == "__main__":
    run_oracle_prototype_diagnostics()
