import torch
import torch.nn as nn
import torch.optim as optim
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

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # If in_dim is 10k, hidden_dim of 256 is plenty for probing
        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def fit_probe(X, y, probe_type="linear", task="semantic", epochs=300, lr=0.01):
    X = X.to(device)
    y = y.to(device)
    
    out_dim = 17 if task == "semantic" else 1
    
    if probe_type == "linear":
        model = nn.Linear(X.shape[1], out_dim).to(device)
    else:
        model = MLP(X.shape[1], out_dim).to(device)
        if X.shape[1] > 1000:
            lr = 0.001
            
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if task == "semantic":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        y = y.float().unsqueeze(1)
        
    X_train = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        out = model(X_train)
        if task == "semantic":
            preds = out.argmax(dim=1)
            score = (preds == y).float().mean().item()
        else:
            probs = torch.sigmoid(out).squeeze()
            score = compute_auroc(probs, y.squeeze())
            
    return score

def get_knn_score(X, y, task="semantic"):
    # Split into 50% train, 50% test for kNN to prevent 100% on self
    perm = torch.randperm(len(X))
    mid = len(X) // 2
    idx_train = perm[:mid]
    idx_test = perm[mid:]
    
    X_train, y_train = X[idx_train].to(device), y[idx_train].to(device)
    X_test, y_test = X[idx_test].to(device), y[idx_test].to(device)
    
    # Cosine distance
    X_train_norm = F.normalize(X_train, dim=1)
    X_test_norm = F.normalize(X_test, dim=1)
    
    # compute chunked to avoid OOM
    chunk_size = 2000
    preds = []
    
    for i in range(0, len(X_test_norm), chunk_size):
        chunk = X_test_norm[i:i+chunk_size]
        sims = torch.mm(chunk, X_train_norm.t())
        
        # 1-NN
        max_idx = sims.argmax(dim=1)
        preds.append(y_train[max_idx])
        
    preds = torch.cat(preds)
    
    if task == "semantic":
        return (preds == y_test).float().mean().item()
    else:
        # For correctness, we return AUROC using 1-NN predictability? No, knn just outputs a binary pred.
        # Let's just return accuracy for the correctness kNN too
        return (preds == y_test).float().mean().item()

def run_phase1_audit():
    print("Initializing model for Phase I Representation Audit...")
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
    all_correctness = []
    
    with torch.no_grad():
        for batch in tqdm(dl):
            proj_in = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            if proj_in.shape[1] == 0: continue
            
            # Backbone
            with torch.amp.autocast('cuda', enabled=True):
                backbone = model.net(proj_in, only_feat=True)
            
            # HDC
            raw_enc, indices, _ = model.encode(proj_in)
            if len(indices) == 0: continue
            
            backbone = backbone.permute(0, 2, 3, 1).reshape(-1, 128)[indices]
            
            # Prototype Sims
            norm_enc = F.normalize(raw_enc, dim=1).to(model.classify.weight.dtype)
            cos_sims = model.classify(norm_enc)
            clean_preds = cos_sims.argmax(dim=1)
            clean_labels = labels[indices]
            
            valid = (clean_labels >= 0) & (clean_labels < 17)
            
            all_backbone.append(backbone[valid].cpu())
            all_hdc.append(raw_enc[valid].cpu())
            all_sims.append(cos_sims[valid].cpu())
            all_labels.append(clean_labels[valid].cpu())
            all_correctness.append((clean_preds[valid] == clean_labels[valid]).cpu())
            
    X_bb = torch.cat(all_backbone, dim=0).float()
    X_hdc = torch.cat(all_hdc, dim=0).float()
    X_sims = torch.cat(all_sims, dim=0).float()
    Y_sem = torch.cat(all_labels, dim=0).long()
    Y_corr = torch.cat(all_correctness, dim=0).long()
    
    # Subsample to 20,000 for probing to save memory and time
    if len(X_bb) > 20000:
        perm = torch.randperm(len(X_bb))[:20000]
        X_bb = X_bb[perm]
        X_hdc = X_hdc[perm]
        X_sims = X_sims[perm]
        Y_sem = Y_sem[perm]
        Y_corr = Y_corr[perm]
        
    print("\n--- Phase I: Representation Audit (Predictive Decodability) ---")
    
    representations = [
        ("Backbone (128D)", X_bb),
        ("HDC Embedding (10000D)", X_hdc),
        ("Prototype Similarities (17D)", X_sims)
    ]
    
    results = {}
    
    for rep_name, X in representations:
        print(f"\nProbing {rep_name}...")
        
        # 1. Semantic Label Probes (Accuracy)
        sem_knn = get_knn_score(X, Y_sem, task="semantic")
        sem_lin = fit_probe(X, Y_sem, probe_type="linear", task="semantic")
        sem_mlp = fit_probe(X, Y_sem, probe_type="mlp", task="semantic")
        
        # 2. Correctness Probes (AUROC/Acc)
        corr_knn = get_knn_score(X, Y_corr, task="correctness")
        corr_lin = fit_probe(X, Y_corr, probe_type="linear", task="correctness")
        corr_mlp = fit_probe(X, Y_corr, probe_type="mlp", task="correctness")
        
        results[rep_name] = {
            "Sem_kNN": sem_knn,
            "Sem_Lin": sem_lin,
            "Sem_MLP": sem_mlp,
            "Corr_kNN": corr_knn,  # Note: this is Acc for kNN
            "Corr_Lin": corr_lin,
            "Corr_MLP": corr_mlp,
        }
        
    print("\n=== FINAL AUDIT REPORT ===")
    print(f"{'Representation Layer':<28} | {'Target':<14} | {'k-NN':<7} | {'Linear':<7} | {'MLP':<7}")
    print("-" * 73)
    for rep_name in results.keys():
        r = results[rep_name]
        print(f"{rep_name:<28} | {'Semantic Acc':<14} | {r['Sem_kNN']:.4f}  | {r['Sem_Lin']:.4f}  | {r['Sem_MLP']:.4f}")
        print(f"{'':<28} | {'Corr Metric':<14} | {r['Corr_kNN']:.4f}  | {r['Corr_Lin']:.4f}  | {r['Corr_MLP']:.4f}")
        print("-" * 73)
        
    print("\nNote: Corr Metric is Accuracy for k-NN, and AUROC for Linear/MLP probes.")

if __name__ == "__main__":
    run_phase1_audit()
