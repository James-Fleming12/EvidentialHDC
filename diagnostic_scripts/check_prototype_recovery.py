import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze(1)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(1)

def fit_classifier(X, y, model_type="logistic", epochs=1000, lr=0.01):
    if model_type == "logistic":
        model = LogisticRegression(X.shape[1]).to(X.device)
    else:
        model = MLP(X.shape[1]).to(X.device)
        
    # Scale LR down for MLP with high dimensions to avoid divergence
    if model_type == "mlp" and X.shape[1] > 1000:
        lr = 0.001
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Subsample if dataset is too massive to train quickly
    if len(X) > 100000:
        perm = torch.randperm(len(X))[:100000]
        X_train = X[perm]
        y_train = y[perm]
    else:
        X_train = X
        y_train = y
        
    X_train = (X_train - X_train.mean(dim=0)) / (X_train.std(dim=0) + 1e-8)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        X_eval = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
        probs = torch.sigmoid(model(X_eval))
        auroc = compute_auroc(probs, y)
    return auroc


def run_prototype_recovery():
    print("Initializing model...")
    ckpt_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = ukc.load_hdc_model(ckpt_path, num_classes=17)
    model.eval()
    
    # 1. Extract Source Bank
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
            
    source_bank = torch.cat(source_feats, dim=0)
    
    # We only need 5000 vectors max for the test
    if len(source_bank) > 5000:
        perm = torch.randperm(len(source_bank))[:5000]
        source_bank = source_bank[perm]
        
    source_bank = F.normalize(source_bank, dim=1).to(device).float()
    
    corruptions = ['crosstalk']
    
    for ct in corruptions:
        print(f"\nExtracting Target Features on {ct}...")
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
        all_correctness = []
        
        with torch.no_grad():
            for batch in tqdm(dl, desc=ct):
                proj_in = batch[0].to(device)
                labels = batch[2].to(device).view(-1)
                if proj_in.shape[1] == 0: continue
                raw_enc, indices, _ = model.encode(proj_in)
                if len(indices) == 0: continue
                
                norm_enc = F.normalize(raw_enc, dim=1).to(model.classify.weight.dtype)
                cos_sims = model.classify(norm_enc)
                clean_preds = cos_sims.argmax(dim=1)
                clean_labels = labels[indices]
                
                correctness = (clean_preds == clean_labels).float()
                valid_gt = (clean_labels >= 0) & (clean_labels < 17)
                
                all_feats.append(raw_enc[valid_gt].cpu())
                all_correctness.append(correctness[valid_gt].cpu())
                
        # Subsample Target to 50k for speed
        X_target = torch.cat(all_feats, dim=0)
        Y_target = torch.cat(all_correctness, dim=0)
        
        if len(X_target) > 50000:
            perm = torch.randperm(len(X_target))[:50000]
            X_target = X_target[perm]
            Y_target = Y_target[perm]
            
        X_target = F.normalize(X_target, dim=1).to(device).float()
        Y_target = Y_target.to(device)
        
        print(f"\n--- Prototype Information Recovery Curve ({ct}) ---")
        ref_counts = [17, 50, 100, 500, 1000, 5000]
        
        print("Num_Refs | Logistic_AUROC | MLP_AUROC")
        print("---------------------------------------")
        for num_refs in ref_counts:
            # Randomly select num_refs from the source bank
            refs = source_bank[:num_refs] # already randomized
            
            # Compute similarities [N, num_refs]
            # using chunking to avoid OOM
            sims = []
            chunk_size = 10000
            with torch.no_grad():
                for i in range(0, len(X_target), chunk_size):
                    chunk = X_target[i:i+chunk_size]
                    sim = torch.mm(chunk, refs.t())
                    sims.append(sim)
            X_sims = torch.cat(sims, dim=0)
            
            # Train classifiers
            base_auroc = fit_classifier(X_sims, Y_target, model_type="logistic", epochs=1000)
            mlp_auroc = fit_classifier(X_sims, Y_target, model_type="mlp", epochs=1000)
            
            print(f"{num_refs:>8} | {base_auroc:>14.4f} | {mlp_auroc:>9.4f}")

if __name__ == "__main__":
    run_prototype_recovery()
