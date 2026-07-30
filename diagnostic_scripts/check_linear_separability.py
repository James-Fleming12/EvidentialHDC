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


def run_separability_diagnostics():
    print("Initializing model...")
    ckpt_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = ukc.load_hdc_model(ckpt_path, num_classes=17)
    model.eval()
    
    corruptions = ['fog', 'crosstalk']
    
    for ct in corruptions:
        print(f"\nExtracting HDC Embeddings on {ct}...")
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
                                batch_size=1,
                                workers=4,
                                gt=True,
                                shuffle_train=False)
        ds = parser_obj.validloader.dataset
        subset = torch.utils.data.Subset(ds, range(20))
        dl = DataLoader(subset, batch_size=1, shuffle=False)
        
        all_features = []
        all_correctness = []
        
        with torch.no_grad():
            for batch in tqdm(dl):
                proj_in = batch[0].to(device)
                labels = batch[2].to(device).view(-1)
                
                if proj_in.shape[1] == 0:
                    continue
                    
                raw_enc, indices, _ = model.encode(proj_in)
                if len(indices) == 0:
                    continue
                    
                norm_enc = F.normalize(raw_enc, dim=1).to(model.classify.weight.dtype)
                cos_sims = model.classify(norm_enc)
                clean_preds = cos_sims.argmax(dim=1)
                clean_labels = labels[indices]
                
                correctness = (clean_preds == clean_labels).float()
                valid_gt = (clean_labels >= 0) & (clean_labels < 17)
                
                all_features.append(raw_enc[valid_gt].cpu())
                all_correctness.append(correctness[valid_gt].cpu())
                
        X = torch.cat(all_features, dim=0).to(device)
        y = torch.cat(all_correctness, dim=0).to(device)
        
        print(f"\n--- HDC Linear Separability Test ({ct}) ---")
        base_auroc = fit_classifier(X, y, model_type="logistic")
        print(f"HDC 128D Embeddings AUROC (Logistic): {base_auroc:.4f}")
        
        mlp_auroc = fit_classifier(X, y, model_type="mlp")
        print(f"HDC 128D Embeddings AUROC (Nonlinear MLP): {mlp_auroc:.4f}")
            
if __name__ == "__main__":
    run_separability_diagnostics()
