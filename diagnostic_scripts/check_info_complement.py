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

def fit_logistic_regression(X, y, epochs=500, lr=0.1):
    model = LogisticRegression(X.shape[1]).to(X.device)
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

def compute_epistemic_uncertainty(logits):
    scaled_logits = logits * 15.0
    evidence = F.softplus(scaled_logits)
    alphas = evidence + 1.0
    S = torch.sum(alphas, dim=1)
    K = alphas.shape[1]
    return K / S

def run_info_diagnostics():
    print("Initializing model...")
    ckpt_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = ukc.load_hdc_model(ckpt_path, num_classes=17)
    model.eval()
    
    corruptions = ['fog', 'crosstalk']
    
    for ct in corruptions:
        print(f"\nGathering TTA Signals on {ct}...")
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
        
        all_signals = []
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
                
                # Signal 1: Dirichlet Epistemic Uncertainty
                dist_unc = compute_epistemic_uncertainty(cos_sims)
                
                # Signal 2: Cosine Confidence (Max Sim)
                max_cos = cos_sims.max(dim=1)[0]
                
                # Signal 3: Entropy
                probs = F.softmax(cos_sims * 100.0, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                
                # Signal 4: Prototype Margin
                top2_cos = torch.topk(cos_sims, k=2, dim=1)[0]
                margin = top2_cos[:, 0] - top2_cos[:, 1]
                
                # Signal 5: Feature Norm
                feat_norm = raw_enc.norm(dim=1)
                
                # Signal 6: Consistency (M2 Augmentation)
                aug_in = proj_in.clone()
                drop_mask = torch.rand_like(aug_in[:, 0:1, :, :]) > 0.20
                aug_in = aug_in * drop_mask
                noise = torch.randn_like(aug_in[:, 1:4, :, :]) * 0.05
                aug_in[:, 1:4, :, :] += noise * drop_mask
                
                aug_logits, _, aug_indices, _ = model(aug_in)
                aug_preds = aug_logits.argmax(dim=1) if len(aug_indices) > 0 else torch.zeros(0, device=device)
                
                B, C_ch, H, W = proj_in.shape
                full_aug_preds = torch.full((B * H * W,), -1, device=device, dtype=clean_preds.dtype)
                if len(aug_indices) > 0:
                    full_aug_preds[aug_indices] = aug_preds
                aligned_aug_preds = full_aug_preds[indices]
                
                consistency = (clean_preds == aligned_aug_preds).float()
                
                correctness = (clean_preds == clean_labels).float()
                
                valid_gt = (clean_labels >= 0) & (clean_labels < 17)
                
                signals = torch.stack([
                    dist_unc[valid_gt],
                    max_cos[valid_gt],
                    entropy[valid_gt],
                    margin[valid_gt],
                    feat_norm[valid_gt],
                    consistency[valid_gt]
                ], dim=1)
                
                all_signals.append(signals.cpu())
                all_correctness.append(correctness[valid_gt].cpu())
                
        X = torch.cat(all_signals, dim=0).to(device)
        y = torch.cat(all_correctness, dim=0).to(device)
        
        signal_names = ["Dirichlet Unc", "Max Cosine", "Entropy", "Margin", "Feat Norm", "Consistency"]
        
        print(f"\n--- Complementary Information Test ({ct}) ---")
        base_auroc = fit_logistic_regression(X, y)
        print(f"Full Model AUROC: {base_auroc:.4f}")
        
        print("\nDrop-One Analysis (Does signal contain unique info?):")
        for i, name in enumerate(signal_names):
            mask = torch.ones(X.shape[1], dtype=torch.bool)
            mask[i] = False
            X_drop = X[:, mask]
            drop_auroc = fit_logistic_regression(X_drop, y)
            drop_diff = base_auroc - drop_auroc
            
            status = "REDUNDANT" if drop_diff < 0.005 else "UNIQUE INFO"
            print(f"- Dropped {name:>15}: AUROC {drop_auroc:.4f} (Diff: {drop_diff:+.4f}) -> {status}")
            
if __name__ == "__main__":
    run_info_diagnostics()
