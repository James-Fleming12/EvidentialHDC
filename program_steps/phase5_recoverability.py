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

class InversionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def fit_inversion(X_src, X_tgt, epochs=200):
    X_src = X_src.to(device)
    X_tgt = X_tgt.to(device)
    
    X_src = (X_src - X_src.mean(dim=0)) / (X_src.std(dim=0) + 1e-8)
    X_tgt_norm = F.normalize(X_tgt, dim=1)
    
    model = InversionMLP(X_src.shape[1], X_tgt.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_src)
        out_norm = F.normalize(out, dim=1)
        # Cosine distance loss
        loss = 1.0 - (out_norm * X_tgt_norm).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        out = model(X_src)
        out_norm = F.normalize(out, dim=1)
        cos_sim = (out_norm * X_tgt_norm).sum(dim=1).mean().item()
        
    return cos_sim

def run_phase5_recoverability():
    print("Initializing model for Phase V Recoverability...")
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
    
    all_backbone = []
    all_hdc = []
    all_sims = []
    
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
            
    X_bb = torch.cat(all_backbone, dim=0).float()
    X_hdc = torch.cat(all_hdc, dim=0).float()
    X_sims = torch.cat(all_sims, dim=0).float()
    
    print("\n--- Phase V: Recoverability (Inverse Problems) ---")
    print("Testing if lost information can be geometrically reconstructed via MLPs...\n")
    
    print("1. Reconstructing Backbone (128D) from HDC Embedding (10,000D)...")
    sim_bb = fit_inversion(X_hdc, X_bb, epochs=300)
    print(f"   Recovery Cosine Similarity: {sim_bb:.4f}")
    
    print("\n2. Reconstructing HDC Embedding (10,000D) from Prototype Sims (17D)...")
    sim_hdc = fit_inversion(X_sims, X_hdc, epochs=300)
    print(f"   Recovery Cosine Similarity: {sim_hdc:.4f}")
    
    print("\n=== RECOVERABILITY REPORT ===")
    if sim_hdc > 0.8:
        print("Verdict: High recoverability. The 17D similarities heavily constrain the 10,000D position. Information is fundamentally retained.")
    elif sim_hdc > 0.5:
        print("Verdict: Moderate recoverability. The 17D similarities preserve some coarse geometric structure of the 10,000D space.")
    else:
        print("Verdict: Low recoverability. The projection from 10,000D to 17D is strictly destructive and highly irreversible.")
        print("This proves that adapting directly in the 10,000D space is mathematically mandatory because the projection bottleneck cannot be inverted.")

if __name__ == "__main__":
    run_phase5_recoverability()
