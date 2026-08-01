import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import yaml
from dataset.kitti.parser import Parser
from modules.HDC_utils import set_uq_model
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class HDCDenoiser(nn.Module):
    def __init__(self, hd_dim=10000, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hd_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, hd_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

@torch.no_grad()
def extract_features(dataloader, model, num_frames=10, device='cuda:0'):
    all_enc = []
    all_true = []
    all_pseudo = []
    
    for i, batch_data in enumerate(dataloader):
        if i >= num_frames: break
        
        proj_in = batch_data[0].to(device)
        proj_mask = batch_data[1].to(device) if len(batch_data) > 1 else None
        proj_labels = batch_data[2].to(device)
        
        valid_mask = (proj_mask.reshape(-1) > 0)
        true_valid = proj_labels.reshape(-1)[valid_mask]
        
        raw_enc = model.encode(proj_in)[0].reshape(-1, 10000)[valid_mask]
        norm_enc = torch.nn.functional.normalize(raw_enc, dim=1).to(torch.float32)
        
        logits = model.get_predictions(norm_enc)
        pseudo_valid = logits.argmax(dim=1)
        
        all_enc.append(norm_enc.cpu())
        all_true.append(true_valid.cpu())
        all_pseudo.append(pseudo_valid.cpu())
        
    return torch.cat(all_enc, dim=0), torch.cat(all_true, dim=0), torch.cat(all_pseudo, dim=0)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Loading architecture and model...")
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    
    model = set_uq_model(ARCH, 'logs/kitti_pretrain', 'rp', 0, 0, 17, device)
    model.load_state_dict(torch.load('logs/kitti_pretrain/hdc_sub.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()

    print("Loading Baseline (Clean) dataset...")
    clean_parser = Parser(
        root="/mnt/alpha/jmfleming/KITTI",
        train_sequences=["08"],
        valid_sequences=["08"],
        test_sequences=["08"],
        labels=yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["labels"],
        color_map={},
        learning_map=yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["learning_map"],
        learning_map_inv=yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        batch_size=1,
        workers=4,
        gt=True,
        shuffle_train=False
    )
    
    print("Extracting source bank (Clean Data)...")
    clean_enc, _, _ = extract_features(clean_parser.get_valid_set(), model, num_frames=5, device=device)
    
    # Subsample 20,000 points to act as our training source bank
    idx = torch.randperm(clean_enc.size(0))[:20000]
    source_bank = clean_enc[idx].to(device)
    
    print("Training HDC Manifold Denoiser (Autoencoder) on Source Bank...")
    denoiser = HDCDenoiser(hd_dim=10000, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    denoiser.train()
    batch_size = 1000
    for epoch in range(15): # 15 epochs should be enough
        epoch_loss = 0
        for i in range(0, source_bank.size(0), batch_size):
            batch = source_bank[i:i+batch_size]
            optimizer.zero_grad()
            recon = denoiser(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    print(f"Final Source Reconstruction Loss: {epoch_loss / (source_bank.size(0) // batch_size):.6f}")
    
    print("\nLoading Fog dataset...")
    fog_parser = Parser(
        root="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/fog/heavy",
        train_sequences=["08"],
        valid_sequences=["08"],
        test_sequences=["08"],
        labels=yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["labels"],
        color_map={},
        learning_map=yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["learning_map"],
        learning_map_inv=yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        batch_size=1,
        workers=4,
        gt=True,
        shuffle_train=False
    )
    
    print("Extracting Fog features...")
    fog_enc, fog_true, fog_pseudo = extract_features(fog_parser.get_valid_set(), model, num_frames=5, device=device)
    
    # Evaluate reconstruction error on Fog
    denoiser.eval()
    with torch.no_grad():
        all_recon_errors = []
        batch_size = 1000
        fog_enc = fog_enc.to(device)
        for i in range(0, fog_enc.size(0), batch_size):
            batch = fog_enc[i:i+batch_size]
            recon = denoiser(batch)
            # Use cosine distance as error metric (since HDC is normalized)
            sim = torch.cosine_similarity(batch, recon, dim=1)
            error = 1.0 - sim
            all_recon_errors.append(error.cpu())
            
        recon_error = torch.cat(all_recon_errors, dim=0)
        
    is_correct = (fog_pseudo == fog_true).numpy()
    is_halluc = ~is_correct
    
    print("\n=== Autoencoder Reconstruction Error (Manifold Distance) ===")
    print(f"Clean (Source) Mean Error: {epoch_loss / (source_bank.size(0) // batch_size):.4f}")
    print(f"Fog Correct Mean Error   : {recon_error[is_correct].mean():.4f}")
    print(f"Fog Halluc. Mean Error   : {recon_error[is_halluc].mean():.4f}")
    
    # Calculate AUROC to separate hallucination from correct using reconstruction error
    # We want higher error = hallucination (1), lower error = correct (0)
    auroc = roc_auc_score(is_halluc, recon_error.numpy())
    print(f"\nAUROC for Hallucination Separation: {auroc:.4f}")
    
    if auroc > 0.90:
        print("Verdict: MASSIVE SUCCESS. The Manifold Denoiser easily identifies hallucinations as out-of-distribution!")
    else:
        print("Verdict: FAILURE. The Autoencoder still reconstructs fog efficiently. The fog cluster lies ON the learned clean manifold.")

if __name__ == '__main__':
    main()
