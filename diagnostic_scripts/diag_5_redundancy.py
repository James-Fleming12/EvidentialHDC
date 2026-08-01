import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import yaml
from dataset.kitti.parser import Parser
from modules.HDC_utils import set_uq_model
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

@torch.no_grad()
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Loading architecture and model...")
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    
    model = set_uq_model(ARCH, 'logs/kitti_pretrain', 'rp', 0, 0, 17, device)
    model.load_state_dict(torch.load('logs/kitti_pretrain/hdc_sub.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()

    print("Loading Fog dataset...")
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
    dataloader = fog_parser.get_valid_set()

    print("Extracting signals from Fog dataset...")
    
    all_max_cos = []
    all_margin = []
    all_entropy = []
    all_density = []
    all_labels = [] # 1 if correct, 0 if hallucination
    
    for i, batch_data in enumerate(dataloader):
        if i >= 5: # Sample 5 frames for speed
            break
            
        proj_in = batch_data[0].to(device)
        proj_mask = batch_data[1].to(device) if len(batch_data) > 1 else None
        proj_labels = batch_data[2].to(device)
        
        valid_mask = (proj_mask.reshape(-1) > 0)
        true_valid = proj_labels.reshape(-1)[valid_mask]
        
        # Get embeddings
        raw_enc = model.encode(proj_in)[0].reshape(-1, 10000)[valid_mask]
        norm_enc = torch.nn.functional.normalize(raw_enc, dim=1).to(torch.float32)
        
        # Get predictions
        logits = model.get_predictions(norm_enc)
        
        # Signals
        topk_sims, topk_idx = logits.topk(k=2, dim=1)
        max_cos = topk_sims[:, 0]
        margin = topk_sims[:, 0] - topk_sims[:, 1]
        
        # Entropy
        probs = torch.softmax(logits * 15.0, dim=1) # Scaled for HDC
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Density (Internal 10-NN similarity)
        # Randomly subsample 2000 points to act as local memory bank for density
        if norm_enc.size(0) > 2000:
            bank_idx = torch.randperm(norm_enc.size(0))[:2000]
            bank = norm_enc[bank_idx]
        else:
            bank = norm_enc
            
        sim_matrix = torch.mm(norm_enc, bank.t())
        density = sim_matrix.topk(k=10, dim=1)[0].mean(dim=1)
        
        pseudo_valid = topk_idx[:, 0]
        is_correct = (pseudo_valid == true_valid).float()
        
        all_max_cos.append(max_cos.cpu())
        all_margin.append(margin.cpu())
        all_entropy.append(entropy.cpu())
        all_density.append(density.cpu())
        all_labels.append(is_correct.cpu())

    features = torch.stack([
        torch.cat(all_max_cos, dim=0),
        torch.cat(all_margin, dim=0),
        torch.cat(all_entropy, dim=0),
        torch.cat(all_density, dim=0)
    ], dim=1).numpy()
    
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # Subsample for correlation/training
    if features.shape[0] > 10000:
        idx = np.random.choice(features.shape[0], 10000, replace=False)
        features = features[idx]
        labels = labels[idx]

    feature_names = ["Max Cosine", "Margin", "Entropy", "Density (k-NN)"]

    print("\n=== Pairwise Correlations (Spearman) ===")
    corr_matrix, _ = scipy.stats.spearmanr(features)
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            print(f"{feature_names[i]} vs {feature_names[j]}: {corr_matrix[i, j]:.4f}")

    print("\n=== Random Forest Probe Ablation ===")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    baseline_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"Full Features AUROC: {baseline_auc:.4f}")
    
    print("\nLeave-One-Feature-Out AUROC Drops:")
    for i, name in enumerate(feature_names):
        X_train_ablated = np.delete(X_train, i, axis=1)
        X_test_ablated = np.delete(X_test, i, axis=1)
        
        rf_ablated = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        rf_ablated.fit(X_train_ablated, y_train)
        auc = roc_auc_score(y_test, rf_ablated.predict_proba(X_test_ablated)[:, 1])
        drop = baseline_auc - auc
        
        print(f"Removed {name}: AUROC = {auc:.4f} (Drop = {drop:+.4f})")
        if drop < 0.005:
            print(f"  -> {name} is entirely REDUNDANT.")
        else:
            print(f"  -> {name} contains UNIQUE information.")

if __name__ == '__main__':
    main()
