import os
import torch
import numpy as np
import yaml
from dataset.kitti.parser import Parser
from modules.HDC_utils import set_uq_model

@torch.no_grad()
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Loading architecture and model...")
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA = yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))
    
    # Load model
    model = set_uq_model(method='rp', arch=ARCH, data=DATA, save_dir='logs/kitti_pretrain', device=device)
    model.eval()

    # Load small subset of data (just sequence 08, first 5 frames)
    # Actually just taking the baseline dataloader
    print("Loading baseline dataset...")
    baseline_parser = Parser(
        root="/mnt/alpha/jmfleming/KITTI/dataset",
        train_sequences=["08"],
        valid_sequences=["08"],
        test_sequences=["08"],
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        batch_size=1,
        workers=4,
        gt=True,
        shuffle_train=False
    )
    dataloader = baseline_parser.get_valid_set()

    print("Extracting features from clean dataset...")
    features_clean, pseudo_clean, true_clean = extract_features(model, dataloader, num_frames=5, device=device)

    # Load Fog dataset
    print("Loading Fog dataset...")
    fog_parser = Parser(
        root="/mnt/alpha/jmfleming/KITTI/kitti_c/fog/severe",
        train_sequences=["08"],
        valid_sequences=["08"],
        test_sequences=["08"],
        labels=DATA["labels"],
        color_map=DATA["color_map"],
        learning_map=DATA["learning_map"],
        learning_map_inv=DATA["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        batch_size=1,
        workers=4,
        gt=True,
        shuffle_train=False
    )
    dataloader_fog = fog_parser.get_valid_set()
    features_fog, pseudo_fog, true_fog = extract_features(model, dataloader_fog, num_frames=5, device=device)

    print("Computing Effective Rank for Clean Data...")
    r_eff_clean_correct, r_eff_clean_halluc = compute_local_rank(features_clean, pseudo_clean, true_clean)
    
    print("Computing Effective Rank for Fog Data...")
    r_eff_fog_correct, r_eff_fog_halluc = compute_local_rank(features_fog, pseudo_fog, true_fog)

    print("\n=== Local Effective Rank Diagnostics ===")
    print(f"Clean Correct       : mean={np.mean(r_eff_clean_correct):.2f}, median={np.median(r_eff_clean_correct):.2f}")
    if len(r_eff_clean_halluc) > 0:
        print(f"Clean Hallucination : mean={np.mean(r_eff_clean_halluc):.2f}, median={np.median(r_eff_clean_halluc):.2f}")
    
    print(f"Fog Correct         : mean={np.mean(r_eff_fog_correct):.2f}, median={np.median(r_eff_fog_correct):.2f}")
    if len(r_eff_fog_halluc) > 0:
        print(f"Fog Hallucination   : mean={np.mean(r_eff_fog_halluc):.2f}, median={np.median(r_eff_fog_halluc):.2f}")


def extract_features(model, dataloader, num_frames, device):
    all_features = []
    all_pseudo = []
    all_true = []
    
    for i, (proj_in, proj_mask, _, _, path_seq, path_name, _, _, proj_labels, _, _, _, _, _, _) in enumerate(dataloader):
        if i >= num_frames:
            break
            
        proj_in = proj_in.to(device)
        proj_mask = proj_mask.to(device)
        proj_labels = proj_labels.to(device)
        
        with torch.amp.autocast('cuda', enabled=True):
            latent_x = model.net(proj_in, only_feat=True)
            
        latent_x = latent_x.permute(0, 2, 3, 1).reshape(-1, 128)
        proj_mask = proj_mask.reshape(-1)
        proj_labels = proj_labels.reshape(-1)
        
        valid_mask = proj_mask > 0
        latent_valid = latent_x[valid_mask]
        true_valid = proj_labels[valid_mask]
        
        # Get pseudo labels
        raw_enc = model.encode(proj_in)[0].reshape(-1, 10000)[valid_mask]
        norm_enc = torch.nn.functional.normalize(raw_enc, dim=1).to(torch.float32)
        logits = model.get_predictions(norm_enc)
        pseudo_valid = logits.argmax(dim=1)
        
        all_features.append(latent_valid.cpu())
        all_pseudo.append(pseudo_valid.cpu())
        all_true.append(true_valid.cpu())
        
    return torch.cat(all_features, dim=0), torch.cat(all_pseudo, dim=0), torch.cat(all_true, dim=0)


def compute_local_rank(features, pseudo, true, k=50, num_samples=2000):
    # Subsample for speed
    if features.size(0) > num_samples:
        indices = torch.randperm(features.size(0))[:num_samples]
        feats_sub = features[indices]
        pseudo_sub = pseudo[indices]
        true_sub = true[indices]
    else:
        feats_sub = features
        pseudo_sub = pseudo
        true_sub = true

    # Calculate pairwise distances (L2)
    # feats_sub is (N, 128), features is (M, 128)
    # We find k-NN from features to feats_sub
    dist = torch.cdist(feats_sub.float(), features.float())
    _, topk_idx = dist.topk(k=k, dim=1, largest=False)
    
    r_eff_correct = []
    r_eff_halluc = []
    
    for i in range(feats_sub.size(0)):
        neighborhood = features[topk_idx[i]] # (k, 128)
        
        # Center the neighborhood
        neighborhood = neighborhood - neighborhood.mean(dim=0, keepdim=True)
        
        # SVD
        _, S, _ = torch.svd(neighborhood)
        
        # Effective Rank
        r_eff = (S.sum() ** 2) / (S ** 2).sum()
        r_eff = r_eff.item()
        
        is_correct = pseudo_sub[i] == true_sub[i]
        
        if is_correct:
            r_eff_correct.append(r_eff)
        else:
            r_eff_halluc.append(r_eff)
            
    return r_eff_correct, r_eff_halluc

if __name__ == '__main__':
    main()
