import argparse
import logging
import os
import json
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from common.laserscan import SemLaserScan, LaserScan
from dataset.kitti.parser import Parser
import unsup_main
from unsup_main import train_extractor, train_hdc, extract_metrics_from_conf_matrix, setup_logger, save_graphic
from modules.HDC_utils import UQModel
from modules.HDC_utils import set_uq_model
from torchhd import functional

NUM_CLASSES = 17
KITTI_DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
CORRUPTIONS = [
    'fog', 
    'wet_ground', 
    'snow', 
    'motion_blur', 
    'beam_missing', 
    'crosstalk', 
    'incomplete_echo', 
    'cross_sensor'
]
# Note on Severity: D3CTTA evaluates on "moderate" severity. 
# Depending on Robo3D version, this maps to severity 2 (light/moderate/heavy) or 3 (1-5 scale).
# When comparing to D3CTTA, ensure you run with the severity integer that maps to 'moderate'.
SEVERITY_MAP = {1: 'light', 2: 'moderate', 3: 'heavy', 4: 'extreme'}

CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS_KITTI_ALL = "config/labels/semantic-kitti-all.yaml"  # Standard 17 classes

def evaluate_and_adapt(model, target_dataloader, device, eval_only=False, update_method='frozen', dry_run=False, custom_update_fn=None):
    miou_history = []
    acc_history = []
    iou_per_class_history = []
    num_classes = model.num_classes
    cumulative_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    
    prev_preds_2d = None


    for batch_idx, batch_data in enumerate(tqdm(target_dataloader, desc="Adapting", leave=False)):
        if dry_run and batch_idx >= 2:
            break
            
        if dry_run and batch_idx == 0:
            print(f"\n[DEBUG] len(batch_data): {len(batch_data)}")
            if len(batch_data) > 10:
                print(f"[DEBUG] batch_data[10] shape: {batch_data[10].shape}")
        
        proj_in = batch_data[0].to(device)
        proj_labels = batch_data[2].to(device).view(-1)
        if batch_idx == 0:
            pass # debug printing removed for cleanliness
            
        proj_xyz = batch_data[10].to(device) if len(batch_data) > 10 else None
        
        if proj_in.shape[1] > 0:
            model.eval()
            with torch.no_grad():
                if type(model).__name__ == 'D3CTTA':
                    logits, _, indices, h = model(proj_in, xyz=proj_xyz)
                    predictions = torch.argmax(logits, dim=1)
                else:
                    # Get raw latent and encodings for updates
                    with torch.amp.autocast('cuda', enabled=True):
                        latent_x = model.net(proj_in, only_feat=True)
                    latent_x = latent_x.permute(0, 2, 3, 1).reshape(-1, 128)
                    
                    raw_enc, indices, _ = model.encode(proj_in)
                    norm_enc = F.normalize(raw_enc, dim=1)
                    
                    if norm_enc.dtype != model.classify.weight.dtype:
                        norm_enc = norm_enc.to(model.classify.weight.dtype)
                    
                    logits = model.classify(norm_enc)
                    predictions = torch.argmax(logits, dim=1)
                
                selected_labels = proj_labels[indices]
                mask = (selected_labels >= 0) & (selected_labels < num_classes)
                if mask.any():
                    hist = torch.bincount(
                        num_classes * selected_labels[mask] + predictions[mask], 
                        minlength=num_classes ** 2
                    ).reshape(num_classes, num_classes)
                    cumulative_confusion_matrix += hist
                
            cumulative_miou, cumulative_acc, cumulative_iou_per_class = extract_metrics_from_conf_matrix(cumulative_confusion_matrix)
            miou_history.append(cumulative_miou)
            acc_history.append(cumulative_acc)
            iou_per_class_history.append(cumulative_iou_per_class)
            
            # Adapt: Inference Update
            if not eval_only and update_method != 'frozen':
                model.eval()
                with torch.no_grad():
                    if type(model).__name__ == 'D3CTTA':
                        model.inference_update(h, predictions, proj_xyz)
                        continue
                        
                    update_lr = 0.01
                    proto_norm = F.normalize(model.classify.weight, dim=1)
                    cos_sims = F.linear(norm_enc, proto_norm)
                    max_cos_sim, pseudo_labels = torch.max(cos_sims, dim=1)
                    
                    # Soft Gating Initialization
                    # HDC cosine similarities are extremely small (e.g. ~0.05 to ~0.15). 
                    # We use a sharp temperature scaling (x 100.0) to properly stretch these into [0, 1] probability weights.
                    cos_sim_probs = F.softmax(cos_sims * 100.0, dim=1)
                    base_weights = cos_sim_probs.max(dim=1)[0]
                    update_weights = base_weights.clone()
                    
                    latent_x_valid = latent_x[indices]

                    core_method = update_method.replace('balanced_', '').replace('ledger_', '')
                    
                    if 'balanced' in update_method or update_method in ['core_method', 'variant_1', 'variant_2', 'variant_3']:
                        # Simple margin-based probabilistic drop to prevent over-updating on easy samples
                        sorted_cos_sims, _ = torch.sort(cos_sims, dim=1, descending=True)
                        margins = sorted_cos_sims[:, 0] - sorted_cos_sims[:, 1]
                        
                        # High confidence threshold (0.05 is significant in HDC cosine space)
                        margin_threshold = 0.05
                        drop_prob = 0.5
                        
                        # Detect high confidence samples and probabilistically drop them
                        random_drops = torch.rand_like(margins) < drop_prob
                        drop_mask = (margins > margin_threshold) & random_drops
                        
                        update_weights = update_weights * (~drop_mask).float()
                        
                    use_dirichlet = 'dirichlet_density' in update_method or update_method in ['core_method', 'variant_1', 'variant_2', 'variant_3']
                    use_energy = 'energy_density' in update_method or update_method in ['core_method', 'variant_1', 'variant_2', 'variant_3']
                    use_momentum = 'momentum_veto' in update_method or update_method in ['variant_1', 'variant_3']
                    use_spatial = 'orthogonal_spatial_veto' in update_method or update_method in ['variant_2', 'variant_3']
                    
                    if use_dirichlet:
                        # Softplus to map cosine sims to positive evidence
                        evidence = F.softplus(cos_sims * 100.0)
                        alpha = evidence + 1.0
                        S = torch.sum(alpha, dim=1, keepdim=True)
                        uncertainty = num_classes / S.squeeze(1) # Epistemic uncertainty
                        
                        # Soft Veto: High uncertainty scales down update weight
                        u_threshold = 0.5 
                        decay = torch.exp(-2.0 * torch.relu(uncertainty - u_threshold))
                        update_weights = update_weights * decay
                        
                    if use_energy:
                        # Free Energy = -T * logsumexp(logits / T)
                        energy = -torch.logsumexp(cos_sims * 100.0, dim=1)
                        
                        if not hasattr(model, 'ema_energy'):
                            model.ema_energy = energy.mean().item()
                            
                        if batch_idx > 0:
                            model.ema_energy = 0.9 * model.ema_energy + 0.1 * energy.mean().item()
                            
                        # Soft Veto: if energy is significantly higher than EMA (less negative)
                        energy_diff = torch.relu(energy - model.ema_energy)
                        decay = torch.exp(-0.693 * (energy_diff / (abs(model.ema_energy) * 0.1 + 1e-5)))
                        update_weights = update_weights * decay
                        
                    elif 'ledger' in update_method:
                        # Parse the margin type from the method string
                        margin_type = 'absolute'
                        budget_margin = 50
                        
                        if 'relaxed' in update_method:
                            margin_type = 'absolute'
                            budget_margin = 5000
                        elif 'dynamic' in update_method:
                            margin_type = 'proportional'
                            
                        # Full class-budgeting ledger
                        update_weights = model._consult_budget_ledger(
                            latent_x_valid, 
                            pseudo_labels, 
                            update_weights, 
                            budget_margin=budget_margin,
                            margin_type=margin_type
                        )
                        
                    if 'multi_rp' in update_method:
                        num_rp = 5
                        rp_preds = []
                        for i in range(num_rp):
                            temp_proj = model.multi_rp_projs[i]
                            temp_proto = model.multi_rp_prototypes[i]
                            temp_hv = functional.normalize(F.linear(latent_x_valid.float(), temp_proj))
                            temp_logits = F.linear(F.normalize(temp_hv.to(temp_proto.dtype), dim=1), temp_proto)
                            rp_preds.append(torch.argmax(temp_logits, dim=1))
                        rp_preds = torch.stack(rp_preds, dim=0)
                        rp_agreement = (rp_preds == pseudo_labels.unsqueeze(0)).float().mean(dim=0)
                        
                        update_weights = update_weights * rp_agreement
                        
                    if 'euclidean_density' in update_method or 'epistemic_density' in update_method or 'multi_rp_density' in update_method or 'temporal_density' in update_method:
                        pred_means = model.class_latent_means[pseudo_labels]
                        dist_to_mean = torch.norm(latent_x_valid.float() - pred_means, p=2, dim=1)
                        # Soft Veto: Exponential decay based on distance. 
                        decay = torch.exp(-0.693 * (dist_to_mean / (model.source_density_std + 1e-8)))
                        update_weights = update_weights * decay
                        
                    if 'magnitude' in update_method:
                        raw_magnitude = torch.norm(latent_x_valid.float(), p=2, dim=1)
                        mag_diff = torch.abs(raw_magnitude - model.source_mean_magnitude)
                        # Soft Veto: Exponential decay based on difference from clean mean.
                        decay = torch.exp(-0.693 * (mag_diff / (model.source_std_magnitude + 1e-8)))
                        update_weights = update_weights * decay
                        
                    if use_spatial:
                        # ViM: Orthogonal Noise Detection (Residual Subspace)
                        # Q is an orthonormal basis for the principal semantic subspace spanned by prototypes
                        Q, _ = torch.linalg.qr(model.classify.weight.T)
                        
                        # Project incoming points into the subspace
                        subspace_projection = norm_enc[indices] @ Q @ Q.T
                        proj_norm = torch.norm(subspace_projection, dim=1)
                        
                        # Track the expected magnitude of the semantic projection
                        if not hasattr(model, 'ema_proj_norm'):
                            model.ema_proj_norm = proj_norm.mean().item()
                        if batch_idx > 0:
                            model.ema_proj_norm = 0.9 * model.ema_proj_norm + 0.1 * proj_norm.mean().item()
                            
                        # Veto if the projection norm drops significantly below the moving average
                        # (Meaning the point's geometry has shifted heavily into the orthogonal null space)
                        proj_drop = torch.relu(model.ema_proj_norm - proj_norm)
                        decay = torch.exp(-10.0 * (proj_drop / (model.ema_proj_norm + 1e-5)))
                        update_weights = update_weights * decay
                        
                    if use_momentum:
                        H, W = proj_in.shape[2], proj_in.shape[3]
                        if not hasattr(model, 'ema_logits'):
                            model.ema_logits = torch.zeros((1, num_classes, H, W), device=device)
                            
                        curr_logits_full = torch.zeros((H * W, num_classes), device=device)
                        curr_logits_full[indices] = cos_sims
                        curr_logits_2d = curr_logits_full.T.reshape(1, num_classes, H, W)
                        
                        if batch_idx == 0:
                            model.ema_logits = curr_logits_2d
                        else:
                            model.ema_logits = 0.9 * model.ema_logits + 0.1 * curr_logits_2d
                            
                        ema_logits_valid = model.ema_logits.squeeze(0).reshape(num_classes, -1).T[indices]
                        
                        # Cosine distance to measure curvature/divergence from central flow
                        cos_dist = 1.0 - F.cosine_similarity(cos_sims, ema_logits_valid + 1e-8, dim=1)
                        
                        decay = torch.exp(-5.0 * cos_dist)
                        update_weights = update_weights * decay
                        
                    if 'temporal' in update_method:
                        H, W = proj_in.shape[2], proj_in.shape[3]
                        
                        curr_preds_full = torch.full((H * W,), -1, dtype=torch.long, device=device)
                        curr_preds_full[indices] = pseudo_labels
                        curr_preds_2d = curr_preds_full.reshape(1, 1, H, W).float()
                        
                        if prev_preds_2d is not None:
                            unfolded_prev = F.unfold(prev_preds_2d, kernel_size=5, padding=2).squeeze(0).long()
                            prev_neighborhoods = unfolded_prev[:, indices] 
                            valid_past = (prev_neighborhoods != -1).any(dim=0)
                            label_matched = (prev_neighborhoods == pseudo_labels.unsqueeze(0)).any(dim=0)
                            conflict = valid_past & (~label_matched)
                            # Hard filter for temporal conflict: drops weight to 0.0
                            update_weights = update_weights * (~conflict).float()
                            
                        prev_preds_2d = curr_preds_2d.clone()
                        
                    # Calculate tracking metrics
                    # We define a "veto" as any point where the uncertainty method cut the base confidence by >50%
                    veto_mask = update_weights < (0.5 * base_weights)
                    # We define a point as "fired" if it contributes more than 1% weight
                    fired_mask = update_weights > 0.01
                    
                    if not hasattr(model, '_firing_log'):
                        model._firing_log = []
                    
                    # Guard against empty tensors causing NaN
                    if len(update_weights) > 0:
                        model._firing_log.append(update_weights.mean().item())
                    else:
                        model._firing_log.append(0.0)
                    
                    if update_method != 'prototype_cosine':
                        valid_gt_mask = (proj_labels >= 0) & (proj_labels < num_classes)
                        true_errors_rejected = (veto_mask & valid_gt_mask & (pseudo_labels != proj_labels)).sum().item()
                        correct_labels_rejected = (veto_mask & valid_gt_mask & (pseudo_labels == proj_labels)).sum().item()
                        if not hasattr(model, '_veto_stats'):
                            model._veto_stats = {'true_errors_rejected': 0, 'correct_labels_rejected': 0}
                        model._veto_stats['true_errors_rejected'] += true_errors_rejected
                        model._veto_stats['correct_labels_rejected'] += correct_labels_rejected
                    
                    if fired_mask.any():
                        valid_enc = norm_enc[indices]
                        for c in range(num_classes):
                            c_mask = (pseudo_labels == c)
                            if c_mask.any():
                                c_weights = update_weights[c_mask].unsqueeze(1)
                                c_update = (valid_enc[c_mask] * c_weights).mean(dim=0)
                                
                                if c_update.norm(p=2) > 1e-6:
                                    model.classify.weight[c].data += update_lr * c_update.to(model.classify.weight.dtype)
                                    model.classify.weight[c].data = F.normalize(model.classify.weight[c].data, p=2, dim=0)
                                    
                                    if not hasattr(model, '_update_magnitude_log'):
                                        model._update_magnitude_log = []
                                    model._update_magnitude_log.append((update_lr * c_update.norm(p=2)).item())
    
    if hasattr(model, '_veto_stats') and model._veto_stats['correct_labels_rejected'] > 0:
        purity_ratio = model._veto_stats['true_errors_rejected'] / model._veto_stats['correct_labels_rejected']
        print(f"\n[Stats] Veto Purity Ratio: {purity_ratio:.2f} ({model._veto_stats['true_errors_rejected']} true errors rejected / {model._veto_stats['correct_labels_rejected']} correct labels rejected)")
        model._veto_stats = {'true_errors_rejected': 0, 'correct_labels_rejected': 0}
        
    avg_firing_rate = 0.0
    if hasattr(model, '_firing_log') and len(model._firing_log) > 0:
        avg_firing_rate = sum(model._firing_log) / len(model._firing_log)
        model._firing_log = []
        
    avg_update_magnitude = 0.0
    if hasattr(model, '_update_magnitude_log') and len(model._update_magnitude_log) > 0:
        avg_update_magnitude = sum(model._update_magnitude_log) / len(model._update_magnitude_log)
        model._update_magnitude_log = []
        
    return {"mIoU": miou_history, "Accuracy": acc_history, "IoU_per_class": iou_per_class_history, "FiringRate": avg_firing_rate, "UpdateMagnitude": avg_update_magnitude}


def pretrain_pipeline(ARCH, DATA, data_dir, pretrained_path, return_trainer=False, skip_extractor=False, resume_path=None, hdc_epochs=15, extractor_epochs=60):
    log_base = os.path.dirname(pretrained_path)
    os.makedirs(log_base, exist_ok=True)
    
    unsup_main.LOG_DIR = log_base
    unsup_main.MODEL_DIR = log_base
    unsup_main.HDC_SAVE_PATH = os.path.join(log_base, "hdc.pth")
    unsup_main.HDC_SUB_PATH = pretrained_path

    if not skip_extractor:
        ARCH["train"]["batch_size"] = 24
        print(f"Pretraining feature extractor on {data_dir}...")
        trainer = train_extractor(ARCH, DATA, epochs=extractor_epochs, data_dir=data_dir, return_trainer=True, resume_path=resume_path)
    else:
        print(f"Skipping feature extractor pretraining...")
        trainer = None
    
    ARCH["train"]["batch_size"] = 6
    print(f"Pretraining HDC density model on {data_dir} for {hdc_epochs} epochs...")
    model, _ = train_hdc(ARCH, DATA, epochs=hdc_epochs, data_dir=data_dir, return_extractor=True)
    

    
    if return_trainer:
        return model, trainer
    return model


def save_degradation_plot(save_path, title, data_dict, metric="mIoU", baseline_val=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    severities = [1, 2, 3, 4, 5]
    colors = plt.cm.tab10.colors
    
    for i, (corr, sev_dict) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        initial_vals = [sev_dict.get(s, (0, 0))[0] for s in severities]
        final_vals = [sev_dict.get(s, (0, 0))[1] for s in severities]
        
        plt.plot(severities, initial_vals, marker='x', linestyle=':', color=color, alpha=0.6, label=f'{corr} (Initial)')
        plt.plot(severities, final_vals, marker='o', linestyle='-', color=color, label=f'{corr} (Final)')
        
    if baseline_val is not None:
        plt.axhline(y=baseline_val, color='r', linestyle='--', label=f'Clean Baseline ({baseline_val:.4f})')
    
    plt.title(f"{title} - {metric} Degradation")
    plt.xlabel("Severity")
    plt.ylabel(metric)
    plt.xticks(severities)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_hdc_model(path, num_classes=NUM_CLASSES):
    print(f"Loading pretrained HDC model from {path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
    modeldir = os.path.dirname(path)

    model = set_uq_model(ARCH, modeldir, 'rp', 0, 0, num_classes, device)
    
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.to(device)
    return model

def populate_source_statistics(model, data_dir, arch_cfg, data_cfg, device):
    print(f"Populating source statistics from {data_dir}...")
    parser = Parser(root=data_dir,
                    train_sequences=data_cfg["split"]["train"],
                    valid_sequences=data_cfg["split"]["valid"],
                    test_sequences=None,
                    labels=data_cfg["labels"],
                    color_map=data_cfg.get("color_map", {}),
                    learning_map=data_cfg["learning_map"],
                    learning_map_inv=data_cfg["learning_map_inv"],
                    sensor=arch_cfg["dataset"]["sensor"],
                    max_points=arch_cfg["dataset"]["max_points"],
                    batch_size=1,
                    workers=arch_cfg["train"]["workers"],
                    gt=True,
                    shuffle_train=True) 
    
    dataloader = DataLoader(parser.trainloader.dataset, batch_size=1, shuffle=True, num_workers=4)
    model.eval()
    
    all_magnitudes = []
    num_classes = model.num_classes
    class_latent_sums = torch.zeros(num_classes, 128, device=device)
    class_latent_counts = torch.zeros(num_classes, device=device)
    
    # Collect latents for KMeans subcluster initialization
    model.class_latents_for_clustering = {c: [] for c in range(num_classes)}
    max_cluster_samples_per_class = 10000
    
    num_rp = 5
    model.multi_rp_projs = []
    model.multi_rp_prototypes = torch.zeros(num_rp, num_classes, model.hd_dim, device=device)
    for _ in range(num_rp):
        temp_proj = torch.randn(model.hd_dim, 128, device=device)
        q, _ = torch.linalg.qr(temp_proj)
        temp_proj = q * torch.sqrt(torch.tensor(model.hd_dim, dtype=torch.float32, device=device))
        model.multi_rp_projs.append(temp_proj)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Populating Source Stats")):
            if batch_idx > 500: # Limit to a subset to save time
                break
            proj_in = batch_data[0].to(device)
            proj_labels = batch_data[2].to(device).view(-1)
            
            if proj_in.shape[1] > 0:
                with torch.amp.autocast('cuda', enabled=True):
                    latent_x = model.net(proj_in, only_feat=True)
                latent_x = latent_x.permute(0, 2, 3, 1).reshape(-1, 128)
                
                _, indices, _ = model.encode(proj_in)
                selected_labels = proj_labels[indices]
                valid_mask = (selected_labels >= 0) & (selected_labels < num_classes)
                
                if not valid_mask.any():
                    continue
                    
                latent_valid = latent_x[valid_mask].float()
                labels_valid = selected_labels[valid_mask]
                
                raw_magnitude = torch.norm(latent_valid, p=2, dim=1)
                all_magnitudes.append(raw_magnitude.cpu())
                
                for c in range(num_classes):
                    c_mask = labels_valid == c
                    if c_mask.any():
                        class_latent_sums[c] += latent_valid[c_mask].sum(dim=0)
                        class_latent_counts[c] += c_mask.sum()
                        
                        # Accumulate up to max_cluster_samples_per_class for clustering
                        current_len = sum(len(x) for x in model.class_latents_for_clustering[c])
                        if current_len < max_cluster_samples_per_class:
                            c_latents = latent_valid[c_mask].cpu()
                            model.class_latents_for_clustering[c].append(c_latents)
                        
                for i in range(num_rp):
                    temp_hv = functional.normalize(F.linear(latent_valid, model.multi_rp_projs[i]))
                    for c in range(num_classes):
                        c_mask = labels_valid == c
                        if c_mask.any():
                            model.multi_rp_prototypes[i, c] += temp_hv[c_mask].sum(dim=0)
                            
    if len(all_magnitudes) > 0:
        all_magnitudes = torch.cat(all_magnitudes, dim=0)
        model.source_mean_magnitude = all_magnitudes.mean().item()
        model.source_std_magnitude = all_magnitudes.std().item()
    else:
        raise ValueError("Source statistics population failed: No valid latent features found in the first 500 frames.")
    
    counts_safe = torch.clamp(class_latent_counts, min=1).unsqueeze(1)
    model.class_latent_means = class_latent_sums / counts_safe
    
    # Initialize the subcluster budget ledger here!
    print("Initializing Subcluster Budget Ledger using KMeans...")
    model._initialize_subcluster_ledger(num_clusters=10)
    model.multi_rp_prototypes = F.normalize(model.multi_rp_prototypes, p=2, dim=2)
    
    # Pass 2: Calculate density standard deviation
    all_dists = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Populating Density Std")):
            if batch_idx > 50:
                break
            proj_in = batch_data[0].to(device)
            proj_labels = batch_data[2].to(device).view(-1)
            
            if proj_in.shape[1] > 0:
                with torch.amp.autocast('cuda', enabled=True):
                    latent_x = model.net(proj_in, only_feat=True)
                latent_x = latent_x.permute(0, 2, 3, 1).reshape(-1, 128)
                _, indices, _ = model.encode(proj_in)
                selected_labels = proj_labels[indices]
                valid_mask = (selected_labels >= 0) & (selected_labels < num_classes)
                
                if not valid_mask.any():
                    continue
                latent_valid = latent_x[valid_mask].float()
                labels_valid = selected_labels[valid_mask]
                
                pred_means = model.class_latent_means[labels_valid]
                dists = torch.norm(latent_valid - pred_means, p=2, dim=1)
                all_dists.append(dists.cpu())
                
    if len(all_dists) > 0:
        model.source_density_std = torch.cat(all_dists, dim=0).std().item()
    else:
        model.source_density_std = 1.0

def main():
    parser = argparse.ArgumentParser(description="Test Unsupervised Updates on KITTI-C")
    parser.add_argument('--pretrain', action='store_true', help='Run pretraining on SemanticKITTI before evaluating')
    parser.add_argument('--chunked', action='store_true', help='Use D3CTTA chunked protocol: continuous adaptation across disjoint 1/7th splits instead of full independent sequences.')
    parser.add_argument('--reset_per_corruption', action='store_true', help='Reset the model to the clean pretrained weights before adapting on each corruption (even when using chunks).')
    parser.add_argument('--skip_extractor', action='store_true', help='Skip feature extractor pretraining and only retrain the HDC model')
    parser.add_argument('--pretrained_path', type=str, default='logs/kitti_pretrain/hdc_sub.pth', help='Path to load pretrained model')
    parser.add_argument('--log_dir', type=str, default='logs/kitti_c_test', help='Directory to save logs and graphics')
    parser.add_argument('--method', type=str, default='frozen', help='Method to test.')
    parser.add_argument('--dry_run', action='store_true', help='Run only 2 batches per condition to quickly verify no crashes will occur.')
    parser.add_argument('--continue_pretrain', action='store_true', help='Resume pretraining from the existing pretrained_path')
    parser.add_argument('--continue', dest='continue_epochs', type=int, default=0, help='Continue feature extractor training for this many epochs, reinitialize HDC, and perform adaptation')
    parser.add_argument('--extractor_epochs', type=int, default=60, help='Number of epochs to train the feature extractor')
    parser.add_argument('--hdc_epochs', type=int, default=15, help='Number of epochs to train the HDC density model')
    parser.add_argument('--severity', type=int, default=3, help='Severity level for corruptions')
    parser.add_argument('--kitti_dir', type=str, default='/mnt/alpha/jmfleming/KITTI', help='Path to SemanticKITTI dataset for pretraining')
    parser.add_argument('--kittic_dir', type=str, default='/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C', help='Path to real SemanticKITTI-C dataset')
    parser.add_argument('--corruptions', type=str, default=None, help='Comma separated list of corruptions to test. Defaults to all 8.')
    args = parser.parse_args()

    if args.continue_epochs > 0:
        args.pretrain = True
        args.continue_pretrain = True
        args.extractor_epochs = args.continue_epochs

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, 'kitti_c.log'))

    try:
        ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
        DATA = yaml.safe_load(open(CONFIG_LABELS_KITTI_ALL, 'r'))
    except Exception as e:
        logger.error(f"Error loading configs: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.pretrain:
        logger.info(f"Starting Pretraining on SemanticKITTI at {args.kitti_dir}...")
        resume_dir = os.path.dirname(args.pretrained_path) if args.continue_pretrain else None
            
        model, trainer = pretrain_pipeline(
            ARCH, DATA, data_dir=args.kitti_dir, 
            pretrained_path=args.pretrained_path, return_trainer=True, 
            skip_extractor=args.skip_extractor, resume_path=resume_dir, 
            hdc_epochs=args.hdc_epochs, extractor_epochs=args.extractor_epochs
        )
        
        if trainer is not None:
            opt_path = os.path.join(os.path.dirname(args.pretrained_path), 'feature_optimizer.pth')
            torch.save(trainer.optimizer.state_dict(), opt_path)
            logger.info(f"Successfully pretrained model on SemanticKITTI. Optimizer state saved to {opt_path}")
            
    sev = args.severity
    methods_to_run = [
        'balanced_dirichlet_density',
        'balanced_energy_density',
        'balanced_orthogonal_spatial_veto'
    ] if args.method == 'all' else [args.method]
    
    global_results_path = os.path.join(args.log_dir, 'global_results.json')
    if os.path.exists(global_results_path):
        with open(global_results_path, 'r') as f:
            global_results = json.load(f)
        # Ensure the dicts for the current methods exist in case they were never run
        for m in methods_to_run:
            if m not in global_results['mIoU']:
                global_results['mIoU'][m] = {c: {} for c in CORRUPTIONS}
            if m not in global_results['Accuracy']:
                global_results['Accuracy'][m] = {c: {} for c in CORRUPTIONS}
    else:
        global_results = {
            'mIoU': {m: {c: {} for c in CORRUPTIONS} for m in methods_to_run},
            'Accuracy': {m: {c: {} for c in CORRUPTIONS} for m in methods_to_run},
        }
    
    shared_init_metrics = {}
    
    # Load dataset once and partition it to find chunks
    # Note on Protocol: D3CTTA divides the valid set into 7 disjoint chunks (1 per corruption).
    # This evaluates each corruption on 1/7 of the validation set (e.g., ~581 frames) instead 
    # of the full set. We are preserving this behavior to identically match their protocol. 
    # Per-domain metrics will be noisier on 400 frames, so do not directly compare these 
    # chunked metrics to full-set benchmarks.
    logger.info("Initializing baseline dataset to calculate chunk sizes...")
    parser_obj = Parser(root=KITTI_DATA_DIR,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=None,
                    labels=DATA["labels"],
                    color_map=DATA.get("color_map", {}),
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=1,
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=False)
    
    target_dataset = parser_obj.validloader.dataset
    total_len = len(target_dataset)
    chunk_size = total_len // len(CORRUPTIONS)
    
    indices = list(range(total_len))
    chunks = []
    for i in range(len(CORRUPTIONS)):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < len(CORRUPTIONS) - 1 else total_len
        chunks.append(indices[start_idx:end_idx])

    if any('epistemic' in m for m in methods_to_run) or args.method == 'all':
        base_model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES)
        populate_source_statistics(base_model, args.kitti_dir, ARCH, DATA, device)
        
        source_stats_cache = {
            'multi_rp_projs': base_model.multi_rp_projs,
            'multi_rp_prototypes': base_model.multi_rp_prototypes,
            'class_latent_means': base_model.class_latent_means,
            'source_mean_magnitude': getattr(base_model, 'source_mean_magnitude', None),
            'source_std_magnitude': getattr(base_model, 'source_std_magnitude', None),
            'source_density_std': getattr(base_model, 'source_density_std', None),
            'num_clusters': getattr(base_model, 'num_clusters', None),
            'subcluster_centroids': getattr(base_model, 'subcluster_centroids', None),
            'subcluster_update_counts': getattr(base_model, 'subcluster_update_counts', None),
            'actual_k_per_class': getattr(base_model, 'actual_k_per_class', None)
        }
    else:
        source_stats_cache = None

    clean_state_dict = torch.load(args.pretrained_path, map_location=device)
    
    logger.info("Pre-loading corruption datasets...")
    corruption_datasets = {}
    for ctype in CORRUPTIONS:
        sev_str = SEVERITY_MAP.get(sev, 'moderate')
        corruption_root = os.path.join(args.kittic_dir, ctype, sev_str)
        seq_dir = os.path.join(corruption_root, "sequences")
        if not os.path.exists(seq_dir):
            logger.info(f"Directory structure doesn't match standard KITTI. Creating 'sequences/08' symlink in {corruption_root}...")
            os.makedirs(seq_dir, exist_ok=True)
            os.symlink("..", os.path.join(seq_dir, "08"))
        try:
            parser_obj = Parser(root=corruption_root,
                                train_sequences=DATA["split"]["valid"],
                                valid_sequences=DATA["split"]["valid"],
                                test_sequences=None,
                                labels=DATA["labels"],
                                color_map=DATA.get("color_map", {}),
                                learning_map=DATA["learning_map"],
                                learning_map_inv=DATA["learning_map_inv"],
                                sensor=ARCH["dataset"]["sensor"],
                                max_points=ARCH["dataset"]["max_points"],
                                batch_size=1,
                                workers=ARCH["train"]["workers"],
                                gt=True,
                                shuffle_train=False)
            corruption_datasets[ctype] = parser_obj.validloader.dataset
        except Exception as e:
            logger.error(f"Failed to load KITTI-C corruption dataset at {corruption_root}: {e}")

    # Initialize the model exactly ONCE to be shared
    model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES)
    if source_stats_cache is not None:
        model.multi_rp_projs = source_stats_cache['multi_rp_projs']
        model.multi_rp_prototypes = source_stats_cache['multi_rp_prototypes']
        model.class_latent_means = source_stats_cache['class_latent_means']
        model.source_mean_magnitude = source_stats_cache['source_mean_magnitude']
        model.source_std_magnitude = source_stats_cache['source_std_magnitude']
        model.source_density_std = source_stats_cache['source_density_std']
        if source_stats_cache['num_clusters'] is not None:
            model.num_clusters = source_stats_cache['num_clusters']
            model.subcluster_centroids = source_stats_cache['subcluster_centroids']
            model.subcluster_update_counts = source_stats_cache['subcluster_update_counts']
            model.actual_k_per_class = source_stats_cache['actual_k_per_class']

    for current_method in methods_to_run:
        logger.info(f"=========================================")
        logger.info(f"Starting Evaluation for Method: {current_method}")
        logger.info(f"=========================================")
        
        active_corruptions = CORRUPTIONS
        if args.corruptions:
            active_corruptions = [c.strip() for c in args.corruptions.split(',')]

        results_miou = {c: {} for c in active_corruptions}
        results_acc = {c: {} for c in active_corruptions}

        # Reset model at the start of each new method loop
        model.load_state_dict(clean_state_dict, strict=False)
        if hasattr(model, 'subcluster_update_counts') and model.subcluster_update_counts is not None:
            model.subcluster_update_counts.zero_()
            
        if current_method == 'd3ctta':
            from modules.D3CTTA import D3CTTA
            eval_model = D3CTTA(model.net, num_classes=NUM_CLASSES, feature_dim=128).to(device)
        else:
            eval_model = model

        for i, ctype in enumerate(active_corruptions):
            if args.reset_per_corruption and args.chunked:
                logger.info("Resetting model to clean pretrained weights for this corruption.")
                model.load_state_dict(clean_state_dict, strict=False)
                if hasattr(model, 'subcluster_update_counts') and model.subcluster_update_counts is not None:
                    model.subcluster_update_counts.zero_()
                
            logger.info(f"Testing {ctype} severity {sev} (Chunk {i+1}/{len(active_corruptions)})")
            
            if ctype not in corruption_datasets:
                continue
                
            full_corruption_dataset = corruption_datasets[ctype]
            
            # Prevent silent misalignment bugs by ensuring corrupted frame count matches baseline clean chunk length
            assert len(full_corruption_dataset) == total_len, (
                f"Length mismatch: Clean baseline length is {total_len}, "
                f"but {ctype}-{sev_str} length is {len(full_corruption_dataset)}. "
                f"Chunks will misalign."
            )
            
            if not args.chunked:
                # Standard protocol: full sequence, independent adaptation
                chunk_dataset = full_corruption_dataset
                # Reset model before each corruption
                model.load_state_dict(clean_state_dict, strict=False)
            else:
                # D3CTTA protocol: chunks, continuous adaptation
                chunk_dataset = torch.utils.data.Subset(full_corruption_dataset, chunks[i])
            
            target_dataloader = DataLoader(chunk_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
            
            try:
                if not args.chunked or args.reset_per_corruption:
                    # Pass 1: True Initial (Frozen on chunk)
                    if (ctype, sev) not in shared_init_metrics:
                        logger.info("  -> Pass 1: Computing True Initial metrics (Frozen)")
                        init_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=True, dry_run=args.dry_run)
                        shared_init_metrics[(ctype, sev)] = init_metrics
                    else:
                        logger.info("  -> Pass 1: Reusing cached True Initial metrics (Frozen)")
                        init_metrics = shared_init_metrics[(ctype, sev)]
                    
                    # Pass 2: Adapt (only if method is not frozen)
                    if current_method != 'frozen':
                        logger.info("  -> Pass 2: Adapting model weights")
                        adapt_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=False, update_method=current_method, dry_run=args.dry_run)
                    else:
                        adapt_metrics = init_metrics
                        
                    # Pass 3: True Final (Frozen on chunk using adapted weights)
                    logger.info("  -> Pass 3: Computing True Final metrics (Frozen)")
                    final_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=True, dry_run=args.dry_run)
                    
                    # We only care about the absolute end of the frozen evaluations for the sequence
                    metrics = adapt_metrics  # Just for the trajectory json
                    if len(init_metrics["mIoU"]) > 0:
                        initial_miou = init_metrics["mIoU"][-1]
                        final_miou = final_metrics["mIoU"][-1]
                        initial_acc = init_metrics["Accuracy"][-1]
                        final_acc = final_metrics["Accuracy"][-1]
                    else:
                        initial_miou = final_miou = initial_acc = final_acc = 0.0
                        
                    firing_rate_str = ""
                    if "FiringRate" in adapt_metrics:
                        firing_rate_str = f", FiringRate={adapt_metrics['FiringRate']*100:.2f}%"
                        if "UpdateMagnitude" in adapt_metrics:
                            firing_rate_str += f", UpdateMag={adapt_metrics['UpdateMagnitude']:.4f}"
                else:
                    # Original single-pass continuous evaluation
                    metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=(current_method == 'frozen'), update_method=current_method, dry_run=args.dry_run)
                    if len(metrics["mIoU"]) > 0:
                        initial_miou = metrics["mIoU"][0]
                        final_miou = metrics["mIoU"][-1]
                        initial_acc = metrics["Accuracy"][0]
                        final_acc = metrics["Accuracy"][-1]
                    else:
                        initial_miou = final_miou = initial_acc = final_acc = 0.0
                        
                    firing_rate_str = ""
                    if "FiringRate" in metrics:
                        firing_rate_str = f", FiringRate={metrics['FiringRate']*100:.2f}%"
                        if "UpdateMagnitude" in metrics:
                            firing_rate_str += f", UpdateMag={metrics['UpdateMagnitude']:.4f}"
            except Exception as e:
                import traceback
                logger.error(f"FATAL ERROR during {ctype} sev {sev} ({current_method}): {e}")
                logger.error(traceback.format_exc())
                logger.info("Skipping to next cell to protect the overnight run...")
                continue
            
            if len(metrics["mIoU"]) > 0:
                results_miou[ctype][sev] = (initial_miou, final_miou)
                results_acc[ctype][sev] = (initial_acc, final_acc)
                
                global_results['mIoU'][current_method][ctype][sev] = (initial_miou, final_miou)
                global_results['Accuracy'][current_method][ctype][sev] = (initial_acc, final_acc)
                
                logger.info(f"Result for {ctype}-{sev}: Initial mIoU={initial_miou:.4f} -> Final={final_miou:.4f}, Initial Acc={initial_acc:.4f} -> Final={final_acc:.4f}{firing_rate_str}")
                suffix = f"_{current_method}"
                
                traj_json_path = os.path.join(args.log_dir, f'traj_{ctype}_{sev}{suffix}.json')
                with open(traj_json_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                    
                save_graphic(os.path.join(args.log_dir, f'traj_{ctype}_{sev}{suffix}.png'), f'{ctype} Sev {sev}', metrics)
                
                with open(os.path.join(args.log_dir, f'results{suffix}.json'), 'w') as f:
                    json.dump({'mIoU': results_miou, 'Accuracy': results_acc}, f, indent=4)
                    
                with open(os.path.join(args.log_dir, 'global_results.json'), 'w') as f:
                    json.dump(global_results, f, indent=4)
            else:
                logger.info(f"No valid frames evaluated for {ctype}-{sev}")

        suffix = f"_{current_method}"
        save_degradation_plot(os.path.join(args.log_dir, f'degradation_miou{suffix}.png'), 'KITTI-C', results_miou, metric='mIoU', baseline_val=None)
        save_degradation_plot(os.path.join(args.log_dir, f'degradation_acc{suffix}.png'), 'KITTI-C', results_acc, metric='Accuracy', baseline_val=None)

if __name__ == "__main__":
    main()
