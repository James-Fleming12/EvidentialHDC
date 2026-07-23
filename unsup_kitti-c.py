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

def evaluate_and_adapt(model, target_dataloader, device, eval_only=False, update_method='frozen', dry_run=False, custom_update_fn=None, test_1b='none', test_1c=1.0, reproduce_bug=False):
    miou_history = []
    head_miou_history = []
    mid_miou_history = []
    tail_miou_history = []
    acc_history = []
    iou_per_class_history = []
    num_classes = model.num_classes
    cumulative_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    
    prev_preds_2d = None

    # Test 1c: Partial Calibration (Shrinkage)
    if test_1c < 1.0:
        global_mu = model.source_mu_cos.mean()
        global_sigma = model.source_sigma_cos.mean()
        active_mu_cos = test_1c * model.source_mu_cos + (1 - test_1c) * global_mu
        active_sigma_cos = test_1c * model.source_sigma_cos + (1 - test_1c) * global_sigma
    else:
        active_mu_cos = model.source_mu_cos
        active_sigma_cos = model.source_sigma_cos

    if not eval_only and hasattr(model, 'classify'):
        import logging
        logger = logging.getLogger("EvalAdapt")
        norms = {c: round(model.classify.weight[c].norm().item(), 4) for c in range(17)}
        logger.info(f"\n[Stats] Initial Prototype Norms: {norms}")

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
                
            cumulative_miou, head_miou, mid_miou, tail_miou, cumulative_acc, cumulative_iou_per_class = extract_metrics_from_conf_matrix(cumulative_confusion_matrix)
            miou_history.append(cumulative_miou)
            head_miou_history.append(head_miou)
            mid_miou_history.append(mid_miou)
            tail_miou_history.append(tail_miou)
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

                    if not hasattr(model, 'class_freq_ema'):
                        model.class_freq_ema = torch.ones(num_classes, device=device) / num_classes
                    
                    beta = 0.99
                    batch_freq = torch.bincount(pseudo_labels, minlength=num_classes).float() / max(1, pseudo_labels.size(0))
                    model.class_freq_ema = beta * model.class_freq_ema + (1 - beta) * batch_freq
                    
                    # Hard lower bound to prevent network freeze when rare classes drop to 0 in a batch size of 1
                    min_freq = torch.clamp(model.class_freq_ema.min(), min=0.01)
                    f_y = torch.clamp(model.class_freq_ema[pseudo_labels], min=0.01)
                    
                    if update_method == 'evidential_hdc_tta':
                        # 1. Balanced Updates: Inverse Frequency Soft Weighting (gamma = 0.1)
                        gamma = 0.1
                        balance_weights = (min_freq / f_y) ** gamma
                        update_weights = update_weights * balance_weights
                        
                        # 2. Network Uncertainty (128D Manifold Anchor)
                        pred_means = model.class_latent_means[pseudo_labels]
                        dist_sq = torch.norm(latent_x_valid.float() - pred_means, p=2, dim=1) ** 2
                        variance = 2.0 * (model.source_density_std[pseudo_labels] ** 2) + 1e-8
                        tier1_decay = torch.exp(-dist_sq / variance)
                        update_weights = update_weights * tier1_decay
                        
                        # 3. Hypervector Uncertainty: Z-Score Calibrated Dirichlet Evidence
                        z_c = (cos_sims - active_mu_cos) / (active_sigma_cos + 1e-8)
                        gamma_sharpness = 5.0
                        evidence = F.softplus(gamma_sharpness * z_c)
                        alpha = evidence + 1.0
                        S = torch.sum(alpha, dim=1, keepdim=True)
                        uncertainty = num_classes / S.squeeze(1) # Epistemic uncertainty
                        
                        # Soft Veto: High uncertainty scales down update weight
                        u_threshold = 0.5 
                        dirichlet_decay = torch.exp(-2.0 * torch.relu(uncertainty - u_threshold))
                        update_weights = update_weights * dirichlet_decay
                    
                    # 4. Temporal Uncertainty (Latent Prototype Drift Tracking)
                    # Handled by anchor_spring inside the prototype update block!
                        
                    # Calculate tracking metrics
                    # We define a "veto" as any point where the uncertainty method cut the base confidence by >50%
                    veto_mask = update_weights < (0.5 * base_weights)
                    # We define a point as "fired" if it contributes more than 1% weight
                    fired_mask = update_weights > 0.01
                    
                    if not hasattr(model, '_firing_log'):
                        model._firing_log = []
                    if not hasattr(model, '_class_firing_log'):
                        model._class_firing_log = {c: [] for c in range(num_classes)}
                        
                    if not eval_only and not hasattr(model, 'initial_classify_weights'):
                        model.initial_classify_weights = model.classify.weight.clone().detach()
                    
                    # Guard against empty tensors causing NaN
                    if len(update_weights) > 0:
                        model._firing_log.append(update_weights.mean().item())
                        for c in range(num_classes):
                            c_mask = (pseudo_labels == c)
                            if c_mask.any():
                                n_pts = c_mask.sum().item()
                                n_fired = (c_mask & fired_mask).sum().item()
                                sum_w = update_weights[c_mask].sum().item()
                                model._class_firing_log[c].append({'n_points': n_pts, 'n_fired': n_fired, 'sum_w': sum_w})
                    else:
                        model._firing_log.append(0.0)
                    
                    if update_method != 'prototype_cosine':
                        valid_gt_mask = (selected_labels >= 0) & (selected_labels < num_classes)
                        true_errors_rejected = (veto_mask & valid_gt_mask & (pseudo_labels != selected_labels)).sum().item()
                        correct_labels_rejected = (veto_mask & valid_gt_mask & (pseudo_labels == selected_labels)).sum().item()
                        if not hasattr(model, '_veto_stats'):
                            model._veto_stats = {'true_errors_rejected': 0, 'correct_labels_rejected': 0}
                        if not hasattr(model, '_class_veto_stats'):
                            model._class_veto_stats = {c: {'true_errors_rejected': 0, 'correct_labels_rejected': 0} for c in range(num_classes)}
                            
                        model._veto_stats['true_errors_rejected'] += true_errors_rejected
                        model._veto_stats['correct_labels_rejected'] += correct_labels_rejected
                        
                        for c in range(num_classes):
                            c_mask = (pseudo_labels == c)
                            if c_mask.any():
                                true_errs = (veto_mask & valid_gt_mask & c_mask & (pseudo_labels != selected_labels)).sum().item()
                                corr_labels = (veto_mask & valid_gt_mask & c_mask & (pseudo_labels == selected_labels)).sum().item()
                                model._class_veto_stats[c]['true_errors_rejected'] += true_errs
                                model._class_veto_stats[c]['correct_labels_rejected'] += corr_labels
                    
                    if fired_mask.any():
                        valid_enc = norm_enc[indices]
                        for c in range(num_classes):
                            c_mask = (pseudo_labels == c)
                            if c_mask.any():
                                c_weights = update_weights[c_mask].unsqueeze(1)
                                c_update = (valid_enc[c_mask] * c_weights).mean(dim=0)
                                
                                if c_update.norm(p=2) > 1e-6:
                                    c_update = F.normalize(c_update, p=2, dim=0)
                                    
                                    step_mag = update_lr * update_weights[c_mask].mean().item()
                                    
                                    if test_1b == 'mean_evidence':
                                        g = evidence[c_mask, c].mean().item()
                                        step_mag = step_mag * g
                                    elif test_1b == 'coherence':
                                        g = (valid_enc[c_mask].mean(dim=0).norm(p=2).item())
                                        step_mag = step_mag * g
                                    if update_method == 'evidential_hdc_tta':
                                        if hasattr(model, 'initial_classify_weights'):
                                            w_0 = F.normalize(model.initial_classify_weights[c], dim=0)
                                            w_t = F.normalize(model.classify.weight[c], dim=0)
                                            cos_sim = F.linear(w_t.unsqueeze(0), w_0.unsqueeze(0)).item()
                                            cos_sim = max(-1.0, min(1.0, cos_sim))
                                            angle = torch.acos(torch.tensor(cos_sim)).item() * (180.0 / torch.pi)
                                            if angle > 40.0:  # Hard cap at 40 degrees
                                                step_mag = 0.0
                                                
                                    model.classify.weight[c].data += step_mag * c_update.to(model.classify.weight.dtype)
                                    
                                    if update_method == 'evidential_hdc_tta' and hasattr(model, 'initial_classify_weights'):
                                        spring_k = 0.0005
                                        w_0 = F.normalize(model.initial_classify_weights[c], dim=0).to(model.classify.weight.device)
                                        model.classify.weight[c].data.copy_((1 - spring_k) * model.classify.weight[c].data + spring_k * w_0)
                                        
                                    if not reproduce_bug:
                                        model.classify.weight[c].data.copy_(F.normalize(model.classify.weight[c].data, p=2, dim=0))
                                    
                                    if not hasattr(model, '_update_magnitude_log'):
                                        model._update_magnitude_log = []
                                    model._update_magnitude_log.append(step_mag)
    
    if hasattr(model, '_veto_stats') and model._veto_stats['correct_labels_rejected'] > 0:
        purity_ratio = model._veto_stats['true_errors_rejected'] / model._veto_stats['correct_labels_rejected']
        # print(f"\n[Stats] Veto Purity Ratio: {purity_ratio:.2f} ({model._veto_stats['true_errors_rejected']} true errors rejected / {model._veto_stats['correct_labels_rejected']} correct labels rejected)")
        model._veto_stats = {'true_errors_rejected': 0, 'correct_labels_rejected': 0}
        
    if not eval_only and hasattr(model, 'initial_classify_weights'):
        import logging
        logger = logging.getLogger("EvalAdapt")
        class_rotations = {}
        for c in range(num_classes):
            w_0 = F.normalize(model.initial_classify_weights[c], dim=0)
            w_t = F.normalize(model.classify.weight[c], dim=0)
            cos_sim = F.linear(w_t.unsqueeze(0), w_0.unsqueeze(0)).item()
            cos_sim = max(-1.0, min(1.0, cos_sim))
            angle = torch.acos(torch.tensor(cos_sim)).item() * (180.0 / torch.pi)
            class_rotations[c] = angle
            
        logger.info(f"\n[Stats] Prototype Rotation (Degrees):")
        head_rot = {c: round(class_rotations[c], 2) for c in [11, 13, 14, 15, 16]}
        tail_rot = {c: round(class_rotations[c], 2) for c in [2, 3, 6, 7, 10]}
        logger.info(f"  Head Rotation: {head_rot}")
        logger.info(f"  Tail Rotation: {tail_rot}")
        
    if hasattr(model, '_class_veto_stats') and not eval_only:
        import logging
        logger = logging.getLogger("EvalAdapt")
        logger.info(f"\n[Stats] Per-Class Veto Purity (True Errors / Correct Labels Rejected):")
        head_purity = {}
        tail_purity = {}
        for c in range(num_classes):
            stats = model._class_veto_stats[c]
            if stats['correct_labels_rejected'] > 0:
                purity = stats['true_errors_rejected'] / stats['correct_labels_rejected']
            else:
                purity = -1.0
            if c in [11, 13, 14, 15, 16]:
                head_purity[c] = round(purity, 2)
            elif c in [2, 3, 6, 7, 10]:
                tail_purity[c] = round(purity, 2)
        logger.info(f"  Head Purity: {head_purity}")
        logger.info(f"  Tail Purity: {tail_purity}")
        model._class_veto_stats = {c: {'true_errors_rejected': 0, 'correct_labels_rejected': 0} for c in range(num_classes)}
        
    if hasattr(model, '_class_firing_log') and not eval_only:
        import logging
        logger = logging.getLogger("EvalAdapt")
        head_firing = {}
        tail_firing = {}
        for c in range(num_classes):
            logs = model._class_firing_log[c]
            if len(logs) > 0:
                n_frames = len(logs)
                n_points = sum(l['n_points'] for l in logs)
                n_fired = sum(l['n_fired'] for l in logs)
                mean_w = sum(l['sum_w'] for l in logs) / n_points if n_points > 0 else 0
                
                # Firing rate is now true fired points / total points
                rate = n_fired / n_points if n_points > 0 else 0.0
            else:
                rate = 0.0
            if c in [11, 13, 14, 15, 16]:
                head_firing[c] = round(rate, 4)
            elif c in [2, 3, 6, 7, 10]:
                tail_firing[c] = round(rate, 4)
        logger.info(f"\n[Stats] Per-Class Firing Rates (True Fired/Total):")
        logger.info(f"  Head Firing: {head_firing}")
        logger.info(f"  Tail Firing: {tail_firing}")
        model._class_firing_log = {c: [] for c in range(num_classes)}
        
    avg_firing_rate = 0.0
    if hasattr(model, '_firing_log') and len(model._firing_log) > 0:
        avg_firing_rate = sum(model._firing_log) / len(model._firing_log)
        model._firing_log = []
        
    avg_update_magnitude = 0.0
    if hasattr(model, '_update_magnitude_log') and len(model._update_magnitude_log) > 0:
        avg_update_magnitude = sum(model._update_magnitude_log) / len(model._update_magnitude_log)
        model._update_magnitude_log = []
        
    return {
        "mIoU": miou_history, 
        "Head_mIoU": head_miou_history,
        "Mid_mIoU": mid_miou_history,
        "Tail_mIoU": tail_miou_history,
        "Accuracy": acc_history, 
        "IoU_per_class": iou_per_class_history, 
        "FiringRate": avg_firing_rate, 
        "UpdateMagnitude": avg_update_magnitude
    }


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
                        
    counts_safe = torch.clamp(class_latent_counts, min=1).unsqueeze(1)
    model.class_latent_means = class_latent_sums / counts_safe
    
    # Initialize Latent Anchors for Temporal Drift tracking
    model.drift_mu_0 = model.class_latent_means.clone()
    
    # Pass 2: Calculate per-class density standard deviation and cos similarity statistics
    all_dists_per_class = {c: [] for c in range(num_classes)}
    all_cos_per_class = {c: [] for c in range(num_classes)}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Populating Source Stats")):
            if batch_idx > 50:
                break
            proj_in = batch_data[0].to(device)
            proj_labels = batch_data[2].to(device).view(-1)
            
            if proj_in.shape[1] > 0:
                with torch.amp.autocast('cuda', enabled=True):
                    latent_x = model.net(proj_in, only_feat=True)
                latent_x = latent_x.permute(0, 2, 3, 1).reshape(-1, 128)
                raw_enc, indices, _ = model.encode(proj_in)
                selected_labels = proj_labels[indices]
                valid_mask = (selected_labels >= 0) & (selected_labels < num_classes)
                
                if not valid_mask.any():
                    continue
                latent_valid = latent_x[valid_mask].float()
                labels_valid = selected_labels[valid_mask]
                
                pred_means = model.class_latent_means[labels_valid]
                dists = torch.norm(latent_valid - pred_means, p=2, dim=1)
                
                # Compute cos sims for Z-score calibration
                norm_enc = F.normalize(raw_enc[valid_mask], dim=1)
                if norm_enc.dtype != model.classify.weight.dtype:
                    norm_enc = norm_enc.to(model.classify.weight.dtype)
                logits = model.classify(norm_enc)
                true_cos = logits[torch.arange(logits.size(0)), labels_valid]
                
                for c in range(num_classes):
                    c_mask = labels_valid == c
                    if c_mask.any():
                        all_dists_per_class[c].append(dists[c_mask].cpu())
                        all_cos_per_class[c].append(true_cos[c_mask].cpu())
                
    
    model.source_density_std = torch.zeros(num_classes, device=device)
    model.source_mu_cos = torch.zeros(num_classes, device=device)
    model.source_sigma_cos = torch.zeros(num_classes, device=device)
    
    # We need a global fallback for classes that might not have appeared
    global_dists = []
    global_cos = []
    for c in range(num_classes):
        if len(all_dists_per_class[c]) > 0:
            c_dists = torch.cat(all_dists_per_class[c], dim=0)
            global_dists.append(c_dists)
        if len(all_cos_per_class[c]) > 0:
            c_cos = torch.cat(all_cos_per_class[c], dim=0)
            global_cos.append(c_cos)
            
    if len(global_dists) == 0:
        raise ValueError("Source statistics population failed: No valid latent features found in the first 50 frames.")
        
    global_dist_std = torch.cat(global_dists, dim=0).std().item()
    global_cos_tensor = torch.cat(global_cos, dim=0)
    global_cos_mean = global_cos_tensor.mean().item()
    global_cos_std = global_cos_tensor.std().item()
    
    for c in range(num_classes):
        if len(all_dists_per_class[c]) > 0:
            model.source_density_std[c] = torch.cat(all_dists_per_class[c], dim=0).std().item()
            c_cos = torch.cat(all_cos_per_class[c], dim=0)
            model.source_mu_cos[c] = c_cos.mean().item()
            model.source_sigma_cos[c] = c_cos.std().item()
        else:
            # Fallback to global statistics if class is completely missing from the first 50 frames
            model.source_density_std[c] = global_dist_std
            model.source_mu_cos[c] = global_cos_mean
            model.source_sigma_cos[c] = global_cos_std
    
    return {
        'source_density_std': model.source_density_std,
        'source_mu_cos': model.source_mu_cos,
        'source_sigma_cos': model.source_sigma_cos,
        'drift_mu_0': model.drift_mu_0.clone().cpu()
    }

def main():
    import random
    parser = argparse.ArgumentParser(description="Test Unsupervised Updates on KITTI-C")
    parser.add_argument('--pretrain', action='store_true', help='Run pretraining on SemanticKITTI before evaluating')
    parser.add_argument('--chunked', action='store_true', help='Use D3CTTA chunked protocol: continuous adaptation across disjoint 1/7th splits instead of full independent sequences.')
    parser.add_argument('--reset_per_corruption', action='store_true', help='Reset the model to the clean pretrained weights before adapting on each corruption (requires --chunked).')
    parser.add_argument('--pretrained_path', type=str, default='logs/kitti_pretrain/hdc_sub.pth', help='Path to load pretrained model')
    parser.add_argument('--log_dir', type=str, default='logs/kitti_c_test', help='Directory to save logs and graphics')
    parser.add_argument('--method', type=str, default='frozen', help='Method to test.')
    parser.add_argument('--reproduce_bug', action='store_true', help='Skip post-step normalization to reproduce the phase 1 magnitude explosion bug.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for noise floor tests')
    parser.add_argument('--dry_run', action='store_true', help='Run only 2 batches per condition to quickly verify no crashes will occur.')
    parser.add_argument('--continue_pretrain', action='store_true', help='Resume pretraining from the existing pretrained_path')
    parser.add_argument('--continue', dest='continue_epochs', type=int, default=0, help='Continue feature extractor training for this many epochs, reinitialize HDC, and perform adaptation')
    parser.add_argument('--skip_extractor', action='store_true', help='Skip feature extractor pretraining and only retrain the HDC model')
    parser.add_argument('--extractor_epochs', type=int, default=60, help='Number of epochs to train the feature extractor')
    parser.add_argument('--hdc_epochs', type=int, default=15, help='Number of epochs to train the HDC density model')
    parser.add_argument('--severity', type=int, default=3, help='Severity level for corruptions')
    parser.add_argument('--kitti_dir', type=str, default='/mnt/alpha/jmfleming/KITTI', help='Path to SemanticKITTI dataset for pretraining')
    parser.add_argument('--kittic_dir', type=str, default='/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C', help='Path to real SemanticKITTI-C dataset')
    parser.add_argument('--corruptions', type=str, default='snow,beam_missing,wet_ground', help='Comma separated list of corruptions to test. Defaults to diagnostic panel.')
    parser.add_argument('--test_1b', type=str, default='none', help='Test 1b: Comma-separated list of step magnitudes (e.g., none,count_throttle,rotation_cap,anchor_spring)')
    parser.add_argument('--test_1c', type=str, default='1.0', help='Test 1c: Comma-separated list of shrinkage lambdas (e.g., 1.0,0.75,0.50)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    methods_to_run = ['evidential_hdc_tta']
    
    global_results_path = os.path.join(args.log_dir, 'global_results.json')
    global_results = None
    if os.path.exists(global_results_path):
        try:
            with open(global_results_path, 'r') as f:
                global_results = json.load(f)
        except json.JSONDecodeError:
            global_results = None
            
    test_1b_list = args.test_1b.split(',')
    test_1c_list = [float(x) for x in args.test_1c.split(',')]
    
    configs_to_run = []
    for m in methods_to_run:
        if m == 'frozen':
            configs_to_run.append((m, 'none', 1.0))
        else:
            for tb in test_1b_list:
                for tc in test_1c_list:
                    configs_to_run.append((m, tb, tc))
    
    full_method_names = [f"{m}_1b_{tb}_1c_{tc}" if m != 'frozen' else 'frozen' for (m, tb, tc) in configs_to_run]
                    
    if global_results is not None:
        # Ensure the dicts for the current methods exist in case they were never run
        for m in full_method_names:
            if m not in global_results.get('mIoU', {}):
                global_results.setdefault('mIoU', {})[m] = {c: {} for c in CORRUPTIONS}
            if m not in global_results.get('Accuracy', {}):
                global_results.setdefault('Accuracy', {})[m] = {c: {} for c in CORRUPTIONS}
    else:
        global_results = {
            'mIoU': {m: {c: {} for c in CORRUPTIONS} for m in full_method_names},
            'Accuracy': {m: {c: {} for c in CORRUPTIONS} for m in full_method_names},
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

    base_model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES)
    populate_source_statistics(base_model, args.kitti_dir, ARCH, DATA, device)
    
    source_stats_cache = {
        'class_latent_means': base_model.class_latent_means,
        'source_density_std': getattr(base_model, 'source_density_std', None),
        'source_mu_cos': getattr(base_model, 'source_mu_cos', None),
        'source_sigma_cos': getattr(base_model, 'source_sigma_cos', None),
        'drift_mu_0': getattr(base_model, 'drift_mu_0', None)
    }

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
        model.class_latent_means = source_stats_cache['class_latent_means']
        model.source_density_std = source_stats_cache['source_density_std']
        model.source_mu_cos = source_stats_cache['source_mu_cos']
        model.source_sigma_cos = source_stats_cache['source_sigma_cos']
        model.drift_mu_0 = source_stats_cache['drift_mu_0']

    for (current_method, t1b, t1c), full_method_name in zip(configs_to_run, full_method_names):
        logger.info(f"=========================================")
        logger.info(f"Starting Evaluation for Method: {full_method_name}")
        logger.info(f"=========================================")
        
        active_corruptions = CORRUPTIONS
        if args.corruptions:
            active_corruptions = [c.strip() for c in args.corruptions.split(',')]

        results_miou = {c: {} for c in active_corruptions}
        results_acc = {c: {} for c in active_corruptions}

        # Reset model at the start of each new method loop
        model.load_state_dict(clean_state_dict, strict=False)
        if hasattr(model, 'initial_classify_weights'):
            del model.initial_classify_weights
        if hasattr(model, 'drift_mu_c'):
            del model.drift_mu_c
        if hasattr(model, 'class_freq_ema'):
            del model.class_freq_ema
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
                if hasattr(model, 'initial_classify_weights'):
                    del model.initial_classify_weights
                if hasattr(model, 'drift_mu_c'):
                    del model.drift_mu_c
                if hasattr(model, 'class_freq_ema'):
                    del model.class_freq_ema
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
                if hasattr(model, 'initial_classify_weights'):
                    del model.initial_classify_weights
                if hasattr(model, 'drift_mu_c'):
                    del model.drift_mu_c
                if hasattr(model, 'class_freq_ema'):
                    del model.class_freq_ema
                if hasattr(model, 'subcluster_update_counts') and model.subcluster_update_counts is not None:
                    model.subcluster_update_counts.zero_()
            else:
                # D3CTTA protocol: chunks, continuous adaptation
                chunk_dataset = torch.utils.data.Subset(full_corruption_dataset, chunks[i])
            
            target_dataloader = DataLoader(chunk_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
            
            try:
                if not args.chunked or args.reset_per_corruption:
                    # Pass 1: True Initial (Frozen on chunk)
                    if (ctype, sev) not in shared_init_metrics:
                        logger.info("  -> Pass 1: Computing True Initial metrics (Frozen)")
                        init_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=True, dry_run=args.dry_run, test_1b='none', test_1c=1.0, reproduce_bug=getattr(args, 'reproduce_bug', False))
                        shared_init_metrics[(ctype, sev)] = init_metrics
                    else:
                        logger.info("  -> Pass 1: Reusing cached True Initial metrics (Frozen)")
                        init_metrics = shared_init_metrics[(ctype, sev)]
                    
                    # Pass 2: Adapt (only if method is not frozen)
                    if current_method != 'frozen':
                        logger.info("  -> Pass 2: Adapting model weights")
                        eval_model.train()
                        adapt_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=False, update_method=current_method, dry_run=args.dry_run, test_1b=t1b, test_1c=t1c, reproduce_bug=getattr(args, 'reproduce_bug', False))
                    else:
                        adapt_metrics = init_metrics
                        
                    # Pass 3: True Final (Frozen on chunk using adapted weights)
                    logger.info("  -> Pass 3: Computing True Final metrics (Frozen)")
                    eval_model.eval()
                    final_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=True, dry_run=args.dry_run, test_1b=t1b, test_1c=t1c, reproduce_bug=getattr(args, 'reproduce_bug', False))
                    
                    # We only care about the absolute end of the frozen evaluations for the sequence
                    metrics = adapt_metrics  # Just for the trajectory json
                    if len(init_metrics["mIoU"]) > 0:
                        initial_miou = init_metrics["mIoU"][-1]
                        final_miou = final_metrics["mIoU"][-1]
                        online_miou = adapt_metrics["mIoU"][-1]
                        initial_acc = init_metrics["Accuracy"][-1]
                        final_acc = final_metrics["Accuracy"][-1]
                    else:
                        initial_miou = final_miou = online_miou = initial_acc = final_acc = 0.0
                        
                    firing_rate_str = ""
                    if "FiringRate" in adapt_metrics:
                        firing_rate_str = f", FiringRate={adapt_metrics['FiringRate']*100:.2f}%"
                        if "UpdateMagnitude" in adapt_metrics:
                            firing_rate_str += f", UpdateMag={adapt_metrics['UpdateMagnitude']:.4f}"
                else:
                    # Original single-pass continuous evaluation
                    metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=(current_method == 'frozen'), update_method=current_method, dry_run=args.dry_run, test_1b=t1b, test_1c=t1c, reproduce_bug=getattr(args, 'reproduce_bug', False))
                    if len(metrics["mIoU"]) > 0:
                        initial_miou = metrics["mIoU"][0]
                        final_miou = metrics["mIoU"][-1]
                        online_miou = final_miou
                        initial_acc = metrics["Accuracy"][0]
                        final_acc = metrics["Accuracy"][-1]
                    else:
                        initial_miou = final_miou = online_miou = initial_acc = final_acc = 0.0
                        
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
                
                global_results['mIoU'][full_method_name][ctype][sev] = (initial_miou, final_miou)
                global_results['Accuracy'][full_method_name][ctype][sev] = (initial_acc, final_acc)
                
                if not args.chunked or args.reset_per_corruption:
                    initial_tail = init_metrics["Tail_mIoU"][-1] if current_method != 'frozen' else metrics["Tail_mIoU"][0]
                    final_tail = final_metrics["Tail_mIoU"][-1] if current_method != 'frozen' else metrics["Tail_mIoU"][-1]
                else:
                    initial_tail = metrics["Tail_mIoU"][0]
                    final_tail = metrics["Tail_mIoU"][-1]
                
                logger.info(f"Result for {ctype}-{sev}: Initial mIoU={initial_miou:.4f} -> Final (Online)={online_miou:.4f} -> Final (Frozen)={final_miou:.4f} (Tail: {initial_tail:.4f} -> {final_tail:.4f}), Acc={initial_acc:.4f} -> {final_acc:.4f}{firing_rate_str}")
                suffix = f"_{full_method_name}"
                
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
