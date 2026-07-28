import math
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
from modules import compare as _baselines
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
SEVERITY_MAP = {1: 'light', 2: 'moderate', 3: 'heavy', 4: 'extreme'}

CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS_KITTI_ALL = "config/labels/semantic-kitti-all.yaml"  # Standard 17 classes

ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
DATA = yaml.safe_load(open(CONFIG_LABELS_KITTI_ALL, 'r'))

def compute_auroc_torch(scores, labels):
    if len(scores) == 0: return float('nan')
    pos_mask = labels.bool()
    n1 = pos_mask.sum().item()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    unique_vals, inv, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    cum_counts = torch.cumsum(counts, dim=0)
    start_ranks = cum_counts - counts + 1
    avg_ranks = (start_ranks + cum_counts) / 2.0
    ranks = avg_ranks[inv].to(torch.float64)
    r1 = ranks[pos_mask].sum().item()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))

def compute_correlations_torch(x, y):
    if len(x) < 2: return "N/A"
    x = x.to(torch.float64); y = y.to(torch.float64)
    vx = x - x.mean(); vy = y - y.mean()
    cov = (vx * vy).mean()
    sx = torch.sqrt((vx**2).mean()); sy = torch.sqrt((vy**2).mean())
    pearson = float(cov / (sx * sy)) if sx != 0 and sy != 0 else 0.0
    _, inv_x, counts_x = torch.unique(x, return_inverse=True, return_counts=True)
    cum_x = torch.cumsum(counts_x, dim=0)
    ranks_x = ((cum_x - counts_x + 1 + cum_x) / 2.0)[inv_x].to(torch.float64)
    _, inv_y, counts_y = torch.unique(y, return_inverse=True, return_counts=True)
    cum_y = torch.cumsum(counts_y, dim=0)
    ranks_y = ((cum_y - counts_y + 1 + cum_y) / 2.0)[inv_y].to(torch.float64)
    vrx = ranks_x - ranks_x.mean(); vry = ranks_y - ranks_y.mean()
    cov_r = (vrx * vry).mean()
    srx = torch.sqrt((vrx**2).mean()); sry = torch.sqrt((vry**2).mean())
    spearman = float(cov_r / (srx * sry)) if srx != 0 and sry != 0 else 0.0
    return f"Pearson r={pearson:.6f}, Spearman rho={spearman:.6f} (over {len(x):,} pairs)"

def evaluate_and_adapt(model, target_dataloader, device, eval_only=False, update_method='frozen', dry_run=False, custom_update_fn=None, ic_method='none', tau=None, kappa=15.0, normalize_weights=False, mv_tta='none', gate_mode='epistemic', dynamic_geom=False, diagnostics=True, dump_features=False):
    if ic_method not in ['none', 'ic4']:
        raise ValueError(f"Unknown ic_method: {ic_method}")
    logger = logging.getLogger("EvalAdapt")

    model_was_training = model.training
    miou_history = []
    head_miou_history = []
    mid_miou_history = []
    tail_miou_history = []
    acc_history = []
    iou_per_class_history = []
    num_classes = model.num_classes
    cumulative_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    agree_conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    disagree_conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    
    prev_preds_2d = None

    active_mu_cos = model.source_mu_cos
    active_sigma_cos = model.source_sigma_cos

    if not eval_only and hasattr(model, 'classify'):
        norms = {c: round(model.classify.weight[c].norm().item(), 4) for c in range(17)}
        logger.info(f"\n[Stats] Initial Prototype Norms: {norms}")

    if hasattr(model, 'class_latent_means') and model.class_latent_means is not None:
        model.class_latent_means = model.class_latent_means.to(device)
        means_norm_128_cached = F.normalize(model.class_latent_means.float(), dim=1)
    else:
        means_norm_128_cached = None

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
                # Get raw latent and encodings for updates
                with torch.amp.autocast('cuda', enabled=True):
                    latent_x = model.net(proj_in, only_feat=True)
                latent_x = latent_x.permute(0, 2, 3, 1).reshape(-1, 128)
                
                # === MULTI-VIEW TTA AUGMENTATIONS (BATCHED) ===
                if mv_tta != 'none' or gate_mode == 'view_var_gate':
                    B_val, _, H_val, W_val = proj_in.shape
                    shift_amount = W_val // 4
                    proj_m1 = torch.roll(proj_in, shifts=shift_amount, dims=3)
                    proj_m2 = proj_in * 0.95
                    
                    # Batch all 3 views into a single forward pass! (Size: [3 * B, 5, H, W])
                    batched_proj = torch.cat([proj_in, proj_m1, proj_m2], dim=0)
                    raw_enc_batched, _, _ = model.encode(batched_proj)
                    C_val = raw_enc_batched.shape[-1]
                    
                    # Unpack batched encodings: shape [3, B, H, W, C]
                    raw_enc_batched = raw_enc_batched.view(3, B_val, H_val, W_val, C_val)
                    
                    # Base view
                    raw_enc = raw_enc_batched[0].view(-1, C_val)
                    norm_enc = F.normalize(raw_enc, dim=1).to(model.classify.weight.dtype)
                    indices = torch.arange(raw_enc.shape[0], device=device)
                    
                    # M1 (Yaw Shift) - Roll back horizontally to align features
                    raw_enc_m1 = raw_enc_batched[1]
                    raw_enc_m1 = torch.roll(raw_enc_m1, shifts=-shift_amount, dims=2).view(-1, C_val)
                    norm_enc_m1 = F.normalize(raw_enc_m1, dim=1).to(model.classify.weight.dtype)
                    
                    # M2 (Depth Scale)
                    raw_enc_m2 = raw_enc_batched[2].view(-1, C_val)
                    norm_enc_m2 = F.normalize(raw_enc_m2, dim=1).to(model.classify.weight.dtype)
                    
                    norm_enc_base = norm_enc.clone()
                else:
                    raw_enc, indices, _ = model.encode(proj_in)
                    assert torch.equal(indices, torch.arange(raw_enc.shape[0], device=device)), "encode() indices do not match full arange point set!"
                    norm_enc = F.normalize(raw_enc, dim=1).to(model.classify.weight.dtype)
                # ====================================
                
                norm_enc = norm_enc.to(model.classify.weight.dtype)
                if 'norm_enc_base' in locals():
                    norm_enc_base = norm_enc_base.to(model.classify.weight.dtype)
                
                # --- BASELINE METHODS: use the baseline's OWN head for prediction ---
                # Previously the HDC classifier produced predictions and only DAPL
                # prototypes were synced back, so D3CTTA's Domain-Specific
                # Decorrelation never influenced a single prediction.
                _baseline_state = None
                if update_method in _baselines.BASELINE_METHODS:
                    _b_logits, _baseline_state = _baselines.baseline_forward(
                        model, update_method, proj_in, proj_xyz, num_classes, device)
                    predictions = torch.argmax(_b_logits, dim=1)
                    indices = torch.arange(_b_logits.shape[0], device=device)

                if tau is not None:
                    w_norm = F.normalize(model.classify.weight, p=2, dim=1)
                    
                    def get_logits(enc):
                        l = F.linear(enc, w_norm) * kappa
                        if tau != 0.0 and hasattr(model, 'source_class_freq'):
                            pi = torch.clamp(model.source_class_freq, min=1e-5).to(device)
                            l = l - tau * torch.log(pi).unsqueeze(0)
                        return l
                        
                    logits = get_logits(norm_enc)
                else:
                    def get_logits(enc):
                        return model.classify(enc)
                        
                    logits = get_logits(norm_enc)
                    
                # Calculate multi-view predictions if needed for MV-1
                if mv_tta == 'conf_pred' and 'norm_enc_m1' in locals():
                    logits_m1 = get_logits(norm_enc_m1)
                    logits_m2 = get_logits(norm_enc_m2)
                    
                    # Average probabilities across views
                    prob_base = F.softmax(logits, dim=1)
                    prob_m1 = F.softmax(logits_m1, dim=1)
                    prob_m2 = F.softmax(logits_m2, dim=1)
                    mean_probs = (prob_base + prob_m1 + prob_m2) / 3.0
                    predictions = torch.argmax(mean_probs, dim=1)
                elif _baseline_state is None:
                    predictions = torch.argmax(logits, dim=1)
                    
                # MV-2: View Disagreement Precision Tracking & V2 Soft View Variance
                if (mv_tta != 'none' or gate_mode == 'view_var_gate') and 'norm_enc_m1' in locals():
                    l_base_mv2 = get_logits(norm_enc_base)
                    l_m1_mv2 = get_logits(norm_enc_m1)
                    l_m2_mv2 = get_logits(norm_enc_m2)
                    pred_base_mv2 = torch.argmax(l_base_mv2, dim=1)
                    pred_m1_mv2 = torch.argmax(l_m1_mv2, dim=1)
                    pred_m2_mv2 = torch.argmax(l_m2_mv2, dim=1)
                    view_preds = [pred_m1_mv2, pred_m2_mv2]
                    view_disagreement = (pred_base_mv2 != pred_m1_mv2) | (pred_base_mv2 != pred_m2_mv2)
                    
                    pb_mv2 = F.softmax(l_base_mv2, dim=1)
                    p1_mv2 = F.softmax(l_m1_mv2, dim=1)
                    p2_mv2 = F.softmax(l_m2_mv2, dim=1)
                    pm_mv2 = (pb_mv2 + p1_mv2 + p2_mv2) / 3.0
                    soft_view_var_all = ((pb_mv2 - pm_mv2)**2 + (p1_mv2 - pm_mv2)**2 + (p2_mv2 - pm_mv2)**2).sum(dim=1) / 3.0
                else:
                    view_preds = None
                    view_disagreement = torch.zeros_like(predictions, dtype=torch.bool)
                    soft_view_var_all = torch.zeros(len(predictions), device=device)

                
                selected_labels = proj_labels[indices]
                mask = (selected_labels >= 0) & (selected_labels < num_classes)
                if mask.any():
                    hist = torch.bincount(
                        num_classes * selected_labels[mask] + predictions[mask], 
                        minlength=num_classes ** 2
                    ).reshape(num_classes, num_classes)
                    cumulative_confusion_matrix += hist
                    
                    if mv_tta != 'none' or gate_mode == 'view_var_gate':
                        agree_mask = mask & (~view_disagreement)
                        if agree_mask.any():
                            agree_hist = torch.bincount(
                                num_classes * selected_labels[agree_mask] + predictions[agree_mask], 
                                minlength=num_classes ** 2
                            ).reshape(num_classes, num_classes)
                            agree_conf_matrix += agree_hist
                            
                        disagree_mask = mask & view_disagreement
                        if disagree_mask.any():
                            disagree_hist = torch.bincount(
                                num_classes * selected_labels[disagree_mask] + predictions[disagree_mask], 
                                minlength=num_classes ** 2
                            ).reshape(num_classes, num_classes)
                            disagree_conf_matrix += disagree_hist
                
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
                    update_lr = 0.01
                    proto_norm = F.normalize(model.classify.weight, dim=1)
                    cos_sims = F.linear(norm_enc, proto_norm)
                    
                    if tau is not None and tau != 0.0 and hasattr(model, 'source_class_freq'):
                        pl_logits = cos_sims * kappa
                        pi = torch.clamp(model.source_class_freq, min=1e-5).to(device)
                        pl_logits = pl_logits - tau * torch.log(pi).unsqueeze(0)
                        pseudo_labels = torch.argmax(pl_logits, dim=1)
                    else:
                        pseudo_labels = torch.argmax(cos_sims, dim=1)
                        
                    max_cos_sim = cos_sims[torch.arange(cos_sims.size(0)), pseudo_labels]
                    
                    # Soft Gating Initialization
                    # HDC cosine similarities are extremely small (e.g. ~0.05 to ~0.15). 
                    # We use a sharp temperature scaling (x 100.0) to properly stretch these into [0, 1] probability weights.
                    cos_sim_probs = F.softmax(cos_sims * 100.0, dim=1)
                    base_weights = cos_sim_probs.max(dim=1)[0]
                    update_weights = base_weights.clone()
                    
                    uncertainty = None
                    z_score = None
                    
                    # Avoid multi-GB copies since indices is always arange
                    latent_x_valid = latent_x

                    if update_method in ['evidential_hdc_tta', 'bm_ic4', 'bm'] or 'evidential' in update_method or 'bm' in update_method:
                        # 1. & 2. Compute Confidence & Gating via DualGateModel / MV_TTAModel
                        update_weights, uncertainty, z_score = model.get_confidence(
                            latent_x_valid,
                            preds=pseudo_labels,
                            method=gate_mode,
                            logits=cos_sims,
                            active_mu_cos=active_mu_cos,
                            active_sigma_cos=active_sigma_cos,
                            dynamic_geom=dynamic_geom,
                            view_preds=view_preds,
                            view_var=soft_view_var_all
                        )
                        epistemic_decay = torch.exp(-2.0 * torch.relu(uncertainty - 0.5))
                        geom_decay = torch.exp(-2.0 * torch.relu(z_score - 0.5))
                        
                        valid_gt_mask = (selected_labels >= 0) & (selected_labels < num_classes)
                        gt_corr = (pseudo_labels == selected_labels)
                        
                        if diagnostics:
                            # Track decay statistics, cosines, and GT-labelled contingency table (subsample by ::256 to prevent RAM bloat and quantile indexing limits)
                            if not hasattr(model, '_decay_logs'):
                                model._decay_logs = {'geom': [], 'epi': [], 'geom_score': [], 'epi_score': [], 'gt_valid': [], 'gt_corr': [], 'cos128': [], 'cos10k': [], 'view_dis': [], 'margin': []}
                            model._decay_logs['geom'].append(geom_decay.detach()[::256].cpu())
                            model._decay_logs['epi'].append(epistemic_decay.detach()[::256].cpu())
                            model._decay_logs['geom_score'].append((-z_score).detach()[::256].cpu())
                            model._decay_logs['epi_score'].append((-uncertainty).detach()[::256].cpu())
                            model._decay_logs['gt_valid'].append(valid_gt_mask.detach()[::256].cpu())
                            model._decay_logs['gt_corr'].append(gt_corr.detach()[::256].cpu())
                            
                            # Margin between top-1 and top-2 cosine similarities
                            top2_cos = torch.topk(cos_sims, k=min(2, cos_sims.size(1)), dim=1)[0]
                            margin = (top2_cos[:, 0] - top2_cos[:, 1]) if top2_cos.size(1) > 1 else top2_cos[:, 0]
                            model._decay_logs['margin'].append(margin.detach()[::256].cpu())
                            
                            # Test D0: Random point pairs for reference-free isometry check
                            if len(latent_x_valid) > 1:
                                perm = torch.randperm(len(latent_x_valid), device=device)
                                n_pairs = min(len(latent_x_valid) // 2, 512)
                                idx1, idx2 = perm[:n_pairs], perm[n_pairs:2*n_pairs]
                                l_norm_128 = F.normalize(latent_x_valid.float(), dim=1)
                                c128_vals = (l_norm_128[idx1] * l_norm_128[idx2]).sum(dim=1)
                                c10k_vals = (norm_enc[idx1].float() * norm_enc[idx2].float()).sum(dim=1)
                                model._decay_logs['cos128'].append(c128_vals.detach().cpu())
                                model._decay_logs['cos10k'].append(c10k_vals.detach().cpu())
                                
                            model._decay_logs['view_dis'].append(view_disagreement.detach()[::256].cpu())
                            
                            if not hasattr(model, '_contingency_table'):
                                model._contingency_table = {
                                    'geom_adm_epi_adm': {'n': torch.tensor(0, dtype=torch.long, device=device), 'correct': torch.tensor(0, dtype=torch.long, device=device)},
                                    'geom_adm_epi_rej': {'n': torch.tensor(0, dtype=torch.long, device=device), 'correct': torch.tensor(0, dtype=torch.long, device=device)},
                                    'geom_rej_epi_adm': {'n': torch.tensor(0, dtype=torch.long, device=device), 'correct': torch.tensor(0, dtype=torch.long, device=device)},
                                    'geom_rej_epi_rej': {'n': torch.tensor(0, dtype=torch.long, device=device), 'correct': torch.tensor(0, dtype=torch.long, device=device)},
                                }
                            if not hasattr(model, '_mv_contingency_table'):
                                model._mv_contingency_table = {
                                    'view_agree_epi_adm': {'n': torch.tensor(0, dtype=torch.long, device=device), 'correct': torch.tensor(0, dtype=torch.long, device=device)},
                                    'view_agree_epi_rej': {'n': torch.tensor(0, dtype=torch.long, device=device), 'correct': torch.tensor(0, dtype=torch.long, device=device)},
                                    'view_disagree_epi_adm': {'n': torch.tensor(0, dtype=torch.long, device=device), 'correct': torch.tensor(0, dtype=torch.long, device=device)},
                                    'view_disagree_epi_rej': {'n': torch.tensor(0, dtype=torch.long, device=device), 'correct': torch.tensor(0, dtype=torch.long, device=device)},
                                }
                            if valid_gt_mask.any():
                                ga = (geom_decay >= 0.5)[valid_gt_mask]
                                ea = (epistemic_decay >= 0.5)[valid_gt_mask]
                                corr = gt_corr[valid_gt_mask]
                                
                                m_aa = ga & ea
                                model._contingency_table['geom_adm_epi_adm']['n'] += m_aa.sum()
                                model._contingency_table['geom_adm_epi_adm']['correct'] += (m_aa & corr).sum()
                                
                                m_ar = ga & (~ea) # The rescue cell
                                model._contingency_table['geom_adm_epi_rej']['n'] += m_ar.sum()
                                model._contingency_table['geom_adm_epi_rej']['correct'] += (m_ar & corr).sum()
                                
                                m_ra = (~ga) & ea
                                model._contingency_table['geom_rej_epi_adm']['n'] += m_ra.sum()
                                model._contingency_table['geom_rej_epi_adm']['correct'] += (m_ra & corr).sum()
                                
                                m_rr = (~ga) & (~ea)
                                model._contingency_table['geom_rej_epi_rej']['n'] += m_rr.sum()
                                model._contingency_table['geom_rej_epi_rej']['correct'] += (m_rr & corr).sum()

                                va = (~view_disagreement)[valid_gt_mask]
                                vd = view_disagreement[valid_gt_mask]
                                
                                va_ea = va & ea
                                model._mv_contingency_table['view_agree_epi_adm']['n'] += va_ea.sum()
                                model._mv_contingency_table['view_agree_epi_adm']['correct'] += (va_ea & corr).sum()
                                
                                va_er = va & (~ea) # Test D4 Rescue cell!
                                model._mv_contingency_table['view_agree_epi_rej']['n'] += va_er.sum()
                                model._mv_contingency_table['view_agree_epi_rej']['correct'] += (va_er & corr).sum()
                                
                                vd_ea = vd & ea
                                model._mv_contingency_table['view_disagree_epi_adm']['n'] += vd_ea.sum()
                                model._mv_contingency_table['view_disagree_epi_adm']['correct'] += (vd_ea & corr).sum()
                                
                                vd_er = vd & (~ea)
                                model._mv_contingency_table['view_disagree_epi_rej']['n'] += vd_er.sum()
                                model._mv_contingency_table['view_disagree_epi_rej']['correct'] += (vd_er & corr).sum()
                                    
                        if dump_features:
                            dump_mask = valid_gt_mask
                            if dump_mask.any():
                                indices_dump = torch.nonzero(dump_mask, as_tuple=True)[0]
                                if len(indices_dump) > 1000:
                                    indices_dump = indices_dump[::len(indices_dump)//1000]
                                if not hasattr(model, '_feature_dump_list'):
                                    model._feature_dump_list = []
                                if not hasattr(model, '_proj_32'):
                                    torch.manual_seed(42)
                                    proj_mat = torch.randn(128, 32, device=device)
                                    q_mat, _ = torch.linalg.qr(proj_mat)
                                    model._proj_32 = q_mat
                                    
                                top2_c = torch.topk(cos_sims[indices_dump], k=min(2, cos_sims.size(1)), dim=1)[0]
                                m_val = (top2_c[:, 0] - top2_c[:, 1]) if top2_c.size(1) > 1 else top2_c[:, 0]
                                probs_dump = F.softmax(cos_sims[indices_dump] * 100.0, dim=1)
                                msp_val = probs_dump.max(dim=1)[0]
                                ent_val = -(probs_dump * torch.log(probs_dump + 1e-8)).sum(dim=1)
                                energy_val = -torch.logsumexp(cos_sims[indices_dump] * 15.0, dim=1)
                                
                                z_dump = (cos_sims[indices_dump] - active_mu_cos) / (active_sigma_cos + 1e-8)
                                alpha_dump = F.softplus(5.0 * z_dump) + 1.0
                                S_dump = alpha_dump.sum(dim=1, keepdim=True)
                                p_dump = alpha_dump / S_dump
                                h_tot = -(p_dump * torch.log(p_dump + 1e-8)).sum(dim=1)
                                aleatoric = (p_dump * (torch.special.digamma(S_dump + 1.0) - torch.special.digamma(alpha_dump + 1.0))).sum(dim=1)
                                mi_val = h_tot - aleatoric
                                
                                if hasattr(model, 'class_latent_means') and model.class_latent_means is not None:
                                    all_d = torch.cdist(latent_x_valid[indices_dump].float(), model.class_latent_means.float())
                                    d_own = all_d[torch.arange(len(indices_dump)), pseudo_labels[indices_dump]]
                                    all_d_other = all_d.clone()
                                    all_d_other[torch.arange(len(indices_dump)), pseudo_labels[indices_dump]] = float('inf')
                                    d_other = all_d_other.min(dim=1)[0]
                                    rel_mahal_val = d_own - d_other
                                else:
                                    logger.warning("CRITICAL WARNING: model.class_latent_means is missing or None during diagnostic feature dump! Defaulting rel_mahal_val to 0.0.")
                                    rel_mahal_val = torch.zeros(len(indices_dump), device=device)
                                    
                                if hasattr(model, 'source_bank') and model.source_bank is not None:
                                    d_bank = torch.cdist(latent_x_valid[indices_dump].float(), model.source_bank.float())
                                    knn_val = torch.topk(d_bank, k=min(5, d_bank.size(1)), dim=1, largest=False)[0].mean(dim=1)
                                else:
                                    logger.warning("CRITICAL WARNING: model.source_bank is missing or None during diagnostic feature dump! Defaulting knn_val to 0.0.")
                                    knn_val = torch.zeros(len(indices_dump), device=device)
                                    
                                vd_val = view_disagreement[indices_dump].float()
                                svv_val = soft_view_var_all[indices_dump].float()
                                
                                pin_flat = proj_in.permute(0, 2, 3, 1).reshape(-1, proj_in.shape[1])
                                r_val = pin_flat[indices_dump, 0] if pin_flat.shape[1] > 0 else torch.zeros(len(indices_dump), device=device)
                                i_val = pin_flat[indices_dump, 1] if pin_flat.shape[1] > 1 else torch.zeros(len(indices_dump), device=device)
                                p_norm_val = model.classify.weight[pseudo_labels[indices_dump]].norm(p=2, dim=1)
                                
                                model._feature_dump_list.append({
                                    'gt_corr': gt_corr[indices_dump].float().cpu(),
                                    'epi_score': (-uncertainty[indices_dump]).float().cpu(),
                                    'msp': msp_val.float().cpu(),
                                    'margin': m_val.float().cpu(),
                                    'entropy': ent_val.float().cpu(),
                                    'energy': energy_val.float().cpu(),
                                    'dirichlet_mi': mi_val.float().cpu(),
                                    'z_score': z_score[indices_dump].float().cpu(),
                                    'rel_mahal': rel_mahal_val.float().cpu(),
                                    'knn_dist': knn_val.float().cpu(),
                                    'latent_norm': latent_x_valid[indices_dump].norm(p=2, dim=1).float().cpu(),
                                    'latent_proj': (latent_x_valid[indices_dump].float() @ model._proj_32).cpu(),
                                    'view_dis': vd_val.cpu(),
                                    'soft_view_var': svv_val.float().cpu(),
                                    'range': r_val.float().cpu(),
                                    'intensity': i_val.float().cpu(),
                                    'pseudo_c': pseudo_labels[indices_dump].float().cpu(),
                                    'proto_norm_c': p_norm_val.float().cpu()
                                })

                        if gate_mode == 'oracle':
                            gt_mask_full = (pseudo_labels == selected_labels) & (selected_labels >= 0) & (selected_labels < num_classes)
                            update_weights = update_weights * gt_mask_full.float()
                        
                    # Calculate tracking metrics
                    # We define a "veto" as any point where the uncertainty method cut the base confidence by >50%
                    veto_mask = update_weights < (0.5 * base_weights)
                    # We define a point as "fired" if it passed the Epistemic Veto
                    fired_mask = ~veto_mask
                    if mv_tta == 'veto_disagree':
                        fired_mask = fired_mask & (~view_disagreement)
                    effective_veto = ~fired_mask
                    
                    if not hasattr(model, '_firing_log'):
                        model._firing_log = []
                    if not hasattr(model, '_class_n_points'):
                        model._class_n_points = torch.zeros(num_classes, dtype=torch.long, device=device)
                        model._class_n_fired = torch.zeros(num_classes, dtype=torch.long, device=device)
                        
                    if not eval_only and not hasattr(model, 'initial_classify_weights'):
                        model.initial_classify_weights = model.classify.weight.clone().detach()
                    
                    # Guard against empty tensors causing NaN
                    if len(update_weights) > 0:
                        model._firing_log.append(fired_mask.float().mean().item())
                        model._class_n_points += torch.bincount(pseudo_labels, minlength=num_classes)
                        if fired_mask.any():
                            model._class_n_fired += torch.bincount(pseudo_labels[fired_mask], minlength=num_classes)
                    else:
                        model._firing_log.append(0.0)
                    
                    valid_gt_mask = (selected_labels >= 0) & (selected_labels < num_classes)
                    err_mask = effective_veto & valid_gt_mask & (pseudo_labels != selected_labels)
                    corr_mask = effective_veto & valid_gt_mask & (pseudo_labels == selected_labels)
                    
                    if not hasattr(model, '_veto_stats'):
                        model._veto_stats = {'true_errors_rejected': 0, 'correct_labels_rejected': 0}
                    if not hasattr(model, '_class_true_errors_rejected'):
                        model._class_true_errors_rejected = torch.zeros(num_classes, dtype=torch.long, device=device)
                        model._class_correct_rejected = torch.zeros(num_classes, dtype=torch.long, device=device)
                        
                    model._veto_stats['true_errors_rejected'] += err_mask.sum().item()
                    model._veto_stats['correct_labels_rejected'] += corr_mask.sum().item()
                    
                    if err_mask.any():
                        model._class_true_errors_rejected += torch.bincount(pseudo_labels[err_mask], minlength=num_classes)
                    if corr_mask.any():
                        model._class_correct_rejected += torch.bincount(pseudo_labels[corr_mask], minlength=num_classes)
                    
                    # BUG FIX: this used to be `elif update_method in [...]` hanging off
                    # `if fired_mask.any():`. For baseline methods the gating block is
                    # skipped, so update_weights == base_weights exactly, veto_mask is
                    # all-False and fired_mask is all-True -- the elif was unreachable.
                    # All three "baselines" silently ran the generic HDC path, which is
                    # why they report bit-identical numbers in every cell.
                    if update_method in _baselines.BASELINE_METHODS:
                        _baselines.baseline_update(_baseline_state, predictions, proj_xyz)
                    elif fired_mask.any():
                        model.online_update(
                            norm_enc,
                            pseudo_labels,
                            update_weights,
                            update_method=update_method,
                            ic_method=ic_method,
                            uncertainty=uncertainty,
                            update_lr=update_lr,
                            normalize_weights=normalize_weights,
                            view_preds=view_preds
                        )
    
    try:
        if hasattr(model, '_veto_stats') and model._veto_stats['correct_labels_rejected'] > 0:
            purity_ratio = model._veto_stats['true_errors_rejected'] / model._veto_stats['correct_labels_rejected']
            logger = logging.getLogger("EvalAdapt")
            logger.info(f"[Stats] Global Veto Purity Ratio (True Errors / Correct Rejected): {purity_ratio:.4f} (True Errors Rejected: {model._veto_stats['true_errors_rejected']}, Correct Rejected: {model._veto_stats['correct_labels_rejected']})")
            model._veto_stats = {'true_errors_rejected': 0, 'correct_labels_rejected': 0}
    except Exception as e:
        logger.warning(f"Diagnostic logging (Global Veto Purity) failed non-fatally: {e}")
        
    try:
        if hasattr(model, 'initial_classify_weights'):
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
            
            final_norms = {c: round(model.classify.weight[c].norm().item(), 4) for c in range(17)}
            logger.info(f"[Stats] Final Prototype Norms: {final_norms}")
    except Exception as e:
        logger.warning(f"Diagnostic logging (Prototype Rotation/Norms) failed non-fatally: {e}")

    try:
        if hasattr(model, '_class_true_errors_rejected') and not eval_only:
            logger = logging.getLogger("EvalAdapt")
            logger.info(f"\n[Stats] Per-Class Veto Purity (True Errors / Correct Labels Rejected):")
            err_np = model._class_true_errors_rejected.cpu().numpy()
            corr_np = model._class_correct_rejected.cpu().numpy()
            head_purity, tail_purity = {}, {}
            for c in range(num_classes):
                p = err_np[c] / corr_np[c] if corr_np[c] > 0 else -1.0
                if c in [11, 13, 14, 15, 16]: head_purity[c] = round(float(p), 2)
                elif c in [2, 3, 6, 7, 10]: tail_purity[c] = round(float(p), 2)
            logger.info(f"  Head Purity: {head_purity}")
            logger.info(f"  Tail Purity: {tail_purity}")
            model._class_true_errors_rejected = torch.zeros(num_classes, dtype=torch.long, device=device)
            model._class_correct_rejected = torch.zeros(num_classes, dtype=torch.long, device=device)
    except Exception as e:
        logger.warning(f"Diagnostic logging (Per-Class Veto Purity) failed non-fatally: {e}")
        
    try:
        if hasattr(model, '_class_n_points') and not eval_only:
            logger = logging.getLogger("EvalAdapt")
            pts_np = model._class_n_points.cpu().numpy()
            fir_np = model._class_n_fired.cpu().numpy()
            head_firing, tail_firing = {}, {}
            for c in range(num_classes):
                if pts_np[c] > 0:
                    val_str = f"{fir_np[c]}/{pts_np[c]} ({fir_np[c]/pts_np[c]*100:.2f}%)"
                    if c in [11, 13, 14, 15, 16]: head_firing[c] = val_str
                    elif c in [2, 3, 6, 7, 10]: tail_firing[c] = val_str
            logger.info(f"\n[Stats] Per-Class Firing Rates (True Fired/Total):")
            logger.info(f"  Head Firing: {head_firing}")
            logger.info(f"  Tail Firing: {tail_firing}")
            model._class_n_points = torch.zeros(num_classes, dtype=torch.long, device=device)
            model._class_n_fired = torch.zeros(num_classes, dtype=torch.long, device=device)
    except Exception as e:
        logger.warning(f"Diagnostic logging (Per-Class Firing Rates) failed non-fatally: {e}")

    try:
        if hasattr(model, '_contingency_table') and not eval_only:
            ct = model._contingency_table
            def get_prec(cell):
                return cell['correct'].item() / cell['n'].item() if cell['n'].item() > 0 else 0.0
            logger.info(f"\n[Section 3.2] GT-Labelled Contingency Table (geom vs epi admission):")
            logger.info(f"  Geom Admits / Epi Admits:  N={ct['geom_adm_epi_adm']['n'].item():,}, Prec={get_prec(ct['geom_adm_epi_adm']):.4f}")
            logger.info(f"  Geom Admits / Epi Rejects (RESCUE CELL): N={ct['geom_adm_epi_rej']['n'].item():,}, Prec={get_prec(ct['geom_adm_epi_rej']):.4f}")
            logger.info(f"  Geom Rejects / Epi Admits: N={ct['geom_rej_epi_adm']['n'].item():,}, Prec={get_prec(ct['geom_rej_epi_adm']):.4f}")
            logger.info(f"  Geom Rejects / Epi Rejects: N={ct['geom_rej_epi_rej']['n'].item():,}, Prec={get_prec(ct['geom_rej_epi_rej']):.4f}")
            model._contingency_table = {k: {'n': torch.tensor(0, dtype=torch.long, device=device), 'correct': torch.tensor(0, dtype=torch.long, device=device)} for k in ct}

        if hasattr(model, '_mv_contingency_table') and not eval_only and (mv_tta != 'none' or gate_mode == 'view_var_gate'):
            mct = model._mv_contingency_table
            def get_prec(cell):
                return cell['correct'].item() / cell['n'].item() if cell['n'].item() > 0 else 0.0
            logger.info(f"\n[Test D4] MV-2 Disagreement vs Epistemic Gating Contingency Table:")
            logger.info(f"  View Agree / Epi Admits:    N={mct['view_agree_epi_adm']['n'].item():,}, Prec={get_prec(mct['view_agree_epi_adm']):.4f}")
            logger.info(f"  View Agree / Epi Rejects (RESCUE CELL): N={mct['view_agree_epi_rej']['n'].item():,}, Prec={get_prec(mct['view_agree_epi_rej']):.4f}")
            logger.info(f"  View Disagree / Epi Admits: N={mct['view_disagree_epi_adm']['n'].item():,}, Prec={get_prec(mct['view_disagree_epi_adm']):.4f}")
            logger.info(f"  View Disagree / Epi Rejects:N={mct['view_disagree_epi_rej']['n'].item():,}, Prec={get_prec(mct['view_disagree_epi_rej']):.4f}")
            model._mv_contingency_table = {k: {'n': torch.tensor(0, dtype=torch.long, device=device), 'correct': torch.tensor(0, dtype=torch.long, device=device)} for k in mct}
    except Exception as e:
        logger.warning(f"Diagnostic logging (Contingency Table) failed non-fatally: {e}")
        
    try:
        if hasattr(model, '_decay_logs') and len(model._decay_logs['geom']) > 0 and not eval_only:
            g_all = torch.cat(model._decay_logs['geom']).float()
            e_all = torch.cat(model._decay_logs['epi']).float()
            gt_valid_all = torch.cat(model._decay_logs['gt_valid']).bool() if len(model._decay_logs['gt_valid']) > 0 else torch.zeros_like(g_all, dtype=torch.bool)
            gt_corr_all = torch.cat(model._decay_logs['gt_corr']).bool() if len(model._decay_logs['gt_corr']) > 0 else torch.zeros_like(g_all, dtype=torch.bool)
            cos10k_all = torch.cat(model._decay_logs['cos10k']).float() if len(model._decay_logs['cos10k']) > 0 else None
            cos128_all = torch.cat(model._decay_logs['cos128']).float() if len(model._decay_logs['cos128']) > 0 else None
            view_dis_all = torch.cat(model._decay_logs['view_dis']).bool() if len(model._decay_logs['view_dis']) > 0 else None
            
            def quantiles(t):
                if len(t) == 0: return "N/A"
                q = torch.quantile(t, torch.tensor([0.1, 0.5, 0.9]))
                return f"mean={t.mean().item():.4f}, median={q[1].item():.4f}, p10={q[0].item():.4f}, p90={q[2].item():.4f}"
                
            g_stats = quantiles(g_all)
            e_stats = quantiles(e_all)
            frac_zero = (g_all < 0.01).float().mean().item() * 100.0 if len(g_all) > 0 else 0.0
            
            # Pearson correlation
            if len(g_all) > 1:
                g_c = g_all - g_all.mean()
                e_c = e_all - e_all.mean()
                denom = torch.sqrt((g_c**2).sum() * (e_c**2).sum()) + 1e-8
                corr = (g_c * e_c).sum() / denom
                corr_val = corr.item()
            else:
                corr_val = 0.0
            
            logger.info(f"\n[Section 3.3] Decay Distribution Stats:")
            logger.info(f"  Geom Decay: {g_stats} | Fraction < 0.01: {frac_zero:.2f}%")
            logger.info(f"  Epi Decay:  {e_stats}")
            logger.info(f"  Pearson Correlation (geom vs epi): {corr_val:.4f}")
            
            # Test D0: Cosine similarity correlation (128D vs 10k) on random point pairs
            if cos10k_all is not None and cos128_all is not None and len(cos10k_all) > 1:
                logger.info(f"\n[Test D0] Cosine Similarity Correlation (128D Latent vs 10,000D HDC Space on Random Point Pairs):")
                logger.info(f"  {compute_correlations_torch(cos128_all, cos10k_all)}")
                
            # Test D1: Complementarity AUROC on Epistemic-Rejected subset (using raw un-saturated scores if available, else decays)
            if gt_valid_all.any():
                epi_rej_mask = gt_valid_all & (e_all < 0.5)
                if epi_rej_mask.any() and gt_corr_all[epi_rej_mask].unique().numel() > 1:
                    score_to_eval = torch.cat(model._decay_logs['geom_score']).float()[epi_rej_mask] if len(model._decay_logs.get('geom_score', [])) > 0 else g_all[epi_rej_mask]
                    auroc_val = compute_auroc_torch(score_to_eval, gt_corr_all[epi_rej_mask])
                    logger.info(f"\n[Test D1] Complementarity AUROC (geom score vs GT correctness on Epistemic-Rejected subset):")
                    logger.info(f"  AUROC = {auroc_val:.4f} (over {epi_rej_mask.sum().item():,} epistemic-rejected valid GT points)")
                else:
                    logger.info(f"\n[Test D1] Complementarity AUROC: N/A (no valid epistemic-rejected points or single class)")

                # Test D2: Matched-Rate Contingency Table (ranking on raw geom_score instead of saturated decay)
                epi_adm_rate = (e_all[gt_valid_all] >= 0.5).float().mean().item()
                g_score_to_eval = torch.cat(model._decay_logs['geom_score']).float() if len(model._decay_logs.get('geom_score', [])) > 0 else g_all
                th_geom = torch.quantile(g_score_to_eval[gt_valid_all], max(0.0, min(1.0, 1.0 - epi_adm_rate))).item()
                ga_matched = (g_score_to_eval[gt_valid_all] >= th_geom)
                ea_valid = (e_all[gt_valid_all] >= 0.5)
                corr_valid = gt_corr_all[gt_valid_all]
                
                def calc_cell(m_adm, m_epi):
                    mask = (m_adm & m_epi)
                    n = mask.sum().item()
                    prec = corr_valid[mask].float().mean().item() if n > 0 else 0.0
                    return n, prec

                n_aa, p_aa = calc_cell(ga_matched, ea_valid)
                n_ar, p_ar = calc_cell(ga_matched, ~ea_valid)
                n_ra, p_ra = calc_cell(~ga_matched, ea_valid)
                n_rr, p_rr = calc_cell(~ga_matched, ~ea_valid)
                
                logger.info(f"\n[Test D2] Matched-Rate Contingency Table (Geom th={th_geom:.4f} matching Epi adm_rate={epi_adm_rate*100:.1f}%):")
                logger.info(f"  Geom Admits / Epi Admits:  N={n_aa:,}, Prec={p_aa:.4f}")
                logger.info(f"  Geom Admits / Epi Rejects (RESCUE CELL): N={n_ar:,}, Prec={p_ar:.4f}")
                logger.info(f"  Geom Rejects / Epi Admits: N={n_ra:,}, Prec={p_ra:.4f}")
                logger.info(f"  Geom Rejects / Epi Rejects: N={n_rr:,}, Prec={p_rr:.4f}")

                # Test D7: Conditional Redundancy / Complementarity of MV-2 vs Geom on Epistemic-Rejected subset
                if view_dis_all is not None and epi_rej_mask.any():
                    vd_sub = view_dis_all[epi_rej_mask]
                    corr_sub = gt_corr_all[epi_rej_mask]
                    g_sub = g_all[epi_rej_mask]
                    
                    if corr_sub.unique().numel() > 1:
                        auroc_va = compute_auroc_torch((~vd_sub).float(), corr_sub)
                    else:
                        auroc_va = float('nan')
                        
                    vd_rej_mask = epi_rej_mask & view_dis_all
                    if vd_rej_mask.any() and gt_corr_all[vd_rej_mask].unique().numel() > 1:
                        score_cond = torch.cat(model._decay_logs['geom_score']).float()[vd_rej_mask] if len(model._decay_logs.get('geom_score', [])) > 0 else g_all[vd_rej_mask]
                        auroc_g_cond = compute_auroc_torch(score_cond, gt_corr_all[vd_rej_mask])
                    else:
                        auroc_g_cond = float('nan')
                        
                    logger.info(f"\n[Test D7] Conditional Redundancy (MV-2 View Agreement vs Geom on Epistemic-Rejected subset):")
                    logger.info(f"  AUROC (View Agreement on Epi-Rej): {auroc_va:.4f} (over {len(vd_sub):,} points)")
                    logger.info(f"  AUROC (Geom Score on Epi-Rej AND View-Disagree): {auroc_g_cond:.4f} (over {vd_rej_mask.sum().item():,} points)")
                    
                    va_mask = ~vd_sub
                    ga_mask = (g_sub >= 0.5)
                    def cell_stats(cond):
                        n = cond.sum().item()
                        p = corr_sub[cond].float().mean().item() if n > 0 else 0.0
                        return n, p
                    n_vaga, p_vaga = cell_stats(va_mask & ga_mask)
                    n_vagr, p_vagr = cell_stats(va_mask & ~ga_mask)
                    n_vdga, p_vdga = cell_stats(~va_mask & ga_mask)
                    n_vdgr, p_vdgr = cell_stats(~va_mask & ~ga_mask)
                    logger.info(f"  Contingency Table on Epistemic-Rejected subset (View Agree vs Geom Adm):")
                    logger.info(f"    View Agree / Geom Admits:  N={n_vaga:,}, Prec={p_vaga:.4f}")
                    logger.info(f"    View Agree / Geom Rejects: N={n_vagr:,}, Prec={p_vagr:.4f}")
                    logger.info(f"    View Disagree / Geom Admits (DOUBLE RESCUE): N={n_vdga:,}, Prec={p_vdga:.4f}")
                    logger.info(f"    View Disagree / Geom Rejects: N={n_vdgr:,}, Prec={p_vdgr:.4f}")

            model._decay_logs = {'geom': [], 'epi': [], 'geom_score': [], 'epi_score': [], 'gt_valid': [], 'gt_corr': [], 'cos128': [], 'cos10k': [], 'view_dis': [], 'margin': []}
    except Exception as e:
        logger.warning(f"Diagnostic logging (Decay Stats / Test D0-D2/D7) failed non-fatally: {e}")
        
    try:
        if dynamic_geom and hasattr(model, 'running_density_std') and hasattr(model, 'source_density_std') and not eval_only:
            ratio = (model.running_density_std / (model.source_density_std + 1e-8)).cpu().numpy()
            head_r = {c: round(float(ratio[c]), 2) for c in [11, 13, 14, 15, 16]}
            tail_r = {c: round(float(ratio[c]), 2) for c in [2, 3, 6, 7, 10]}
            logger.info(f"\n[Section 3.3 / Dynamic Geom] Final Variance Inflation Ratio (running_std / source_std):")
            logger.info(f"  Head Classes Ratio: {head_r}")
            logger.info(f"  Tail Classes Ratio: {tail_r}")
            if hasattr(model, 'running_density_mean') and hasattr(model, 'source_density_mean'):
                ratio_mean = (model.running_density_mean / (model.source_density_mean + 1e-8)).cpu().numpy()
                head_r_m = {c: round(float(ratio_mean[c]), 2) for c in [11, 13, 14, 15, 16]}
                tail_r_m = {c: round(float(ratio_mean[c]), 2) for c in [2, 3, 6, 7, 10]}
                logger.info(f"[Section 3.3 / Dynamic Geom] Final Mean Shift Ratio (running_mean / source_mean):")
                logger.info(f"  Head Classes Mean Ratio: {head_r_m}")
                logger.info(f"  Tail Classes Mean Ratio: {tail_r_m}")
    except Exception as e:
        logger.warning(f"Diagnostic logging (Dynamic Geom Ratio) failed non-fatally: {e}")
        
    try:
        if mv_tta != 'none' or gate_mode == 'view_var_gate':
            logger = logging.getLogger("EvalAdapt")
            logger.info(f"\n[MV-2] View Disagreement Precision Tracking")
            
            # Calculate overall precision for agreeing points
            agree_diag = torch.diag(agree_conf_matrix).sum().item()
            agree_total = agree_conf_matrix.sum().item()
            agree_precision = agree_diag / max(1, agree_total)
            
            # Calculate overall precision for disagreeing points
            disagree_diag = torch.diag(disagree_conf_matrix).sum().item()
            disagree_total = disagree_conf_matrix.sum().item()
            disagree_precision = disagree_diag / max(1, disagree_total)
            
            logger.info(f"  Agreeing Points Precision: {agree_precision:.4f} (Total: {agree_total})")
            logger.info(f"  Disagreeing Points Precision: {disagree_precision:.4f} (Total: {disagree_total})")
            
            # Tail classes Person (7), Bus (3), Truck (10)
            for t_class in [3, 7, 10]:
                tp_agree = agree_conf_matrix[t_class, t_class].item()
                fp_agree = agree_conf_matrix[:, t_class].sum().item() - tp_agree
                tp_disagree = disagree_conf_matrix[t_class, t_class].item()
                fp_disagree = disagree_conf_matrix[:, t_class].sum().item() - tp_disagree
                
                p_agree = tp_agree / max(1, tp_agree + fp_agree)
                p_disagree = tp_disagree / max(1, tp_disagree + fp_disagree)
                
                logger.info(f"  Class {t_class} Precision: Agreeing={p_agree:.4f} ({tp_agree} TP, {fp_agree} FP), Disagreeing={p_disagree:.4f} ({tp_disagree} TP, {fp_disagree} FP)")
    except Exception as e:
        logger.warning(f"Diagnostic logging (MV-2 Precision Tracking) failed non-fatally: {e}")
            
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
        "UpdateMagnitude": avg_update_magnitude,
        "ConfusionMatrix": cumulative_confusion_matrix.cpu().numpy().tolist()
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

def load_hdc_model(path, num_classes=NUM_CLASSES, mv_tta='none'):
    # print(f"Loading pretrained HDC model from {path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
    modeldir = os.path.dirname(path)

    if mv_tta != 'none':
        from modules.HDC_utils import set_mv_tta_model
        model = set_mv_tta_model(ARCH, modeldir, 'rp', 0, 0, num_classes, device, mv_tta=mv_tta)
    else:
        model = set_uq_model(ARCH, modeldir, 'rp', 0, 0, num_classes, device)
    
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.to(device)
    return model

def populate_source_statistics(model, data_dir, arch_cfg, data_cfg, device, dry_run=False):
    # print(f"Populating source statistics from {data_dir}...")
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
            if dry_run and batch_idx > 2:
                break
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
    all_latents_per_class = {c: [] for c in range(num_classes)}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Populating Source Stats")):
            if dry_run and batch_idx > 2:
                break
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
                norm_enc = F.normalize(raw_enc[valid_mask], dim=1).to(model.classify.weight.dtype)
                logits = model.classify(norm_enc)
                true_cos = logits[torch.arange(logits.size(0)), labels_valid]
                
                for c in range(num_classes):
                    c_mask = labels_valid == c
                    if c_mask.any():
                        all_dists_per_class[c].append(dists[c_mask].cpu())
                        all_cos_per_class[c].append(true_cos[c_mask].cpu())
                        all_latents_per_class[c].append(latent_valid[c_mask].cpu())
    
    model.source_density_mean = torch.zeros(num_classes, device=device)
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
        
    global_dist_mean = torch.cat(global_dists, dim=0).mean().item()
    global_dist_std = torch.cat(global_dists, dim=0).std().item()
    global_cos_tensor = torch.cat(global_cos, dim=0)
    global_cos_mean = global_cos_tensor.mean().item()
    global_cos_std = global_cos_tensor.std().item()
    
    source_bank_list = []
    for c in range(num_classes):
        if len(all_dists_per_class[c]) > 0:
            c_dists = torch.cat(all_dists_per_class[c], dim=0)
            model.source_density_mean[c] = c_dists.mean().item()
            model.source_density_std[c] = c_dists.std().item()
            c_cos = torch.cat(all_cos_per_class[c], dim=0)
            model.source_mu_cos[c] = c_cos.mean().item()
            model.source_sigma_cos[c] = c_cos.std().item()
        else:
            # Fallback to global statistics if class is completely missing from the first 50 frames
            model.source_density_mean[c] = global_dist_mean
            model.source_density_std[c] = global_dist_std
            model.source_mu_cos[c] = global_cos_mean
            model.source_sigma_cos[c] = global_cos_std
        if len(all_latents_per_class[c]) > 0:
            c_latents = torch.cat(all_latents_per_class[c], dim=0)
            if len(c_latents) > 50:
                perm = torch.randperm(len(c_latents))[:50]
                source_bank_list.append(c_latents[perm])
            else:
                source_bank_list.append(c_latents)
                
    model.source_bank = torch.cat(source_bank_list, dim=0).to(device) if len(source_bank_list) > 0 else None
    model.source_class_freq = (class_latent_counts / class_latent_counts.sum()).cpu()
    return {
        'source_density_mean': model.source_density_mean,
        'source_density_std': model.source_density_std,
        'source_mu_cos': model.source_mu_cos,
        'source_sigma_cos': model.source_sigma_cos,
        'drift_mu_0': model.drift_mu_0.clone().cpu(),
        'source_class_freq': model.source_class_freq,
        'source_bank': model.source_bank.cpu() if model.source_bank is not None else None
    }

def main():
    import random
    parser = argparse.ArgumentParser(description="Test Unsupervised Updates on KITTI-C")
    parser.add_argument('--pretrain', action='store_true', help='Run pretraining on SemanticKITTI before evaluating')
    parser.add_argument('--chunked', action='store_true', help='Use chunked protocol: continuous adaptation across disjoint 1/7th splits instead of full independent sequences.')
    parser.add_argument('--reset_per_corruption', action='store_true', help='Reset the model to the clean pretrained weights before adapting on each corruption (requires --chunked).')
    parser.add_argument('--continual', action='store_true', help='Continual learning mode: continuous adaptation across sequences without resetting.')
    parser.add_argument('--pretrained_path', type=str, default='logs/kitti_pretrain/hdc_sub.pth', help='Path to load pretrained model')
    parser.add_argument('--log_dir', type=str, default='logs/kitti_c_test', help='Directory to save logs and graphics')
    parser.add_argument('--method', type=str, default='frozen', help='Method to test.')
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
    parser.add_argument('--corruptions', type=str, default='fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor', help='Comma separated list of corruptions to test. Defaults to all 8 corruptions.')

    parser.add_argument('--ic_method', type=str, default='none', help='Inter/Intra-Class balancing method: none, ic4')
    parser.add_argument('--tau', type=float, default=None, help='Logit adjustment tau for inference prior. If not set, BM is used. τ=0 is normalized baseline. τ<0 is majority amplifier. τ>0 is balanced softmax.')
    parser.add_argument('--kappa', type=float, default=15.0, help='Logit scale kappa used with tau. Controls evidence weighting vs prior.')
    parser.add_argument('--normalize_weights', action='store_true', help='Force weights to be normalized after every update (disables Bayesian Momentum accumulator).')
    parser.add_argument('--mv_tta', type=str, default='none', help='Multi-View TTA method. Options: none, conf_pred, veto_disagree')
    parser.add_argument('--gate_mode', type=str, default='epistemic', help='Uncertainty gating strategy: epistemic, geometric, and_gate, or_gate, oracle, rescue_gate, ellipsoid_gate, soft_dual_weight, view_var_gate')
    parser.add_argument('--dynamic_geom', action='store_true', help='Use running batch EMA variance for geometric HDC density thresholding.')
    parser.add_argument('--no_diagnostics', action='store_false', dest='diagnostics', help='Disable heavy diagnostic tracking.')
    parser.add_argument('--dump_features', action='store_true', default=False, help='Dump per-point feature tensors for Test D5/D6 offline probe.')
    parser.set_defaults(diagnostics=True)
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
    methods_to_run = [m.strip() for m in args.method.split(',')]
    
    global_results_path = os.path.join(args.log_dir, 'global_results.json')
    global_results = None
    if os.path.exists(global_results_path):
        try:
            with open(global_results_path, 'r') as f:
                global_results = json.load(f)
        except json.JSONDecodeError:
            global_results = None
            
    full_method_names = []
    for m in methods_to_run:
        if m == 'frozen':
            if args.tau is not None and args.tau != 0.0:
                full_method_names.append(f"frozen_tau_{args.tau}")
            else:
                full_method_names.append('frozen')
        elif m in ['conformalhdc', 'hyperdum', 'd3ctta']:
            full_method_names.append(m)
        else:
            m_name = f"{m}_{args.ic_method}_tau_{args.tau}_mv_{args.mv_tta}"
            if args.gate_mode != 'epistemic':
                m_name += f"_gate_{args.gate_mode}"
            if args.dynamic_geom:
                m_name += "_dyn"
            full_method_names.append(m_name)
                    
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
    # Note on Protocol: Divides the valid set into 7 disjoint chunks (1 per corruption).
    # This evaluates each corruption on 1/7 of the validation set (e.g., ~581 frames) instead 
    # of the full set. We are preserving this behavior to identically match the 3-chunk protocol. 
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

    base_model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES, mv_tta=args.mv_tta)
    populate_source_statistics(base_model, args.kitti_dir, ARCH, DATA, device, dry_run=args.dry_run)
    
    source_stats_cache = {
        'class_latent_means': base_model.class_latent_means,
        'source_density_mean': getattr(base_model, 'source_density_mean', None),
        'source_density_std': getattr(base_model, 'source_density_std', None),
        'source_mu_cos': getattr(base_model, 'source_mu_cos', None),
        'source_sigma_cos': getattr(base_model, 'source_sigma_cos', None),
        'drift_mu_0': getattr(base_model, 'drift_mu_0', None),
        'source_class_freq': getattr(base_model, 'source_class_freq', None),
        'source_bank': getattr(base_model, 'source_bank', None)
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
    model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES, mv_tta=args.mv_tta)
    if source_stats_cache is not None:
        model.class_latent_means = source_stats_cache['class_latent_means'].to(device) if source_stats_cache['class_latent_means'] is not None else None
        model.source_density_mean = source_stats_cache['source_density_mean'].to(device) if source_stats_cache.get('source_density_mean') is not None else None
        model.source_density_std = source_stats_cache['source_density_std'].to(device) if source_stats_cache['source_density_std'] is not None else None
        model.source_mu_cos = source_stats_cache['source_mu_cos'].to(device) if source_stats_cache['source_mu_cos'] is not None else None
        model.source_sigma_cos = source_stats_cache['source_sigma_cos'].to(device) if source_stats_cache['source_sigma_cos'] is not None else None
        model.drift_mu_0 = source_stats_cache['drift_mu_0'].to(device) if source_stats_cache['drift_mu_0'] is not None else None
        model.source_class_freq = source_stats_cache['source_class_freq'].to(device) if source_stats_cache['source_class_freq'] is not None else None
        model.source_bank = source_stats_cache['source_bank'].to(device) if source_stats_cache.get('source_bank') is not None else None

    for current_method, full_method_name in zip(methods_to_run, full_method_names):
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
        attrs_to_del = ['drift_mu_c', 'class_freq_ema', 'class_update_counts', 'class_M', 'running_density_std', 'running_density_mean', '_contingency_table', '_mv_contingency_table', '_decay_logs', '_class_n_points', '_class_n_fired', '_class_true_errors_rejected', '_class_correct_rejected', '_firing_log', '_veto_stats', '_update_magnitude_log', 'initial_classify_weights', '_feature_dump_list', '_baseline_adapter', '_baseline_adapter_name']
        for attr in attrs_to_del:
            if hasattr(model, attr):
                delattr(model, attr)
            
        eval_model = model

        for i, ctype in enumerate(active_corruptions):
            if args.reset_per_corruption and args.chunked and not args.continual:
                logger.info("Resetting model to clean pretrained weights for this corruption.")
                model.load_state_dict(clean_state_dict, strict=False)
                attrs_to_del = ['drift_mu_c', 'class_freq_ema', 'class_update_counts', 'class_M', 'running_density_std', 'running_density_mean', '_contingency_table', '_mv_contingency_table', '_decay_logs', '_class_n_points', '_class_n_fired', '_class_true_errors_rejected', '_class_correct_rejected', '_firing_log', '_veto_stats', '_update_magnitude_log', 'initial_classify_weights', '_feature_dump_list', '_baseline_adapter', '_baseline_adapter_name']
                for attr in attrs_to_del:
                    if hasattr(model, attr):
                        delattr(model, attr)
                
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
            
            chunk_dataset = None
            if not args.chunked:
                # Standard protocol: full sequence, independent adaptation
                chunk_dataset = full_corruption_dataset
                if not args.continual:
                    # Reset model before each corruption
                    model.load_state_dict(clean_state_dict, strict=False)
                    attrs_to_del = ['drift_mu_c', 'class_freq_ema', 'class_update_counts', 'class_M', 'running_density_std', 'running_density_mean', '_contingency_table', '_mv_contingency_table', '_decay_logs', '_class_n_points', '_class_n_fired', '_class_true_errors_rejected', '_class_correct_rejected', '_firing_log', '_veto_stats', '_update_magnitude_log', 'initial_classify_weights', '_feature_dump_list', '_baseline_adapter', '_baseline_adapter_name']
                    for attr in attrs_to_del:
                        if hasattr(model, attr):
                            delattr(model, attr)
            else:
                # Chunked protocol: continuous adaptation across disjoint splits
                chunk_dataset = torch.utils.data.Subset(full_corruption_dataset, chunks[i])
            
            assert chunk_dataset is not None, "chunk_dataset was not assigned!"
            target_dataloader = DataLoader(chunk_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
            
            try:
                # Always run 3-pass protocol to get True Initial and True Final
                init_key = (ctype, sev, current_method, args.tau, args.ic_method, args.kappa, args.mv_tta, args.gate_mode, args.chunked)
                if not args.continual and (not args.chunked or args.reset_per_corruption) and init_key in shared_init_metrics:
                    logger.debug("  -> Pass 1: Reusing cached True Initial metrics (Frozen)")
                    init_metrics = shared_init_metrics[init_key]
                else:
                    logger.debug("  -> Pass 1: Computing True Initial metrics (Frozen)")
                    init_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=True, update_method=current_method, dry_run=args.dry_run, ic_method=args.ic_method, tau=args.tau, kappa=args.kappa, normalize_weights=args.normalize_weights, mv_tta=args.mv_tta, gate_mode=args.gate_mode, dynamic_geom=args.dynamic_geom, dump_features=args.dump_features, diagnostics=args.diagnostics)
                    if not args.continual and (not args.chunked or args.reset_per_corruption):
                        shared_init_metrics[init_key] = init_metrics
                
                # Pass 2: Adapt (only if method is not frozen)
                if current_method != 'frozen':
                    logger.debug("  -> Pass 2: Adapting model weights")
                    eval_model.train()
                    adapt_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=False, update_method=current_method, dry_run=args.dry_run, ic_method=args.ic_method, tau=args.tau, kappa=args.kappa, normalize_weights=args.normalize_weights, mv_tta=args.mv_tta, gate_mode=args.gate_mode, dynamic_geom=args.dynamic_geom, dump_features=args.dump_features, diagnostics=args.diagnostics)
                else:
                    adapt_metrics = init_metrics
                    
                # Pass 3: True Final (Frozen on chunk using adapted weights)
                logger.debug("  -> Pass 3: Computing True Final metrics (Frozen)")
                eval_model.eval()
                final_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=True, update_method=current_method, dry_run=args.dry_run, ic_method=args.ic_method, tau=args.tau, kappa=args.kappa, normalize_weights=args.normalize_weights, mv_tta=args.mv_tta, gate_mode=args.gate_mode, dynamic_geom=args.dynamic_geom, dump_features=args.dump_features, diagnostics=args.diagnostics)
                
                metrics = adapt_metrics  # Just for the trajectory json
                if len(init_metrics["mIoU"]) > 0:
                    initial_miou = init_metrics["mIoU"][-1]
                    final_miou = final_metrics["mIoU"][-1]
                    online_miou = adapt_metrics["mIoU"][-1]
                    initial_acc = init_metrics["Accuracy"][-1]
                    final_acc = final_metrics["Accuracy"][-1]
                    
                    if 'ConfusionMatrix' in init_metrics:
                        cm_init = np.array(init_metrics['ConfusionMatrix'])
                        tp_init = np.diag(cm_init)
                        fp_init = cm_init.sum(axis=0) - tp_init
                        fn_init = cm_init.sum(axis=1) - tp_init
                        
                        if 'ConfusionMatrix' in final_metrics:
                            cm_fin = np.array(final_metrics['ConfusionMatrix'])
                            tp_fin = np.diag(cm_fin)
                            fp_fin = cm_fin.sum(axis=0) - tp_fin
                            fn_fin = cm_fin.sum(axis=1) - tp_fin
                        else:
                            tp_fin, fp_fin, fn_fin = tp_init, fp_init, fn_init
                            
                        tail_classes = [2, 3, 6, 7, 10]
                        logger.info(f"  -> Initial Tail TP:  {tp_init[tail_classes].tolist()} | Final Tail TP:  {tp_fin[tail_classes].tolist()} (Delta: {(tp_fin - tp_init)[tail_classes].tolist()})")
                        logger.info(f"  -> Initial Tail FP:  {fp_init[tail_classes].tolist()} | Final Tail FP:  {fp_fin[tail_classes].tolist()} (Delta: {(fp_fin - fp_init)[tail_classes].tolist()})")
                        logger.info(f"  -> Initial Tail FN:  {fn_init[tail_classes].tolist()} | Final Tail FN:  {fn_fin[tail_classes].tolist()} (Delta: {(fn_fin - fn_init)[tail_classes].tolist()})")
                else:
                    initial_miou = final_miou = online_miou = initial_acc = final_acc = 0.0
                    
                firing_rate_str = ""
                if "FiringRate" in adapt_metrics:
                    firing_rate_str = f", FiringRate={adapt_metrics['FiringRate']*100:.2f}%"
                    if "UpdateMagnitude" in adapt_metrics:
                        firing_rate_str += f", UpdateMag={adapt_metrics['UpdateMagnitude']:.4f}"
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
                
                initial_head = init_metrics["Head_mIoU"][-1]
                final_head = final_metrics["Head_mIoU"][-1]
                initial_mid = init_metrics["Mid_mIoU"][-1]
                final_mid = final_metrics["Mid_mIoU"][-1]
                initial_tail = init_metrics["Tail_mIoU"][-1]
                final_tail = final_metrics["Tail_mIoU"][-1]
                
                protocol_str = "continual" if args.continual else ("chunked" if args.chunked else "full")
                n_frames_str = len(target_dataloader)
                logger.info(f"Result for {ctype}-{sev} [protocol={protocol_str}, n_frames={n_frames_str}]: Initial mIoU={initial_miou:.4f} -> Final (Online)={online_miou:.4f} -> Final (Frozen)={final_miou:.4f} (Head: {initial_head:.4f} -> {final_head:.4f}, Mid: {initial_mid:.4f} -> {final_mid:.4f}, Tail: {initial_tail:.4f} -> {final_tail:.4f}), Acc={initial_acc:.4f} -> {final_acc:.4f}{firing_rate_str}")
                suffix = f"_{full_method_name}"
                
                traj_json_path = os.path.join(args.log_dir, f'traj_{ctype}_{sev}{suffix}.json')
                with open(traj_json_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                    
                save_graphic(os.path.join(args.log_dir, f'traj_{ctype}_{sev}{suffix}.png'), f'{ctype} Sev {sev}', metrics)
                
                with open(os.path.join(args.log_dir, f'results{suffix}.json'), 'w') as f:
                    json.dump({'mIoU': results_miou, 'Accuracy': results_acc}, f, indent=4)
                    
                with open(os.path.join(args.log_dir, 'global_results.json'), 'w') as f:
                    json.dump(global_results, f, indent=4)
                    
                if args.dump_features and hasattr(model, '_feature_dump_list') and len(model._feature_dump_list) > 0:
                    dump_path = os.path.join(args.log_dir, f'features_dump_{ctype}_{sev}{suffix}.pt')
                    logger.info(f"Saving feature dump ({len(model._feature_dump_list)} frames) to {dump_path}...")
                    torch.save(model._feature_dump_list, dump_path)
                    model._feature_dump_list = []
            else:
                logger.info(f"No valid frames evaluated for {ctype}-{sev}")

        total_evals = sum(len(sev_dict) for sev_dict in results_miou.values())
        if total_evals == 0 and not args.dry_run:
            logger.error(f"CRITICAL ERROR: No evaluation outcomes recorded in results_miou for {full_method_name}. Check for silent exceptions during evaluation.")
            raise RuntimeError(f"Evaluation failed to record any metrics for {full_method_name}.")

        suffix = f"_{full_method_name}"
        save_degradation_plot(os.path.join(args.log_dir, f'degradation_miou{suffix}.png'), 'KITTI-C', results_miou, metric='mIoU', baseline_val=None)
        save_degradation_plot(os.path.join(args.log_dir, f'degradation_acc{suffix}.png'), 'KITTI-C', results_acc, metric='Accuracy', baseline_val=None)

if __name__ == "__main__":
    main()
