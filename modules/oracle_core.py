"""Core HDC prototype evaluation (frozen-prototype decode, gated EMA ladder,
condition autopsy). Extracted from oracle_gating_eval.py for maintainability.
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from modules.HDC_utils import fuse_uncertainties

def get_hdc_projection(dim_in=128, dim_out=10000, device='cuda'):
    torch.manual_seed(42)
    proj = (torch.rand(dim_in, dim_out) > 0.5).float() * 2 - 1
    return proj.to(device)

def build_hdc_prototypes(feats_128, lbls, proj, num_classes=17, device='cuda', chunk_size=50000):
    protos = torch.zeros(num_classes, proj.shape[1], device=device)
    counts = torch.zeros(num_classes, device=device)
    
    for i in range(0, len(feats_128), chunk_size):
        chunk_f = feats_128[i:i+chunk_size].to(device)
        chunk_l = lbls[i:i+chunk_size].to(device)
        
        h_chunk = torch.sign(torch.matmul(chunk_f, proj))
        
        for c in range(num_classes):
            mask = chunk_l == c
            if mask.sum() > 0:
                protos[c] += h_chunk[mask].sum(dim=0)
                counts[c] += mask.sum()
                
    for c in range(num_classes):
        if counts[c] > 0:
            protos[c] /= counts[c]
            
    base_protos = F.normalize(protos, p=2, dim=1)
    proto_lbls = torch.arange(num_classes, device=device)
    
    # Filter out empty classes
    valid_mask = counts > 0
    return base_protos[valid_mask], proto_lbls[valid_mask]

def weighted_mean_update(base_protos, proto_lbls, pool_f_128, pool_pseudo, weights, proj,
                         device, mask=None, chunk_size=50000):
    """Chunked weighted class-mean prototype update (vectorized).

    Prototype_c = normalize( sum over pool points with pseudo-label c of w_i * sign(z_i @ proj) )

    Replaces the sequential EMA ladder: Phase 13 showed that with a small pool and
    constant alpha, prototypes get erased/re-estimated from ~1/alpha points, and
    that a 20k-point pool is far too small to refine 10kD prototypes whose base
    estimates come from millions of points. A large-pool weighted mean is the
    statistically honest adaptation operator, and chunked index_add keeps it fast.
    Classes with no pool support keep the base prototype.
    """
    num_proto = len(proto_lbls)
    D = proj.shape[1]
    S = torch.zeros(num_proto, D, device=device)
    W = torch.zeros(num_proto, device=device)
    for start in range(0, len(pool_f_128), chunk_size):
        end = min(start + chunk_size, len(pool_f_128))
        chunk = pool_f_128[start:end].to(device)
        pl = pool_pseudo[start:end]
        cw = weights[start:end].to(device)
        if mask is not None:
            cw = cw * mask[start:end].to(device)
        h = torch.sign(torch.matmul(chunk, proj))  # [B, 10000]
        valid = torch.isin(pl, proto_lbls)
        idx = torch.searchsorted(proto_lbls, pl)
        S.index_add_(0, idx[valid], (cw[valid].unsqueeze(1) * h[valid]).float())
        W.index_add_(0, idx[valid], cw[valid].float())
    empty = W <= 0
    S = F.normalize(S, p=2, dim=1)
    S[empty] = F.normalize(base_protos[empty], p=2, dim=1)
    return S

def eval_protos(protos, proto_lbls, val_feats, val_lbls):
    sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), protos.T)
    preds = proto_lbls[sims.argmax(dim=1)]
    return (preds == val_lbls).float().mean().item()

def condition_autopsy(base_protos, proto_lbls, clean_means128, corrupt_feats, corrupt_lbls, proj,
                      device, clf=None, clean_stats=None, corrupt_depths=None, seed=42):
    """Per-condition hyperspace + decode signature (Phase 24).

    Quantifies what separates the stuck conditions (fog/crosstalk) from the
    geometric corruptions:
      - decode: zero-shot acc/mIoU + per-class IoU
      - artifact profile (Phase 22.2 signature): of the misclassified points, how
        many are confident artifacts (high norm / low cos-to-true / high
        perceptron loss / small margin) vs boundary-recoverable
      - margin statistics: correct vs misclassified
      - norm statistics: correct vs misclassified, near-origin fraction
      - cosine shift (128D clean->corrupt class means)
      - ellipticity (top-eigenvalue/trace of the corrupt manifold)
      - linear probe acc (representation headroom)
      - binarized (10kD) class-mean quality: norm ratio + clean<->corrupt cos sim
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    val_idx = perm[-100000:]
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = torch.sign(val_f @ proj)
    norms = torch.norm(val_f, p=2, dim=1)
    sims = F.normalize(val_h, p=2, dim=1) @ F.normalize(base_protos, p=2, dim=1).T
    top2 = torch.topk(sims, 2, dim=1)
    margin = (top2.values[:, 0] - top2.values[:, 1]).clamp(min=0)
    ti = torch.searchsorted(proto_lbls, val_l)
    cos_true = sims[torch.arange(len(val_l), device=device), ti]
    loss = (top2.values[:, 0] - cos_true).clamp(min=0)
    preds = proto_lbls[top2.indices[:, 0]]
    correct = preds == val_l

    acc = float(correct.float().mean().item())
    miou = compute_miou(preds, val_l)

    # per-class IoU
    per_class = {}
    present = set(val_l.tolist())
    for c in range(1, 17):
        if c not in present:
            continue
        tp = int(((preds == c) & (val_l == c)).sum().item())
        fp = int(((preds == c) & (val_l != c)).sum().item())
        fn = int(((preds != c) & (val_l == c)).sum().item())
        d = tp + fp + fn
        per_class[c] = tp / d if d > 0 else 0.0

    # artifact profile on misclassified points
    mis = ~correct
    n_mis = int(mis.sum().item())
    f_norm = int((mis & (norms < 6.0)).sum().item())
    f_true = int((mis & (norms < 6.0) & (cos_true >= 0.05)).sum().item())
    f_loss = int((mis & (norms < 6.0) & (cos_true >= 0.05) & (loss <= 0.15)).sum().item())
    f_margin = int((mis & (norms < 6.0) & (cos_true >= 0.05) & (loss <= 0.15)
                    & (margin >= 0.02)).sum().item())
    conf_artifact = int((mis & (loss > 0.15)).sum().item())

    # margin / norm statistics
    mar_c = float(margin[correct].mean().item()) if correct.any() else 0.0
    mar_m = float(margin[mis].mean().item()) if n_mis else 0.0
    norm_c = float(norms[correct].mean().item()) if correct.any() else 0.0
    norm_m = float(norms[mis].mean().item()) if n_mis else 0.0
    near_origin = float((norms < 4.0).float().mean().item())

    # cosine shift (128D): clean means vs corrupt class means
    cm = {c: clean_means128[c].to(device) for c in clean_means128}
    shifts = []
    for c in sorted(cm):
        fm = val_f[val_l == c]
        if len(fm) >= 500:
            fmu = F.normalize(fm.mean(dim=0), p=2, dim=0)
            shifts.append(1.0 - float(F.cosine_similarity(cm[c].unsqueeze(0), fmu.unsqueeze(0)).item()))
    cos_shift = float(np.mean(shifts)) if shifts else 0.0

    # ellipticity (subsample 20k)
    def ellipticity(x):
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / len(x)
        eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        tr = eig.sum()
        return float((eig[-1] / (tr + 1e-8)).item()) if tr > 1e-8 else 0.0
    sub = val_f[:20000]
    ellipt = ellipticity(sub) if len(sub) >= 500 else 0.0

    # linear probe (representation headroom) + its DECODE mIoU (learned-decoder ceiling)
    lp = 0.0
    lp_miou = 0.0
    lp_per_class = {}
    if clf is not None:
        n_lp = min(50000, len(val_f))
        pl = torch.tensor(clf.predict(val_f.cpu().numpy()[:n_lp])).to(device)
        ll = val_l[:n_lp]
        lp = float((pl == ll).float().mean().item())
        lp_miou = compute_miou(pl, ll)
        present = set(ll.tolist())
        for c in range(1, 17):
            if c not in present:
                continue
            tp = int(((pl == c) & (ll == c)).sum().item())
            fp = int(((pl == c) & (ll != c)).sum().item())
            fn = int(((pl != c) & (ll == c)).sum().item())
            d = tp + fp + fn
            lp_per_class[c] = tp / d if d > 0 else 0.0

    # per-class poison-band structure: fraction of each class's points with norm >= 4
    poison_band_frac = {}
    poison_band_acc = {}
    for c in range(1, 17):
        cm = val_l == c
        if cm.sum() < 500:
            continue
        idx = cm.nonzero(as_tuple=False).view(-1)
        pb = norms[idx] >= 4.0
        poison_band_frac[c] = float(pb.float().mean().item())
        if pb.sum() >= 100:
            pb_preds = preds[idx][pb]
            pb_lbl = val_l[idx][pb]
            poison_band_acc[c] = float((pb_preds == pb_lbl).float().mean().item())
        else:
            poison_band_acc[c] = None

    # binarized class-mean quality (10kD)
    bcs = []
    for c in proto_lbls.tolist():
        m = (val_l == c)
        if m.sum() >= 500:
            bh = torch.sign(val_f[m][:20000] @ proj).float().mean(dim=0)
            bcs.append(bh)
    if bcs:
        bm = torch.stack(bcs)
        bn = F.normalize(bm, p=2, dim=1)
        bm_clean = F.normalize(base_protos, p=2, dim=1)
        cs = float((bn @ bm_clean.T).diag().mean().item())
        bm_norm = float(bm.norm(dim=1).mean().item())
        cl_norm = float(base_protos.norm(dim=1).mean().item())
        binarized_cos = cs
        binarized_ratio = bm_norm / max(cl_norm, 1e-8)
    else:
        binarized_cos, binarized_ratio = 0.0, 0.0

    # Range/depth correlation (the far-field destruction hypothesis, Phase 24.2).
    # NOTE: the fog range channel is NOT calibrated meters (values ~4-7, negatives),
    # so the near/far split is RELATIVE (median of the masked depths), encoding-agnostic.
    depth_stats = {}
    if corrupt_depths is not None:
        dep = corrupt_depths[val_idx].to(device)
        depth_stats['norm_depth_corr'] = float(torch.corrcoef(
            torch.stack([norms, dep.float()]))[0, 1].item())
        depth_stats['far_split'] = 'relative(median)'
        med = torch.median(dep)
        per_class_depth = {}
        for c in sorted(present):
            cm = val_l == c
            if cm.sum() < 500:
                continue
            far = dep[cm] >= med
            n_far = int(far.sum().item())
            row = {'mean_depth': float(dep[cm].mean().item()),
                   'far_frac': float(far.float().mean().item())}
            if n_far >= 100 and (len(far) - n_far) >= 100:
                cp = preds[cm]
                cl = val_l[cm]
                row['near_acc'] = float((cp[~far] == cl[~far]).float().mean().item())
                row['far_acc'] = float((cp[far] == cl[far]).float().mean().item())
            else:
                row['near_acc'] = None
                row['far_acc'] = None
            per_class_depth[c] = row
        depth_stats['per_class'] = per_class_depth

    # BN-style test-time statistic alignment (the D3CTTA mechanism on our encoder):
    # align the corrupt features' per-dimension mean/std to the clean statistics,
    # then re-run the 10kD prototype decode.
    align_acc, align_miou = None, None
    if clean_stats is not None:
        cmean, cstd = clean_stats[0].to(device), clean_stats[1].to(device)
        fmean = val_f.mean(dim=0)
        fstd = val_f.std(dim=0) + 1e-6
        aligned = (val_f - fmean) / fstd * cstd + cmean
        ah = torch.sign(aligned @ proj)
        asims = F.normalize(ah, p=2, dim=1) @ F.normalize(base_protos, p=2, dim=1).T
        apreds = proto_lbls[asims.argmax(dim=1)]
        align_acc = float((apreds == val_l).float().mean().item())
        align_miou = compute_miou(apreds, val_l)

    return {
        'acc': acc, 'miou': miou, 'per_class_iou': per_class,
        'n_mis': n_mis, 'conf_artifact_frac': conf_artifact / max(n_mis, 1),
        'artifact_survivors': (f_margin, f_loss, f_true, f_norm, n_mis),
        'margin_correct': mar_c, 'margin_mis': mar_m,
        'norm_correct': norm_c, 'norm_mis': norm_m, 'near_origin': near_origin,
        'cos_shift': cos_shift, 'ellipticity': ellipt, 'lp_acc': lp,
        'lp_miou': lp_miou, 'lp_per_class': lp_per_class,
        'poison_band_frac': poison_band_frac, 'poison_band_acc': poison_band_acc,
        'binarized_cos': binarized_cos, 'binarized_ratio': binarized_ratio,
        'align_acc': align_acc, 'align_miou': align_miou,
        'depth_stats': depth_stats,
    }

def eval_protos_miou(protos, proto_lbls, val_feats, val_lbls):
    """Point accuracy AND mIoU (classes present in labels; class 0 ignored)."""
    sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), protos.T)
    preds = proto_lbls[sims.argmax(dim=1)]
    acc = float((preds == val_lbls).float().mean().item())
    return acc, compute_miou(preds, val_lbls)

def compute_miou(preds, lbls, num_classes=17):
    """Mean IoU over evaluated classes (class 0 ignored; absent classes excluded)."""
    present = set(lbls.tolist())
    ious = []
    for c in range(1, num_classes):
        if c not in present:
            continue
        tp = int(((preds == c) & (lbls == c)).sum().item())
        fp = int(((preds == c) & (lbls != c)).sum().item())
        fn = int(((preds != c) & (lbls == c)).sum().item())
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
    return float(np.mean(ious)) if ious else 0.0

def compute_signal_aurocs(meta_list):
    """AUROC of each gate signal for separating Helpful (delta > 0) from Harmful (delta < 0) updates."""
    if len(meta_list) < 10:
        return {}
    confs = np.array([m['conf'] for m in meta_list], dtype=np.float64)
    norms = np.array([m['norm'] for m in meta_list], dtype=np.float64)
    deltas = np.array([m['delta'] for m in meta_list])
    y = (deltas > 0).astype(int)
    if len(np.unique(y)) < 2:
        return {}
    c_z = (confs - confs.mean()) / (confs.std() + 1e-8)
    n_z = (norms - norms.mean()) / (norms.std() + 1e-8)
    aucs = {
        'conf': roc_auc_score(y, confs),
        'norm': roc_auc_score(y, -n_z),          # higher norm -> harmful
        'joint_z': roc_auc_score(y, c_z - n_z),  # Phase 11 proposal
    }
    try:
        lr = LogisticRegression(max_iter=1000).fit(np.stack([confs, norms], axis=1), y)
        aucs['lr'] = roc_auc_score(y, lr.decision_function(np.stack([confs, norms], axis=1)))
    except Exception:
        aucs['lr'] = None
    return aucs

def evaluate_oracle_gating(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, clf, proj,
                           device='cuda', pool_size=1000000, gate_cfg=None):
    c_lbl = corrupt_lbls.to(device)
    
    # Get pseudo-labels and confidence for Fog points (in 128D)
    print("      -> Running Probe Inference (128D)...")
    corrupt_probs = clf.predict_proba(corrupt_feats.numpy())
    corrupt_pseudo_lbls = torch.tensor(corrupt_probs.argmax(axis=1)).to(device)
    corrupt_confidences = torch.tensor(corrupt_probs.max(axis=1)).to(device)
    
    # Extract sets to avoid OOM.
    # FIX (Phase 13): pool = first 20k points and val = last 100k points were ~98
    # frames apart (different scenes), so even the ground-truth oracle LOST to
    # zero-shot on every corruption. Both sets now come from one seeded uniform
    # permutation over all points, so adaptation and evaluation share the same
    # frame distribution. The whole pipeline is seeded in main() for reproducibility.
    val_size = 100000
    torch.manual_seed(42)
    perm = torch.randperm(len(corrupt_feats))
    pool_idx = perm[:pool_size]
    val_idx = perm[-val_size:]
    
    pool_f_128 = corrupt_feats[pool_idx].to(device)
    pool_lbls = c_lbl[pool_idx]
    pool_pseudo = corrupt_pseudo_lbls[pool_idx]
    pool_conf = corrupt_confidences[pool_idx]
    
    val_f_128 = corrupt_feats[val_idx].to(device)
    val_lbls = c_lbl[val_idx]
    
    print("      -> Projecting Validation Set to 10kD HDC...")
    val_feats = torch.sign(torch.matmul(val_f_128, proj))
    
    # Project only the leave-one-out subset (5k) to 10kD; the ladder projects its
    # large pool chunk-by-chunk inside weighted_mean_update to bound GPU memory.
    print("      -> Projecting Leave-One-Out Pool (5k) to 10kD HDC...")
    lou_size = min(5000, len(pool_f_128))
    pool_feats = torch.sign(torch.matmul(pool_f_128[:lou_size], proj))
    
    # Gate signals (z-scored over the pool) + all gate-mode weights up front
    pool_norm = torch.norm(pool_f_128, p=2, dim=1)
    n_z = (pool_norm - pool_norm.mean()) / (pool_norm.std() + 1e-8)
    c_z = (pool_conf - pool_conf.mean()) / (pool_conf.std() + 1e-8)
    u_epi = 1.0 - pool_conf.clamp(0.0, 1.0)  # epistemic proxy in (0,1], higher = worse
    
    w_modes = {}
    for mode in ['epistemic', 'geometric', 'soft_dual_weight', 'and_gate', 'ellipsoid_gate']:
        w_modes[mode] = fuse_uncertainties(u_epi, n_z, method=mode, cfg=gate_cfg)
    # Flipped joint (Phase 13 hypothesis): harmful = HIGH confidence AND HIGH norm
    w_modes['joint_flip'] = torch.exp(-torch.relu(c_z)) * torch.exp(-torch.relu(n_z))
    
    # Base accuracy on validation set
    sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), base_protos.T)
    base_preds = proto_lbls[sims.argmax(dim=1)]
    zero_shot_correct = (base_preds == val_lbls).sum().item()
    zero_shot_acc = zero_shot_correct / len(val_lbls)
    zero_shot_miou = compute_miou(base_preds, val_lbls)
    zero_shot_per_class = {}
    present = set(val_lbls.tolist())
    for c in range(1, 17):
        if c not in present:
            continue
        tp = int(((base_preds == c) & (val_lbls == c)).sum().item())
        fp = int(((base_preds == c) & (val_lbls != c)).sum().item())
        fn = int(((base_preds != c) & (val_lbls == c)).sum().item())
        d = tp + fp + fn
        zero_shot_per_class[c] = tp / d if d > 0 else 0.0
    
    # Perfect Oracle test: weighted class-mean update restricted to true-label points
    print("      -> Running Perfect Oracle Test...")
    mask_perfect = (pool_pseudo == pool_lbls).float()
    w_one = torch.ones_like(pool_conf)
    adapted_protos = weighted_mean_update(base_protos, proto_lbls, pool_f_128,
                                          pool_pseudo, w_one, proj, device, mask=mask_perfect)
    perfect_acc, perfect_miou = eval_protos_miou(adapted_protos, proto_lbls, val_feats, val_lbls)
    
    # Leave-One-Update-Out (also collects per-update metadata for the AUROC diagnostics)
    print("      -> Running Leave-One-Update-Out Test...")
    helpful, neutral, harmful, all_meta = [], [], [], []
    alpha_single = 0.05 
    eval_pool_size = min(5000, len(pool_feats)) # 5000 updates tested
    
    for i in tqdm(range(eval_pool_size), desc="         Updates", leave=False):
        pl = pool_pseudo[i]
        idx = (proto_lbls == pl).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        
        new_protos = base_protos.clone()
        new_protos[idx] = new_protos[idx] * (1 - alpha_single) + pool_feats[i] * alpha_single
        new_protos[idx] = F.normalize(new_protos[idx], p=2, dim=0)
        
        sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), new_protos.T)
        preds = proto_lbls[sims.argmax(dim=1)]
        new_correct = (preds == val_lbls).sum().item()
        
        delta = new_correct - zero_shot_correct
        
        meta = {
            'conf': pool_conf[i].item(),
            'norm': pool_norm[i].item(),
            'delta': delta,
        }
        for mode, w in w_modes.items():
            meta[f'w_{mode}'] = w[i].item()
        all_meta.append(meta)
        
        if delta > 0:
            helpful.append(meta)
        elif delta < 0:
            harmful.append(meta)
        else:
            neutral.append(meta)
    
    # Gate-mode selectivity: AUROC of each gate's own weight for Helpful vs Harmful updates.
    # AUC > 0.5 = the gate score is selective in this space; AUC < 0.5 = it admits poison.
    y_lou = np.array([m['delta'] > 0 for m in all_meta]).astype(int)
    mode_auroc = {}
    if len(np.unique(y_lou)) >= 2:
        for mode in w_modes:
            wv = np.array([m[f'w_{mode}'] for m in all_meta], dtype=np.float64)
            if wv.std() > 0:
                mode_auroc[mode] = roc_auc_score(y_lou, wv)
    
    # Gate weight statistics: expose degeneracy (Phase 13: on the VIB-capped space,
    # all shipped modes collapse into one binary norm gate because z-scores saturate)
    weight_stats = {}
    for mode, w in w_modes.items():
        wv = w.detach().cpu().numpy()
        weight_stats[mode] = {
            'mean': float(wv.mean()),
            'frac_one': float((wv >= 0.999).mean()),   # saturated admit
            'frac_zero': float((wv <= 1e-6).mean()),   # saturated veto
        }
    
    # Diagnostic 1: Gated EMA Ladder (weighted class-mean updates)
    print("      -> Running Gated EMA Ladder...")
    
    gated = {'zero_shot': zero_shot_acc, 'zero_shot_miou': zero_shot_miou,
             'perfect_oracle': perfect_acc, 'perfect_oracle_miou': perfect_miou}
    
    w_uniform = torch.ones_like(pool_conf)
    gated['naive_ema'], gated['naive_ema_miou'] = eval_protos_miou(
        weighted_mean_update(base_protos, proto_lbls, pool_f_128, pool_pseudo, w_uniform, proj, device),
        proto_lbls, val_feats, val_lbls)
    
    w_top50 = torch.zeros_like(pool_conf)
    w_top50[torch.topk(pool_conf, pool_size // 2).indices] = 1.0
    gated['top50_conf'], gated['top50_conf_miou'] = eval_protos_miou(
        weighted_mean_update(base_protos, proto_lbls, pool_f_128, pool_pseudo, w_top50, proj, device),
        proto_lbls, val_feats, val_lbls)
    
    for mode in ['epistemic', 'geometric', 'soft_dual_weight', 'and_gate', 'ellipsoid_gate', 'joint_flip']:
        gated[mode], gated[f'{mode}_miou'] = eval_protos_miou(
            weighted_mean_update(base_protos, proto_lbls, pool_f_128, pool_pseudo, w_modes[mode], proj, device),
            proto_lbls, val_feats, val_lbls)
    
    # Threshold envelope sweep: best-case soft_dual_weight / geometric over a small grid.
    # If even the best config cannot beat naive EMA, the gate family is dead in this space;
    # if it can, the shipped defaults were simply old-space calibration.
    print("      -> Sweeping Gate Thresholds (soft_dual_weight, geometric)...")
    best_sdw, best_sdw_miou, best_sdw_cfg = -1.0, 0.0, None
    for u_th in [0.25, 0.5, 0.75]:
        for z_th in [0.0, 0.5, 1.0]:
            w = fuse_uncertainties(u_epi, n_z, method='soft_dual_weight',
                                   cfg={"u_th": u_th, "u_coef": gate_cfg.get("u_coef", 1.5),
                                        "z_th": z_th, "z_coef": gate_cfg.get("z_coef", 1.0)})
            acc, miou = eval_protos_miou(weighted_mean_update(base_protos, proto_lbls, pool_f_128,
                                                              pool_pseudo, w, proj, device),
                                         proto_lbls, val_feats, val_lbls)
            if acc > best_sdw:
                best_sdw, best_sdw_miou, best_sdw_cfg = acc, miou, [u_th, z_th]
    gated['sdw_best'] = best_sdw
    gated['sdw_best_miou'] = best_sdw_miou
    gated['sdw_best_cfg'] = best_sdw_cfg
    
    best_geom, best_geom_miou, best_geom_cfg = -1.0, 0.0, None
    for z_th in [-0.5, 0.0, 0.5, 1.0]:
        w = fuse_uncertainties(u_epi, n_z, method='geometric',
                               cfg={"z_th": z_th, "z_coef": gate_cfg.get("z_coef", 1.0)})
        acc, miou = eval_protos_miou(weighted_mean_update(base_protos, proto_lbls, pool_f_128,
                                                          pool_pseudo, w, proj, device),
                                     proto_lbls, val_feats, val_lbls)
        if acc > best_geom:
            best_geom, best_geom_miou, best_geom_cfg = acc, miou, [z_th]
    gated['geom_best'] = best_geom
    gated['geom_best_miou'] = best_geom_miou
    gated['geom_best_cfg'] = best_geom_cfg
    
    # Query gate end-to-end: veto high-norm points at inference on the FROZEN
    # prototypes (Phase 15/16 band acc: low-norm points classify far better).
    # Reports point accuracy AND mIoU vs retained fraction for a norm-threshold sweep.
    print("      -> Running Query Gate (frozen prototypes, veto norm >= tau)...")
    val_norms = torch.norm(val_f_128, p=2, dim=1)
    norm_val_feats = F.normalize(val_feats, p=2, dim=1)
    query_gate = {}
    for tau in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, float('inf')]:
        keep = val_norms < tau
        n_keep = int(keep.sum().item())
        if n_keep < 100:
            query_gate[f"tau={tau}"] = {'acc': None, 'miou': None, 'retained': n_keep / len(val_norms)}
            continue
        sims = torch.matmul(norm_val_feats[keep], base_protos.T)
        preds = proto_lbls[sims.argmax(dim=1)]
        lbl = val_lbls[keep]
        query_gate[f"tau={tau}"] = {
            'acc': float((preds == lbl).float().mean().item()),
            'miou': compute_miou(preds, lbl),
            'retained': n_keep / len(val_norms),
        }
    
    res = {
        'zero_shot': zero_shot_acc,
        'perfect_acc': perfect_acc,
        'gated': gated,
        'query_gate': query_gate,
        'zero_shot_per_class_iou': zero_shot_per_class,
        'auroc': compute_signal_aurocs(all_meta),
        'mode_auroc': mode_auroc,
        'weight_stats': weight_stats,
        'h_conf': np.mean([m['conf'] for m in helpful]) if helpful else 0.0,
        'hm_conf': np.mean([m['conf'] for m in harmful]) if harmful else 0.0,
        'h_norm': np.mean([m['norm'] for m in helpful]) if helpful else 0.0,
        'hm_norm': np.mean([m['norm'] for m in harmful]) if harmful else 0.0,
        'h_count': len(helpful),
        'hm_count': len(harmful)
    }
    return res
