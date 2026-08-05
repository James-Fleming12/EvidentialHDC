import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.HDC_utils import fuse_uncertainties, GATE_CFG

CORRUPTIONS = [
    'fog', 'snow', 'wet_ground', 'incomplete_echo', 
    'crosstalk', 'beam_missing', 'motion_blur', 'cross_sensor'
]

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


def evaluate_oracle_retrain(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj, device,
                            pool_size=1000000, buffer_frac=0.05, rounds=5,
                            buffer_mode='trainer', update_strength=2, per_class=False, seed=42,
                            artifact_max_norm=6.0, artifact_max_loss=0.15,
                            artifact_min_cos_true=0.05, artifact_min_margin=0.02):
    """Oracle retraining with buffer selection, faithful to Basic_HD.retrain / unsup_main.py.

    Perfect labels (oracle bound). Two buffer-selection modes, both with the
    trainer's 2x perceptron update (w[true] += 2*hv, w[pred] -= 2*hv) applied to
    misclassified points only:

      'trainer' (default, matches the codebase):
        - cumulative is_wrong memory over the pool (points stay until sampled)
        - each round samples buffer_frac of the pool: all remembered-wrong points
          first (up to the cap), then random fill; sampled points are cleared
        - update on this round's misclassified sampled points; newly-wrong re-added
      'hyperlidar' (paper form):
        - per-round losses recomputed; buffer = half top-loss ("hard") + half random

    Returns per-round {round, acc, miou, buffer_size, hard_size, rand_size,
    wrong_size, wrong_mem}.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    pool_idx = perm[:pool_size]
    val_idx = perm[-100000:]
    pool_f = corrupt_feats[pool_idx].to(device)
    pool_l = corrupt_lbls[pool_idx].to(device)
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = torch.sign(val_f @ proj)

    protos = F.normalize(base_protos.clone(), p=2, dim=1)
    n = len(pool_f)
    n_samples = max(1, int(n * buffer_frac))
    is_wrong = torch.zeros(n, dtype=torch.bool, device=device)
    results = []
    for r in range(rounds + 1):
        acc, miou = eval_protos_miou(protos, proto_lbls, val_h, val_l)
        if r == rounds:
            results.append({'round': r, 'acc': acc, 'miou': miou})
            break
        # --- buffer selection ---
        if buffer_mode == 'trainer':
            wrong_idx = is_wrong.nonzero(as_tuple=False).view(-1)
            if per_class:
                # per-class quota from the wrong memory: protects rare classes
                # from being starved of buffer slots by majority-class errors
                quota = max(1, n_samples // len(proto_lbls))
                parts = []
                for c in proto_lbls.tolist():
                    cm = (is_wrong & (pool_l == c))
                    cnt = int(cm.sum().item())
                    if cnt == 0:
                        continue
                    idx = cm.nonzero(as_tuple=False).view(-1)
                    parts.append(idx[torch.randperm(len(idx), device=device)[:min(cnt, quota)]])
                hard_idx = torch.cat(parts) if parts else torch.tensor([], device=device, dtype=torch.long)
            elif wrong_idx.numel() >= n_samples:
                hard_idx = wrong_idx[torch.randperm(len(wrong_idx), device=device)[:n_samples]]
            else:
                hard_idx = wrong_idx
            hard_size = len(hard_idx)
            is_wrong[hard_idx] = False  # sampled points leave the memory
            non_wrong = (~is_wrong).nonzero(as_tuple=False).view(-1)
            remaining = n_samples - hard_size
            fill = non_wrong[torch.randperm(len(non_wrong), device=device)[:remaining]]
            sel = torch.cat([hard_idx, fill])
            rand_size = len(fill)
        elif buffer_mode == 'hyperlidar':  # paper form: per-round top-loss + random
            losses = torch.zeros(n, device=device)
            preds_all = torch.zeros(n, dtype=torch.long, device=device)
            for s in range(0, n, 100000):
                h = torch.sign(pool_f[s:s + 100000] @ proj)
                sims = h @ protos.T
                preds_all[s:s + 100000] = proto_lbls[sims.argmax(dim=1)]
                true_idx = torch.searchsorted(proto_lbls, pool_l[s:s + 100000])
                true_val = sims[torch.arange(len(preds_all[s:s + 100000]), device=device), true_idx]
                losses[s:s + 100000] = (sims.max(dim=1).values - true_val).clamp(min=0)
            n_hard = max(1, n_samples // 2)
            if per_class:
                quota = max(1, n_hard // len(proto_lbls))
                parts = []
                for c in proto_lbls.tolist():
                    cm = (pool_l == c)
                    cnt = int(cm.sum().item())
                    if cnt == 0:
                        continue
                    idx = cm.nonzero(as_tuple=False).view(-1)
                    top = torch.topk(losses[cm], min(cnt, quota)).indices
                    parts.append(idx[top])
                hard = torch.cat(parts) if parts else torch.tensor([], device=device, dtype=torch.long)
            else:
                hard = torch.topk(losses, n_hard).indices
            sel_set = set(hard.tolist())
            rest = torch.tensor([i for i in range(n) if i not in sel_set], device=device, dtype=torch.long)
            rand = rest[torch.randperm(len(rest), device=device)[:n_samples - len(hard)]]
            sel = torch.cat([hard, rand])
            hard_size, rand_size = len(hard), len(rand)
        else:  # 'artifact': hard candidates filtered by artifact signals
            # (1) norm: fog artifacts live at high 128D magnitude (query-gate evidence)
            # (2) too far from true prototype: cos(q, P_true) too low
            # (3) confidently absorbed by a wrong class: perceptron loss too high
            # (4) too ambiguous: top-2 cosine margin too small
            norms = torch.norm(pool_f, p=2, dim=1)
            losses = torch.zeros(n, device=device)
            cos_true = torch.zeros(n, device=device)
            margin = torch.zeros(n, device=device)
            preds_all = torch.zeros(n, dtype=torch.long, device=device)
            for s in range(0, n, 100000):
                h = torch.sign(pool_f[s:s + 100000] @ proj)
                sims = h @ protos.T
                top2 = torch.topk(sims, 2, dim=1)
                preds_all[s:s + 100000] = proto_lbls[top2.indices[:, 0]]
                ti = torch.searchsorted(proto_lbls, pool_l[s:s + 100000])
                tv = sims[torch.arange(len(preds_all[s:s + 100000]), device=device), ti]
                cos_true[s:s + 100000] = tv
                losses[s:s + 100000] = (top2.values[:, 0] - tv).clamp(min=0)
                margin[s:s + 100000] = (top2.values[:, 0] - top2.values[:, 1]).clamp(min=0)
            mis = preds_all != pool_l
            n_mis = int(mis.sum().item())
            f_norm = mis & (norms < artifact_max_norm)
            f_true = f_norm & (cos_true >= artifact_min_cos_true)
            f_loss = f_true & (losses <= artifact_max_loss)
            f_margin = f_loss & (margin >= artifact_min_margin)
            cand = f_margin
            n_cand = int(cand.sum().item())
            if n_cand == 0:
                hard_idx = torch.tensor([], device=device, dtype=torch.long)
            elif per_class:
                quota = max(1, n_samples // len(proto_lbls))
                parts = []
                for c in proto_lbls.tolist():
                    cm = (cand & (pool_l == c))
                    cnt = int(cm.sum().item())
                    if cnt == 0:
                        continue
                    idx = cm.nonzero(as_tuple=False).view(-1)
                    # top-loss within the filtered candidates (hard emphasis, artifact-free)
                    take = min(cnt, quota)
                    top = torch.topk(losses[idx], take).indices
                    parts.append(idx[top])
                hard_idx = torch.cat(parts) if parts else torch.tensor([], device=device, dtype=torch.long)
            else:
                idx = cand.nonzero(as_tuple=False).view(-1)
                take = min(len(idx), max(1, n_samples // 2))
                top = torch.topk(losses[idx], take).indices
                hard_idx = idx[top]
            hard_size = len(hard_idx)
            sel_set = set(hard_idx.tolist())
            rest = torch.tensor([i for i in range(n) if i not in sel_set], device=device, dtype=torch.long)
            rand = rest[torch.randperm(len(rest), device=device)[:n_samples - hard_size]]
            sel = torch.cat([hard_idx, rand])
            rand_size = len(rand)
            filter_stats = {'n_mis': n_mis, 'pass_norm': int(f_norm.sum().item()),
                            'pass_true': int(f_true.sum().item()), 'pass_loss': int(f_loss.sum().item()),
                            'pass_margin': int(f_margin.sum().item()), 'n_cand': n_cand}
        # --- classify the buffer ---
        buf_h = torch.sign(pool_f[sel] @ proj)
        sims = buf_h @ protos.T
        pred = proto_lbls[sims.argmax(dim=1)]
        wrong_now = pred != pool_l[sel]
        # --- trainer-faithful perceptron update (2x), misclassified only ---
        if wrong_now.sum() > 0:
            ti = torch.searchsorted(proto_lbls, pool_l[sel][wrong_now])
            pi = torch.searchsorted(proto_lbls, pred[wrong_now])
            hw = buf_h[wrong_now]
            for _ in range(update_strength):
                protos.index_add_(0, ti, hw)
                protos.index_add_(0, pi, -hw)
            protos = F.normalize(protos, p=2, dim=1)
        # --- re-add newly wrong to the memory (trainer mode) ---
        if buffer_mode == 'trainer':
            is_wrong[sel[wrong_now]] = True
        results.append({'round': r, 'acc': acc, 'miou': miou, 'buffer_size': len(sel),
                        'hard_size': hard_size, 'rand_size': rand_size,
                        'wrong_size': int(wrong_now.sum().item()),
                        'wrong_mem': int(is_wrong.sum().item()),
                        **({'filter_stats': filter_stats} if buffer_mode == 'artifact' else {})})
    return results


def gate_sweep(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj, device,
               clf=None, clean_means128=None, seed=42):
    """In-memory artifact-gate sweep (Phase 23).

    Computes per-point gate signals once (128D norm, 10kD top-1 cosine, top-2 margin,
    and the oracle-aware perceptron loss = cos(top1) - cos(true)), then sweeps the
    threshold space, reporting (acc, mIoU, retention) Pareto bands. The loss-based
    gate is an oracle upper bound: it shows what a label-free gate could achieve if
    the loss were perfectly estimated.

    Goal: does ANY gate config reach ~20 mIoU on Fog/Crosstalk at usable retention
    (>= 25-50%)? If not, artifact gating is exhausted and the fix is the encoder.
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
    cos_top1 = top2.values[:, 0]
    margin = (top2.values[:, 0] - top2.values[:, 1]).clamp(min=0)
    ti = torch.searchsorted(proto_lbls, val_l)
    cos_true = sims[torch.arange(len(val_l), device=device), ti]
    loss = (cos_top1 - cos_true).clamp(min=0)

    # 128D label-free signals: nearest-clean-prototype cosine (proxy for cos(true),
    # where the continuous-space geometry is the meaningful one) and probe confidence
    if clean_means128 is not None:
        cm = torch.stack([clean_means128[c] for c in sorted(clean_means128)]).to(device)
        sims128 = F.normalize(val_f, p=2, dim=1) @ F.normalize(cm, p=2, dim=1).T
        cos128 = sims128.max(dim=1).values
    else:
        cos128 = torch.full((len(val_l),), -1.0, device=device)
    if clf is not None:
        probs = clf.predict_proba(val_f.cpu().numpy())
        conf = torch.tensor(probs.max(axis=1), device=device)
    else:
        conf = torch.zeros(len(val_l), device=device)

    n = len(val_l)
    rows = []
    for max_norm in [1e9, 8.0, 6.0, 5.0, 4.0]:
        for min_margin in [0.0, 0.02, 0.05, 0.1, 0.2]:
            for min_cos1 in [-1.0, 0.0, 0.1, 0.2, 0.3]:
                for min_cos128 in [-1.0, 0.2, 0.3, 0.4]:
                    for min_conf in [0.0, 0.3, 0.5]:
                        keep = ((norms < max_norm) & (margin >= min_margin)
                                & (cos_top1 >= min_cos1) & (cos128 >= min_cos128)
                                & (conf >= min_conf))
                        nk = int(keep.sum().item())
                        if nk < 1000:
                            continue
                        preds = proto_lbls[sims[keep].argmax(dim=1)]
                        lbl = val_l[keep]
                        acc = float((preds == lbl).float().mean().item())
                        rows.append((nk / n, acc, compute_miou(preds, lbl),
                                     max_norm, min_margin, min_cos1, min_cos128, min_conf))
    for max_loss in [0.15, 0.1, 0.05, 0.02]:
        keep = loss <= max_loss
        nk = int(keep.sum().item())
        if nk < 1000:
            continue
        preds = proto_lbls[sims[keep].argmax(dim=1)]
        lbl = val_l[keep]
        rows.append((nk / n, float((preds == lbl).float().mean().item()),
                     compute_miou(preds, lbl), 'loss', max_loss, '-', '-', '-'))

    # Pareto: best mIoU within retention bands
    bands = [(0.75, 1.01, '>=75%'), (0.5, 0.75, '50-75%'), (0.25, 0.5, '25-50%'),
             (0.1, 0.25, '10-25%'), (0.0, 0.1, '<10%')]
    pareto = []
    for lo, hi, name in bands:
        cand = [r for r in rows if lo <= r[0] < hi]
        if cand:
            best = max(cand, key=lambda r: r[2])
            pareto.append({'band': name, 'retention': best[0], 'acc': best[1],
                           'miou': best[2], 'cfg': (best[3], best[4], best[5], best[6], best[7])})
    # oracle-loss bound at 25-50% and 50-75% bands
    loss_band = {}
    for lo, hi, name in bands:
        cand = [r for r in rows if r[3] == 'loss' and lo <= r[0] < hi]
        if cand:
            best = max(cand, key=lambda r: r[2])
            loss_band[name] = {'retention': best[0], 'acc': best[1], 'miou': best[2],
                               'max_loss': best[4]}
    # per-class IoU at the overall best config
    best = max(rows, key=lambda r: r[2])
    if best[3] == 'loss':
        keep = loss <= best[4]
    else:
        keep = ((norms < best[3]) & (margin >= best[4]) & (cos_top1 >= best[5])
                & (cos128 >= best[6]) & (conf >= best[7]))
    preds = proto_lbls[sims[keep].argmax(dim=1)]
    lbl = val_l[keep]
    per_class = {}
    present = set(lbl.tolist())
    for c in range(1, 17):
        if c not in present:
            continue
        tp = int(((preds == c) & (lbl == c)).sum().item())
        fp = int(((preds == c) & (lbl != c)).sum().item())
        fn = int(((preds != c) & (lbl == c)).sum().item())
        d = tp + fp + fn
        per_class[c] = tp / d if d > 0 else 0.0
    return {'pareto': pareto, 'loss_band': loss_band,
            'best': {'retention': best[0], 'acc': best[1], 'miou': best[2],
                     'cfg': (best[3], best[4], best[5])},
            'per_class_iou': per_class}


CLASS_NAMES = {1: 'car', 2: 'bicycle', 3: 'motorcycle', 4: 'truck', 5: 'other-vehicle',
               6: 'person', 7: 'road', 8: 'fence', 9: 'vegetation', 10: 'trunk',
               11: 'terrain', 12: 'pole', 13: 'traffic-sign', 14: 'other-ground',
               15: 'building', 16: 'other-object'}


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

def main():
    parser = argparse.ArgumentParser("./oracle_gating_eval.py")
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--load_path", type=str, default="logs/med_pretrain_supcon_vib",
                        help="Dir containing a trained GenTrainer checkpoint (e.g. logs/micro_pretrain/supcon_vib)")
    parser.add_argument("--method", type=str, default="supcon_vib")
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=1000000,
                        help="Adaptation pool size (seeded uniform sample over all frames). "
                             "Must be large (Phase 13: a 20k pool is ~250x too small to "
                             "refine 10kD prototypes whose base comes from millions of points; "
                             "the update is a vectorized weighted class-mean, so large pools are cheap).")
    parser.add_argument("--u_th", type=float, default=GATE_CFG["u_th"])
    parser.add_argument("--u_coef", type=float, default=GATE_CFG["u_coef"])
    parser.add_argument("--z_th", type=float, default=GATE_CFG["z_th"])
    parser.add_argument("--z_coef", type=float, default=GATE_CFG["z_coef"])
    parser.add_argument("--corruptions", type=str, default="",
                        help="Comma-separated subset of the 8 corruptions (default: all)")
    parser.add_argument("--whiten", action="store_true",
                        help="ZCA-whiten all features (transform from 500k clean points) before "
                             "the ladder — anisotropy probe: does decorrelating the space improve "
                             "the 10kD prototype decode?")
    parser.add_argument("--oracle_retrain", type=int, default=0,
                        help="Run oracle retraining (perfect labels) for this many rounds "
                             "with buffer selection + perceptron updates (Basic_HD.retrain). "
                             "0 = off.")
    parser.add_argument("--buffer_frac", type=float, default=0.05,
                        help="Buffer fraction of the pool per retraining round (the trainer's 0.05).")
    parser.add_argument("--buffer_mode", type=str, default="trainer",
                        choices=["trainer", "hyperlidar", "artifact"],
                        help="'trainer' = cumulative is_wrong memory + random fill (matches the "
                             "codebase). 'hyperlidar' = per-round top-loss + random (paper form). "
                             "'artifact' = hard candidates filtered by artifact signals "
                             "(norm, cosine-to-true, perceptron loss, top-2 margin).")
    parser.add_argument("--update_strength", type=int, default=2,
                        help="Perceptron update multiplier (Basic_HD.retrain applies each "
                             "index_add twice = 2x).")
    parser.add_argument("--buffer_per_class", action="store_true",
                        help="Per-class hard selection (per-class quota from the wrong memory "
                             "/ top-loss), protecting rare classes from majority-class domination.")
    parser.add_argument("--artifact_max_norm", type=float, default=6.0,
                        help="Artifact filter: keep only misclassified points with 128D norm below this "
                             "(fog artifacts live at high magnitude).")
    parser.add_argument("--artifact_max_loss", type=float, default=0.15,
                        help="Artifact filter: keep only misclassified points with perceptron loss "
                             "(cos(pred) - cos(true)) at most this (confidently-wrong = hallucination).")
    parser.add_argument("--artifact_min_cos_true", type=float, default=0.05,
                        help="Artifact filter: keep only points with cosine to their true prototype "
                             "at least this (too far from own prototype = noise).")
    parser.add_argument("--artifact_min_margin", type=float, default=0.02,
                        help="Artifact filter: keep only points with top-2 cosine margin at least "
                             "this (too ambiguous = unreliable).")
    parser.add_argument("--gate_sweep", action="store_true",
                        help="Run the in-memory artifact-gate sweep (Phase 23) and skip the "
                             "ladder: extract per-point signals once, sweep threshold space, "
                             "report acc/mIoU Pareto bands + oracle-loss bound + per-class IoU.")
    parser.add_argument("--autopsy", action="store_true",
                        help="Run the per-condition hyperspace/decode autopsy (Phase 24) for "
                             "each corruption and print the comparison table: artifact profile, "
                             "margin/norm stats, cosine shift, ellipticity, LP headroom, "
                             "binarized mean quality. Skips the ladder.")
    args, _ = parser.parse_known_args()
    
    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    # Seed the full pipeline (extraction workers, point subsampling) so feature
    # extraction and the pool/val split are reproducible across runs.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    method = args.method
    load_path = args.load_path
    
    if not os.path.exists(load_path):
        print(f"Error: {load_path} not found.")
        return
        
    gate_cfg = {"u_th": args.u_th, "u_coef": args.u_coef,
                "z_th": args.z_th, "z_coef": args.z_coef}
    
    print(f"\nLoading Model: {method} from {load_path}")
    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, load_path, path=load_path, method=method)
    model = trainer.model
    model.eval()
    
    clean_parser = Parser(root=args.kitti_dir, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                          labels=DATA["labels"], color_map=DATA["color_map"], learning_map=DATA["learning_map"],
                          learning_map_inv=DATA["learning_map_inv"], sensor=ARCH["dataset"]["sensor"],
                          max_points=ARCH["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)
                          
    clean_loader = clean_parser.get_train_set()
    
    clean_feats, clean_lbls = [], []
    NUM_BATCHES = args.num_batches
    
    print("-> Extracting Clean Latents (100 Frames)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(clean_loader, total=NUM_BATCHES)):
            if i >= NUM_BATCHES: break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            clean_feats.append(z_flat.cpu())
            clean_lbls.append(labels[mask].cpu())
            
    clean_feats = torch.cat(clean_feats, dim=0)
    clean_lbls = torch.cat(clean_lbls, dim=0)
    print(f"   [Total Clean Points Extracted: {len(clean_feats)}]")
    
    # Optional ZCA whitening probe: decorrelate clean features with clean statistics,
    # apply the same transform to every corrupt set, then re-run the ladder.
    # Tests whether the measured anisotropy (ellipticity ~0.5 clean / ~0.6-0.8 fog)
    # is the bottleneck of the 10kD centroid decode.
    if args.whiten:
        print("-> Computing ZCA Whitening (from 500k clean points)...")
        torch.manual_seed(42)
        sub_idx = torch.randperm(len(clean_feats))[:500000]
        mean_c = clean_feats[sub_idx].mean(dim=0)
        Xc = clean_feats[sub_idx] - mean_c
        cov = (Xc.T @ Xc) / len(Xc)
        U, S, _ = torch.linalg.svd(cov)
        W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-6)) @ U.T
        whiten = (mean_c, W)
        clean_feats = (clean_feats - mean_c) @ W
        print(f"   [whiten] covariance eigen-range: {float(S[-1]):.5f} .. {float(S[0]):.5f} "
              f"(ratio {float(S[0] / (S[-1] + 1e-8)):.1f})")
    
    # Train Linear Probe on 128D (to use as our confidence oracle)
    print("-> Training Linear Probe Oracle (128D on 100k points)...")
    clf = LogisticRegression(max_iter=1000)
    train_size = min(100000, len(clean_feats))
    clf.fit(clean_feats[:train_size].numpy(), clean_lbls[:train_size].numpy())
    
    probe_clean_acc = clf.score(clean_feats[:train_size].numpy(), clean_lbls[:train_size].numpy())
    print(f"   [Base] Linear Probe Accuracy (Clean): {probe_clean_acc:.4f}\n")
    
    # Build robust 10kD HDC base prototypes over all clean points
    print("-> Building 10kD HDC Clean Base Prototypes...")
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    base_protos, proto_lbls = build_hdc_prototypes(clean_feats, clean_lbls, proj, device=device)
    
    # 128D clean class means (for the gate sweep's label-free cos-to-prototype signal)
    clean_means128 = {}
    for c in proto_lbls.tolist():
        m = clean_feats[clean_lbls == c]
        if len(m) > 0:
            clean_means128[c] = F.normalize(m.mean(dim=0), p=2, dim=0)
    
    # Per-dimension clean feature statistics (for the BN-style test-time alignment probe)
    torch.manual_seed(42)
    sub_idx = torch.randperm(len(clean_feats))[:500000]
    clean_stats = (clean_feats[sub_idx].mean(dim=0),
                   clean_feats[sub_idx].std(dim=0) + 1e-6)
    
    # Clean-data control: adapting clean -> clean, no poison exists. A good gate must
    # stay ~= naive EMA here; any large degradation is over-gating (a gate fault).
    clean_control = None
    if len(clean_feats) >= args.pool_size + 100000:
        print("\n-> Running Clean-Data Gate Control (adapt clean -> clean, no poison)...")
        clean_control = evaluate_oracle_gating(base_protos, proto_lbls,
                                               clean_feats[:args.pool_size + 100000],
                                               clean_lbls[:args.pool_size + 100000],
                                               clf, proj, device,
                                               pool_size=args.pool_size, gate_cfg=gate_cfg)
        print("   -> Clean Ladder (gate should ~= naive_ema):")
        for k, v in clean_control['gated'].items():
            if not k.endswith('_cfg') and not k.endswith('_miou'):
                print(f"      {k:<16}: {v:.4f}")
    
    # We no longer need the massive clean_feats tensor
    del clean_feats
    del clean_lbls
    
    corruptions = CORRUPTIONS if not args.corruptions else [c.strip() for c in args.corruptions.split(',')]
    all_results = {}
    out_path = os.path.join(load_path, "oracle_gating_results.json")
    if os.path.exists(out_path):
        try:
            with open(out_path, 'r') as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} existing corruption results from {out_path} (results will be merged)")
        except Exception:
            print("Warning: could not parse existing results file; starting fresh.")
    
    for corruption in corruptions:
        print(f"\n{'='*60}")
        print(f"Evaluating Corruption: {corruption.upper()}")
        print(f"{'='*60}")
        
        fog_dir = os.path.join(args.kittic_dir, corruption, 'heavy')
        if not os.path.exists(fog_dir):
            fog_dir = os.path.join(args.kittic_dir, corruption, 'moderate')
            
        corrupt_parser = Parser(root=fog_dir, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                            labels=DATA["labels"], color_map=DATA["color_map"], learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"], sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)
        
        corrupt_loader = corrupt_parser.get_train_set()
        
        corrupt_feats, corrupt_lbls, corrupt_depths = [], [], []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(corrupt_loader, total=NUM_BATCHES, desc=f"   Ext. {corruption}")):
                if i >= NUM_BATCHES: break
                in_vol = batch[0].to(device)
                labels = batch[2].to(device).view(-1)
                mask = (batch[1].to(device) > 0).view(-1)
                
                out_tuple = model(in_vol)
                if len(out_tuple) == 3:
                    _, _, z8 = out_tuple
                else:
                    _, z8 = out_tuple
                z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
                corrupt_feats.append(z_flat.cpu())
                corrupt_lbls.append(labels[mask].cpu())
                corrupt_depths.append(in_vol[:, 0, :, :].reshape(-1)[mask].cpu())  # range channel, same mask order
                
        corrupt_feats = torch.cat(corrupt_feats, dim=0)
        corrupt_lbls = torch.cat(corrupt_lbls, dim=0)
        corrupt_depths = torch.cat(corrupt_depths, dim=0)
        
        if args.whiten:
            corrupt_feats = (corrupt_feats - whiten[0]) @ whiten[1]
        
        probe_corrupt_acc = clf.score(corrupt_feats[:train_size].numpy(), corrupt_lbls[:train_size].numpy())
        print(f"   -> 128D Linear Probe Accuracy: {probe_corrupt_acc:.4f}")
        
        if args.autopsy:
            res = {}
            print("      -> Running Condition Autopsy...")
            au = condition_autopsy(base_protos, proto_lbls, clean_means128,
                                   corrupt_feats, corrupt_lbls, proj, device,
                                   clf=clf, clean_stats=clean_stats,
                                   corrupt_depths=corrupt_depths)
            res['autopsy'] = au
            all_results[corruption] = res
            continue
        if args.gate_sweep:
            res = {}
            print("      -> Running Artifact-Gate Sweep (in-memory)...")
            gs = gate_sweep(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj, device,
                            clf=clf, clean_means128=clean_means128)
            res['gate_sweep'] = gs
            print("   -> Gate-Sweep Pareto (best mIoU per retention band):")
            for b in gs['pareto']:
                cfg = b['cfg']
                if cfg[0] == 'loss':
                    desc = f"loss<={cfg[1]}"
                else:
                    desc = (f"norm<{cfg[0]} marg>={cfg[1]} cos1>={cfg[2]} "
                            f"cos128>={cfg[3]} conf>={cfg[4]}")
                print(f"      {b['band']:<8}: ret {b['retention']*100:5.1f}% | acc {b['acc']:.4f} | "
                      f"mIoU {b['miou']:.4f} | {desc}")
            lb = gs['loss_band']
            if lb:
                print("   -> Oracle-Loss Gate Bound (label-free achievable if loss were learned):")
                for band, v in lb.items():
                    print(f"      {band:<8}: ret {v['retention']*100:5.1f}% | acc {v['acc']:.4f} | "
                          f"mIoU {v['miou']:.4f} | loss<={v['max_loss']}")
            best = gs['best']
            print(f"   -> Best config: mIoU {best['miou']:.4f} | acc {best['acc']:.4f} | "
                  f"ret {best['retention']*100:.1f}% | cfg {best['cfg']}")
            pc = gs['per_class_iou']
            if pc:
                print("   -> Per-class IoU at best config: "
                      + ", ".join(f"{c}:{v:.2f}" for c, v in sorted(pc.items())))
            all_results[corruption] = res
            continue
        res = evaluate_oracle_gating(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, clf, proj,
                                     device, pool_size=args.pool_size, gate_cfg=gate_cfg)
        if args.oracle_retrain > 0:
            print(f"      -> Running Oracle Retraining ({args.oracle_retrain} rounds, "
                  f"buffer {args.buffer_frac*100:.0f}%, mode={args.buffer_mode}, "
                  f"update_strength={args.update_strength})...")
            rt = evaluate_oracle_retrain(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj,
                                         device, pool_size=args.pool_size,
                                         buffer_frac=args.buffer_frac,
                                         rounds=args.oracle_retrain,
                                         buffer_mode=args.buffer_mode,
                                         update_strength=args.update_strength,
                                         per_class=args.buffer_per_class,
                                         artifact_max_norm=args.artifact_max_norm,
                                         artifact_max_loss=args.artifact_max_loss,
                                         artifact_min_cos_true=args.artifact_min_cos_true,
                                         artifact_min_margin=args.artifact_min_margin)
            res['oracle_retrain'] = rt
            print("   -> Oracle Retrain Trajectory (acc | mIoU | buf hard/rand | wrong-now/mem):")
            for row in rt:
                if row.get('buffer_size') is not None:
                    line = (f"      round {row['round']}: {row['acc']:.4f} | {row['miou']:.4f} | "
                            f"{row['hard_size']}/{row['rand_size']} | {row['wrong_size']}/{row['wrong_mem']}")
                    fs = row.get('filter_stats')
                    if fs:
                        line += (f" | filt {fs['n_cand']}/{fs['n_mis']} "
                                 f"(norm {fs['pass_norm']}, true {fs['pass_true']}, "
                                 f"loss {fs['pass_loss']}, marg {fs['pass_margin']})")
                    print(line)
                else:
                    print(f"      round {row['round']}: {row['acc']:.4f} | {row['miou']:.4f}")
        res['probe_acc'] = probe_corrupt_acc
        all_results[corruption] = res
        
        print(f"   -> Perfect Oracle HDC Acc: {res['perfect_acc']:.4f} (Zero-Shot: {res['zero_shot']:.4f})")
        print("   -> Gated EMA Ladder (acc | mIoU):")
        for k, v in res['gated'].items():
            if k.endswith('_cfg') or k.endswith('_miou'):
                continue
            m = res['gated'].get(f'{k}_miou')
            if m is not None:
                print(f"      {k:<16}: {v:.4f} | {m:.4f}")
            else:
                print(f"      {k:<16}: {v:.4f}")
        zs_m = res['gated'].get('zero_shot_miou')
        if zs_m is not None:
            print(f"   -> Zero-Shot mIoU: {zs_m:.4f} | Oracle mIoU: {res['gated'].get('perfect_oracle_miou', 0):.4f} | "
                  f"Naive mIoU: {res['gated'].get('naive_ema_miou', 0):.4f}")
        pc = res.get('zero_shot_per_class_iou')
        if pc:
            print("   -> Zero-Shot Per-Class IoU: " + ", ".join(
                f"{CLASS_NAMES.get(c, str(c))}={v:.3f}" for c, v in sorted(pc.items())))
        if res['gated'].get('sdw_best_cfg'):
            print(f"      [sdw_best at u_th={res['gated']['sdw_best_cfg'][0]}, z_th={res['gated']['sdw_best_cfg'][1]}]")
        if res['gated'].get('geom_best_cfg'):
            print(f"      [geom_best at z_th={res['gated']['geom_best_cfg'][0]}]")
        qg = res.get('query_gate', {})
        if qg:
            print("   -> Query Gate (frozen prototypes, veto norm >= tau): acc | mIoU | retained")
            for k, v in qg.items():
                if v['acc'] is None:
                    print(f"      {k:<10}:   --   |   --   | {v['retained']*100:.1f}%")
                else:
                    print(f"      {k:<10}: {v['acc']:.4f} | {v['miou']:.4f} | {v['retained']*100:.1f}%")
        a = res.get('auroc', {})
        if a:
            print(f"   -> Signal AUROC (Helpful vs Harmful): "
                  f"conf {a.get('conf', 0):.3f} | norm {a.get('norm', 0):.3f} | "
                  f"joint_z {a.get('joint_z', 0):.3f} | lr {a.get('lr', 0):.3f}")
        ma = res.get('mode_auroc', {})
        if ma:
            print("   -> Gate-Mode AUROC (gate's own weight selectivity): "
                  + " | ".join(f"{k} {v:.3f}" for k, v in ma.items()))
        ws = res.get('weight_stats', {})
        if ws:
            print("   -> Gate Weight Stats (mean | %w~1 | %w~0):")
            for k, v in ws.items():
                print(f"      {k:<16}: {v['mean']:.3f} | {v['frac_one']*100:.0f}% | {v['frac_zero']*100:.0f}%")
        print(f"   -> Leave-One-Out (5k tests): {res['h_count']} Helpful, {res['hm_count']} Harmful")
        if res['hm_count'] > 0:
            print(f"      Helpful Conf: {res['h_conf']:.4f} | Harmful Conf: {res['hm_conf']:.4f}")
            print(f"      Helpful Norm: {res['h_norm']:.4f} | Harmful Norm: {res['hm_norm']:.4f}")
    
    if clean_control is not None:
        all_results['clean_control'] = clean_control
    
    print("\n\n" + "="*110)
    print(" GATED EMA LADDER (all corruptions) — acc | mIoU")
    print("="*110)
    header = (f"| {'Corruption':<16} | {'ZeroShot':<8} | {'ZS-mIoU':<8} | {'Naive':<7} | {'SDW*':<7} | "
              f"{'Oracle':<8} | {'Or-mIoU':<8} |")
    print(header)
    print("|" + "-"*17 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*9 + "|")
    for corruption, res in all_results.items():
        if corruption == 'clean_control':
            continue
        if 'gated' not in res:
            continue
        g = res['gated']
        zs_m = g.get('zero_shot_miou', 0.0)
        or_m = g.get('perfect_oracle_miou', 0.0)
        print(f"| {corruption:<16} | {g['zero_shot']:<8.4f} | {zs_m:<8.4f} | "
              f"{g['naive_ema']:<7.4f} | {g.get('sdw_best', 0):<7.4f} | "
              f"{res['perfect_acc']:<8.4f} | {or_m:<8.4f} |")
    print("="*110 + "\n")
    
    if args.autopsy:
        print("\n\n" + "="*140)
        print(" CONDITION AUTOPSY (frozen clean prototypes)")
        print("="*140)
        print(f"| {'Condition':<16} | {'Acc':<7} | {'mIoU':<7} | {'LP':<7} | {'LPmIoU':<8} | {'nMis':<7} | {'ArtFrac':<8} | "
              f"{'ArtSurv':<8} | {'marC/marM':<10} | {'nrmC/nrmM':<10} | {'<4norm':<7} | {'cosShift':<8} | "
              f"{'Ellip':<6} | {'BinCos':<7} | {'AlignAcc':<8} | {'AlignmIoU':<9} |")
        print("|" + "-"*17 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*7 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*10 + "|")
        for corruption, res in all_results.items():
            if 'autopsy' not in res:
                continue
            a = res['autopsy']
            surv = a['artifact_survivors']
            al = f"{a['align_acc']:.3f}/{a['align_miou']:.3f}" if a.get('align_acc') is not None else "n/a"
            lp_m = a.get('lp_miou', 0.0)
            print(f"| {corruption:<16} | {a['acc']:<7.3f} | {a['miou']:<7.3f} | {a['lp_acc']:<7.3f} | "
                  f"{lp_m:<8.3f} | {a['n_mis']:<7d} | {a['conf_artifact_frac']:<8.3f} | {surv[0]}/{surv[4]:<7d} | "
                  f"{a['margin_correct']:.2f}/{a['margin_mis']:.2f} | {a['norm_correct']:.1f}/{a['norm_mis']:.1f} | "
                  f"{a['near_origin']:<7.3f} | {a['cos_shift']:<8.3f} | {a['ellipticity']:<6.3f} | "
                  f"{a['binarized_cos']:<7.3f} | {al:<8} |")
        print("="*140 + "\n")
        print(" DEPTH/RANGE DIAGNOSTIC (far-thresh 25m; near_acc/far_acc of the proto decode)")
        print("="*140)
        for corruption, res in all_results.items():
            if 'autopsy' not in res:
                continue
            ds = res['autopsy'].get('depth_stats')
            if not ds:
                continue
            line = f"{corruption:<16} norm-depth corr {ds['norm_depth_corr']:+.3f} | "
            for c, row in sorted(ds.get('per_class', {}).items()):
                na = f"{row['near_acc']:.2f}" if row['near_acc'] is not None else "  -  "
                fa = f"{row['far_acc']:.2f}" if row['far_acc'] is not None else "  -  "
                line += f"c{c}(d{row['mean_depth']:.0f}m,far{row['far_frac']*100:.0f}%):{na}/{fa}  "
            print(line)
        print("="*140 + "\n")
    
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved Oracle Gating Results ({len(all_results)} corruptions) to {out_path}")

if __name__ == '__main__':
    main()
