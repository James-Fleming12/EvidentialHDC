"""Test-time adaptation iteration diagnostics (tta_iterations.md): gate sweep,
prototype rebalancing, prior correction, TTA battery, pool-size reconciliation,
Iteration 0-4 label analyses, MVAC views, Sinkhorn balance, ReAct, and the deep
label-information analysis. Extracted from oracle_gating_eval.py.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

from modules.oracle_core import weighted_mean_update, compute_miou
from modules.HDC_utils import fuse_uncertainties

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
    # Label-free Pareto: best mIoU per retention band among non-oracle (non-loss) configs.
    # The oracle-loss configs dominate the plain Pareto on healthy conditions, so this is
    # the table the paper needs: what a label-free gate alone achieves per condition.
    pareto_label_free = []
    for lo, hi, name in bands:
        cand = [r for r in rows if r[3] != 'loss' and lo <= r[0] < hi]
        if cand:
            best = max(cand, key=lambda r: r[2])
            pareto_label_free.append({'band': name, 'retention': best[0], 'acc': best[1],
                                      'miou': best[2],
                                      'cfg': (best[3], best[4], best[5], best[6], best[7])})
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
    return {'pareto': pareto, 'pareto_label_free': pareto_label_free,
            'loss_band': loss_band,
            'best': {'retention': best[0], 'acc': best[1], 'miou': best[2],
                     'cfg': (best[3], best[4], best[5])},
            'per_class_iou': per_class}

def prototype_rebalance(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj, device,
                        clf=None, clean_means128=None, min_pts=50, seed=42):
    """Update-side prototype rebalancing (the 'balancer' role).

    Distinct from decode-side gating (retained-subset mIoU): here the label-free
    gate selects WHICH points are allowed to update the prototypes, and the metric
    is the FULL-SCENE mIoU with the rebalanced prototypes. A class keeps its clean
    prototype when the gate retains too few of its points (the class-conditional
    collapse case). The oracle-loss selector is also swept as an upper bound.

    Returns per-config full-scene (acc, mIoU, selection fraction) and the best
    label-free config + the oracle-loss bound.
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
    cos_top1 = top2.values[:, 0]
    ti = torch.searchsorted(proto_lbls, val_l)
    cos_true = sims[torch.arange(len(val_l), device=device), ti]
    loss = (cos_top1 - cos_true).clamp(min=0)
    if clean_means128 is not None:
        cm = torch.stack([clean_means128[c] for c in sorted(clean_means128)]).to(device)
        cos128 = F.normalize(val_f, p=2, dim=1) @ F.normalize(cm, p=2, dim=1).T
        cos128 = cos128.max(dim=1).values
    else:
        cos128 = torch.full((len(val_l),), -1.0, device=device)
    if clf is not None:
        conf = torch.tensor(clf.predict_proba(val_f.cpu().numpy()).max(axis=1), device=device)
    else:
        conf = torch.zeros(len(val_l), device=device)

    pred_all = proto_lbls[sims.argmax(dim=1)]

    def full_scene_metrics(keep):
        nk = int(keep.sum().item())
        if nk < 1000:
            return None
        preds = pred_all[keep]
        new_protos = base_protos.clone()
        for i, c in enumerate(proto_lbls.tolist()):
            sel = keep.clone()
            sel[keep] = (preds == c)
            if int(sel.sum().item()) >= min_pts:
                new_protos[i] = torch.sign(val_h[sel].mean(dim=0))
        sims_r = F.normalize(val_h, p=2, dim=1) @ F.normalize(new_protos, p=2, dim=1).T
        preds_r = proto_lbls[sims_r.argmax(dim=1)]
        return {'selection': nk / len(val_l), 'acc': float((preds_r == val_l).float().mean().item()),
                'miou': compute_miou(preds_r, val_l)}

    rows = []
    for max_norm in [1e9, 8.0, 6.0, 5.0, 4.0]:
        for min_margin in [0.0, 0.02, 0.05, 0.1, 0.2]:
            for min_cos1 in [-1.0, 0.0, 0.1, 0.2, 0.3]:
                for min_cos128 in [-1.0, 0.2, 0.3, 0.4]:
                    for min_conf in [0.0, 0.3, 0.5]:
                        keep = ((norms < max_norm) & (margin >= min_margin)
                                & (cos_top1 >= min_cos1) & (cos128 >= min_cos128)
                                & (conf >= min_conf))
                        m = full_scene_metrics(keep)
                        if m:
                            rows.append((m['selection'], m['acc'], m['miou'],
                                         'free', max_norm, min_margin, min_cos1, min_cos128, min_conf))
    loss_rows = []
    for max_loss in [0.15, 0.1, 0.05, 0.02]:
        m = full_scene_metrics(loss <= max_loss)
        if m:
            loss_rows.append((m['selection'], m['acc'], m['miou'], max_loss))

    # best label-free rebalance: maximize full-scene mIoU with selection >= 25%
    usable = [r for r in rows if r[0] >= 0.25]
    best_free = max(usable, key=lambda r: r[2]) if usable else max(rows, key=lambda r: r[2])
    # oracle-loss bound at selection >= 25%
    usable_l = [r for r in loss_rows if r[0] >= 0.25]
    best_loss = max(usable_l, key=lambda r: r[2]) if usable_l else max(loss_rows, key=lambda r: r[2])
    zero = full_scene_metrics(torch.ones(len(val_l), dtype=torch.bool, device=device))
    return {
        'rows_label_free': [{'selection': r[0], 'acc': r[1], 'miou': r[2], 'cfg': r[3:]} for r in rows],
        'rows_oracle': [{'selection': r[0], 'acc': r[1], 'miou': r[2], 'max_loss': r[3]} for r in loss_rows],
        'zero_shot': zero,
        'best_label_free': {'selection': best_free[0], 'acc': best_free[1], 'miou': best_free[2],
                            'cfg': best_free[3:]},
        'best_oracle': {'selection': best_loss[0], 'acc': best_loss[1], 'miou': best_loss[2],
                        'max_loss': best_loss[3]},
    }

def prior_correction_sweep(base_protos, proto_lbls, prior_vec, corrupt_feats, corrupt_lbls,
                           proj, device,
                           cfgs=((0.0, 1.0), (-1.0, 5.0), (-1.0, 10.0), (-1.0, 20.0),
                                 (-1.0, 50.0), (-1.0, 100.0), (-0.5, 10.0), (-2.0, 20.0)),
                           seed=42):
    """Decision-level source-prior correction (README Pillar 3, sec 5.2).

    score(q, c) = kappa*cos(q, P_c) + tau*log(pi_c), prediction-only (never in
    the gate or the updates). The prior's strength is the ratio tau/kappa: it
    translates each boundary by (tau/kappa)*log(pi_b/pi_a), so kappa must scale
    the cosine term; with kappa=1 the log-prior (+7 for a rare class) overwhelms
    the top-2 cosine margin (~0.05) and collapses the decode onto one class.
    Reports full-scene acc + mIoU per (tau, kappa). tau=0 is the plain
    zero-shot baseline. mIoU-oriented: it trades majority precision for
    rare-class recall, so acc and mIoU must be read together.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    val_idx = perm[-100000:]
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = F.normalize(torch.sign(val_f @ proj), p=2, dim=1)
    sims = val_h @ F.normalize(base_protos, p=2, dim=1).T
    log_prior = torch.log(prior_vec.to(device).clamp(min=1e-9))
    rows = []
    for tau, kappa in cfgs:
        score = kappa * sims + tau * log_prior
        preds = proto_lbls[score.argmax(dim=1)]
        rows.append({'tau': tau, 'kappa': kappa,
                     'acc': float((preds == val_l).float().mean().item()),
                     'miou': compute_miou(preds, val_l)})
    return {'rows': rows}

def tta_oracle_decode(base_protos, proto_lbls, clean_stats, corrupt_feats, corrupt_lbls,
                      clf, proj, device, gate_cfg=None, pool_size=200000, val_size=100000,
                      seed=42):
    """TTA battery + prototype-oracle bounds, full-scene acc + mIoU on a shared val subset.

    TTA methods (self-supervised, no true labels): naive EMA over all pool points,
    soft-dual-weight EMA (uncertainty-weighted), and BN-statistic alignment decode.

    Oracle bounds (true labels, analysis only): full-label prototypes re-estimated
    from the corrupted pool, plus artifact-free oracle prototypes that EXCLUDE
    confident hallucinations (perceptron loss), high-magnitude points, and
    low-margin points, in several filter configurations. Every oracle row also
    reports the pool fraction that survives the filter.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    pool_idx = perm[:pool_size]
    val_idx = perm[-val_size:]
    pool_f = corrupt_feats[pool_idx].to(device)
    pool_l = corrupt_lbls[pool_idx].to(device)
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = torch.sign(val_f @ proj)

    def decode(protos):
        sims = F.normalize(val_h, p=2, dim=1) @ F.normalize(protos, p=2, dim=1).T
        preds = proto_lbls[sims.argmax(dim=1)]
        return {'acc': float((preds == val_l).float().mean().item()),
                'miou': compute_miou(preds, val_l)}

    rows = {'zero_shot': decode(base_protos)}

    # pool signals (artifact filters are relative to the clean-base decode)
    pool_norm = torch.norm(pool_f, p=2, dim=1)
    pool_h = torch.sign(pool_f @ proj)
    pool_sims = F.normalize(pool_h, p=2, dim=1) @ F.normalize(base_protos, p=2, dim=1).T
    top2 = torch.topk(pool_sims, 2, dim=1)
    cos_top1 = top2.values[:, 0]
    margin = (top2.values[:, 0] - top2.values[:, 1]).clamp(min=0)
    ti = torch.searchsorted(proto_lbls, pool_l)
    cos_true = pool_sims[torch.arange(len(pool_l), device=device), ti]
    loss = (cos_top1 - cos_true).clamp(min=0)
    zs_preds = proto_lbls[pool_sims.argmax(dim=1)]
    w_one = torch.ones(len(pool_f), device=device)

    # --- TTA: self-supervised prototype adaptation (no true labels) ---
    protos = weighted_mean_update(base_protos, proto_lbls, pool_f, zs_preds, w_one, proj, device)
    rows['tta_naive_ema'] = decode(protos)
    if clf is not None:
        probs = clf.predict_proba(pool_f.cpu().numpy())
        pool_conf = torch.tensor(probs.max(axis=1), device=device)
        n_z = (pool_norm - pool_norm.mean()) / (pool_norm.std() + 1e-8)
        u_epi = 1.0 - pool_conf.clamp(0.0, 1.0)
        w_sdw = fuse_uncertainties(u_epi, n_z, method='soft_dual_weight', cfg=gate_cfg)
        protos = weighted_mean_update(base_protos, proto_lbls, pool_f, zs_preds, w_sdw, proj, device)
        rows['tta_sdw'] = decode(protos)
    if clean_stats is not None:
        cmean, cstd = clean_stats[0].to(device), clean_stats[1].to(device)
        fmean = val_f.mean(dim=0)
        fstd = val_f.std(dim=0) + 1e-6
        aligned = (val_f - fmean) / fstd * cstd + cmean
        ah = torch.sign(aligned @ proj)
        asims = F.normalize(ah, p=2, dim=1) @ F.normalize(base_protos, p=2, dim=1).T
        apreds = proto_lbls[asims.argmax(dim=1)]
        rows['tta_bn_align'] = {'acc': float((apreds == val_l).float().mean().item()),
                                'miou': compute_miou(apreds, val_l)}

    # --- Oracle bounds: full-label prototypes from the corrupted pool ---
    protos = weighted_mean_update(base_protos, proto_lbls, pool_f, pool_l, w_one, proj, device)
    rows['oracle_full_label'] = decode(protos)

    # --- Artifact-free oracle prototypes (true labels, artifact points excluded) ---
    def oracle_masked(mask):
        protos = weighted_mean_update(base_protos, proto_lbls, pool_f, pool_l, w_one,
                                      proj, device, mask=mask.float())
        return {**decode(protos), 'frac': float(mask.float().mean().item())}

    masks = {
        'af_loss<=0.15': loss <= 0.15,
        'af_loss<=0.05': loss <= 0.05,
        'af_loss<=0.02': loss <= 0.02,
        'af_norm<4': pool_norm < 4.0,
        'af_norm<6': pool_norm < 6.0,
        'af_loss0.15+norm<6': (loss <= 0.15) & (pool_norm < 6.0),
        'af_loss0.02+norm<6': (loss <= 0.02) & (pool_norm < 6.0),
        'af_margin>=0.02': margin >= 0.02,
        'af_margin>=0.05': margin >= 0.05,
        'af_loss0.15+margin0.02': (loss <= 0.15) & (margin >= 0.02),
    }
    for name, mask in masks.items():
        rows['oracle_' + name] = oracle_masked(mask)
    return rows

def oracle_pool_sweep(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj, device,
                      pool_sizes=(200000, 500000, 1000000), val_size=100000, seed=42):
    """Pool-size reconciliation for the full-label oracle (Phase 24.9 finding 4).

    The val subset is IDENTICAL across pool sizes (same seeded perm, perm[-val_size:]),
    so any full-label mIoU difference is purely a pool-size effect. Uses
    weighted_mean_update (chunked projection) so the 1M-pool case is memory-safe.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    val_idx = perm[-val_size:]
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = torch.sign(val_f @ proj)

    def decode(protos):
        sims = F.normalize(val_h, p=2, dim=1) @ F.normalize(protos, p=2, dim=1).T
        preds = proto_lbls[sims.argmax(dim=1)]
        return {'acc': float((preds == val_l).float().mean().item()),
                'miou': compute_miou(preds, val_l)}

    rows = []
    for ps in pool_sizes:
        pool_idx = perm[:ps]
        pool_f = corrupt_feats[pool_idx].to(device)
        pool_l = corrupt_lbls[pool_idx].to(device)
        w_one = torch.ones(len(pool_f), device=device)
        protos = weighted_mean_update(base_protos, proto_lbls, pool_f, pool_l, w_one,
                                      proj, device)
        rows.append({'pool_size': ps, **decode(protos)})
    return {'zero_shot': decode(base_protos), 'rows': rows}

def iteration0_label_info(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj, device,
                          pool_size=500000, val_size=100000, seed=42):
    """Iteration 0 diagnostic (tta_iterations.md): WHAT information do the labels give?

    naive EMA and the full-label oracle use the SAME weighted-mean prototype operator;
    the only difference is the per-point class assignment. So the labels' information
    is assignment, and this quantifies it per class on the pool:
      - pseudo-label precision/recall per class (how contaminated the label-free
        re-estimate's prototypes are),
      - the top contamination source per prototype (which true classes land in it),
      - per-class val IoU for zero-shot / naive EMA / full-label oracle (which classes
        the correct assignment rescues).
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    pool_idx = perm[:pool_size]
    val_idx = perm[-val_size:]
    pool_f = corrupt_feats[pool_idx].to(device)
    pool_l = corrupt_lbls[pool_idx].to(device)
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = torch.sign(val_f @ proj)

    def decode(protos):
        sims = F.normalize(val_h, p=2, dim=1) @ F.normalize(protos, p=2, dim=1).T
        preds = proto_lbls[sims.argmax(dim=1)]
        return preds, {'acc': float((preds == val_l).float().mean().item()),
                       'miou': compute_miou(preds, val_l)}

    zs_preds, zs_metrics = decode(base_protos)
    w_one = torch.ones(len(pool_f), device=device)

    pool_h = torch.sign(pool_f @ proj)
    pool_sims = F.normalize(pool_h, p=2, dim=1) @ F.normalize(base_protos, p=2, dim=1).T
    pool_zs = proto_lbls[pool_sims.argmax(dim=1)]

    class_info = {}
    for c in proto_lbls.tolist():
        t_true = int((pool_l == c).sum().item())
        t_pred = int((pool_zs == c).sum().item())
        correct = int(((pool_l == c) & (pool_zs == c)).sum().item())
        prec = correct / t_pred if t_pred > 0 else 0.0
        rec = correct / t_true if t_true > 0 else 0.0
        contam = {}
        mask = pool_zs == c
        for cc in proto_lbls.tolist():
            n = int((pool_l[mask] == cc).sum().item())
            if n > 0 and cc != c:
                contam[cc] = n
        top_contam = sorted(contam.items(), key=lambda kv: -kv[1])[:3]
        class_info[c] = {'true': t_true, 'pred': t_pred, 'prec': prec, 'rec': rec,
                         'top_contam': dict(top_contam)}

    def per_class_iou(preds):
        present = set(val_l.tolist())
        out = {}
        for c in range(1, 17):
            if c not in present:
                continue
            tp = int(((preds == c) & (val_l == c)).sum().item())
            fp = int(((preds == c) & (val_l != c)).sum().item())
            fn = int(((preds != c) & (val_l == c)).sum().item())
            d = tp + fp + fn
            out[c] = tp / d if d > 0 else 0.0
        return out

    protos_oracle = weighted_mean_update(base_protos, proto_lbls, pool_f, pool_l, w_one, proj, device)
    o_preds, o_metrics = decode(protos_oracle)
    protos_naive = weighted_mean_update(base_protos, proto_lbls, pool_f, pool_zs, w_one, proj, device)
    n_preds, n_metrics = decode(protos_naive)

    return {
        'metrics': {'zero_shot': zs_metrics, 'naive_ema': n_metrics,
                    'full_label_oracle': o_metrics},
        'pool_pseudo_label_acc': float((pool_zs == pool_l).float().mean().item()),
        'class_info': class_info,
        'per_class_val_iou': {
            'zero_shot': per_class_iou(zs_preds),
            'naive_ema': per_class_iou(n_preds),
            'full_label_oracle': per_class_iou(o_preds),
        },
    }

def iteration0_update_diag(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj, device,
                           pool_size=500000, val_size=100000, seed=42):
    """Iteration 0 diagnostic (tta_iterations.md): HOW should the prototypes be updated?

    Distinguishes two failure modes for the label-free re-estimate:
      (A) GATING problem: the correctly-assigned pseudo points are informative (their
          mean points at the oracle prototype) but are drowned out by wrong assignments.
          Fix = weight/gate the pseudo-labels toward the correct subset.
      (B) OVERRUN problem: the correct subset is too small or itself non-informative
          (mean far from the oracle prototype); the minority class is overrun by
          majority artifacts regardless of weighting. Fix = assignment-level repair.

    Per class c it reports:
      - pseudo precision / n_correct / n_assigned (the overrun measure)
      - cosine(naive_proto_c, oracle_proto_c)   (how far the contaminated mean is)
      - cosine(correct_subset_proto_c, oracle_proto_c) (how informative the correct points are)
    And val-side full-scene mIoU for: zero-shot / naive (pseudo) / CORRECT-SUBSET
    re-estimate (the perfect-gating bound) / full-label oracle.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    pool_idx = perm[:pool_size]
    val_idx = perm[-val_size:]
    pool_f = corrupt_feats[pool_idx].to(device)
    pool_l = corrupt_lbls[pool_idx].to(device)
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = torch.sign(val_f @ proj)
    w_one = torch.ones(len(pool_f), device=device)

    def decode(protos):
        sims = F.normalize(val_h, p=2, dim=1) @ F.normalize(protos, p=2, dim=1).T
        preds = proto_lbls[sims.argmax(dim=1)]
        return {'acc': float((preds == val_l).float().mean().item()),
                'miou': compute_miou(preds, val_l)}

    pool_h = torch.sign(pool_f @ proj)
    pool_sims = F.normalize(pool_h, p=2, dim=1) @ F.normalize(base_protos, p=2, dim=1).T
    pool_zs = proto_lbls[pool_sims.argmax(dim=1)]

    class_info = {}
    for c in proto_lbls.tolist():
        assigned = pool_zs == c
        correct = assigned & (pool_l == c)
        true_m = pool_l == c
        n_assigned = int(assigned.sum().item())
        n_correct = int(correct.sum().item())
        n_true = int(true_m.sum().item())
        prec = n_correct / n_assigned if n_assigned > 0 else 0.0
        # prototype vectors (sign-means) for cosine comparison
        def proto_vec(mask):
            if int(mask.sum().item()) < 50:
                return None
            return F.normalize(pool_h[mask].mean(dim=0), p=2, dim=0)
        v_naive = proto_vec(assigned)
        v_correct = proto_vec(correct)
        v_oracle = proto_vec(true_m)
        cos_naive = float((v_naive @ v_oracle).item()) if (v_naive is not None and v_oracle is not None) else None
        cos_correct = float((v_correct @ v_oracle).item()) if (v_correct is not None and v_oracle is not None) else None
        class_info[c] = {'prec': prec, 'n_correct': n_correct, 'n_assigned': n_assigned,
                         'n_true': n_true, 'cos_naive_oracle': cos_naive,
                         'cos_correct_oracle': cos_correct}

    # val-side: perfect-gating bound = weighted_mean_update with pseudo labels but masked
    # to the correct subset
    protos_naive = weighted_mean_update(base_protos, proto_lbls, pool_f, pool_zs, w_one, proj, device)
    correct_mask = (pool_zs == pool_l)
    protos_correct = weighted_mean_update(base_protos, proto_lbls, pool_f, pool_zs, w_one,
                                          proj, device, mask=correct_mask)
    protos_oracle = weighted_mean_update(base_protos, proto_lbls, pool_f, pool_l, w_one, proj, device)

    return {
        'metrics': {'zero_shot': decode(base_protos),
                    'naive_pseudo': decode(protos_naive),
                    'correct_subset_gate_bound': decode(protos_correct),
                    'full_label_oracle': decode(protos_oracle)},
        'class_info': class_info,
    }

def _view_beam_drop(in_vol, p):
    bs, C, h, w = in_vol.shape
    out = in_vol.clone()
    num_drop = int(h * p)
    for b in range(bs):
        idx = torch.randperm(h, device=in_vol.device)[:num_drop]
        out[b, :, idx, :] = 0
    return out

def _rot_z(th):
    c, s = float(np.cos(th)), float(np.sin(th))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def _rot_x(th):
    c, s = float(np.cos(th)), float(np.sin(th))
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

VIEW_CONFIGS = [
    ('base', dict(scale=1.0, yaw=0.0, pitch=0.0, trans=(0.0, 0.0, 0.0), dropout=0.0)),
    ('scale095_yaw5', dict(scale=0.95, yaw=5.0, pitch=0.0, trans=(0.0, 0.0, 0.0), dropout=0.0)),
    ('scale105_yawm5', dict(scale=1.05, yaw=-5.0, pitch=0.0, trans=(0.0, 0.0, 0.0), dropout=0.0)),
    ('yaw2_pitch4', dict(scale=1.0, yaw=2.0, pitch=4.0, trans=(0.0, 0.0, 0.0), dropout=0.0)),
    ('scale09_yawm8', dict(scale=0.9, yaw=-8.0, pitch=0.0, trans=(0.0, 0.0, 0.0), dropout=0.0)),
    ('dropout30', dict(scale=1.0, yaw=0.0, pitch=0.0, trans=(0.0, 0.0, 0.0), dropout=0.3)),
]

def build_mv_views(batch, means, stds, device):
    """Build the multi-view volumes for one batch. batch[0]=in_vol (normalized),
    batch[1]=proj_mask, batch[10]=proj_xyz (raw), batch[12]=proj_remission (raw).
    Returns a list of (name, view_volume) aligned to the base mask."""
    views = []
    for name, p in VIEW_CONFIGS:
        if p['dropout'] > 0.0:
            views.append((name, _view_beam_drop(batch[0].to(device), p['dropout'])))
            continue
        R = (_rot_x(np.deg2rad(p['pitch'])) @ _rot_z(np.deg2rad(p['yaw']))).to(device)
        xyz = batch[10].float().to(device)  # [B,H,W,3]
        rem = batch[12].float().to(device)  # [B,H,W]
        t = torch.tensor(p['trans'], device=device, dtype=torch.float32).view(1, 1, 1, 3)
        xyz_t = torch.einsum('bhwc,dc->bhwd', xyz, R) * p['scale'] + t
        rng = xyz_t.norm(dim=-1, keepdim=True).permute(0, 3, 1, 2)   # [B,1,H,W]
        ch_xyz = xyz_t.permute(0, 3, 1, 2)                            # [B,3,H,W]
        ch_rem = rem.unsqueeze(1)                                     # [B,1,H,W]
        raw = torch.cat([rng, ch_xyz, ch_rem], dim=1)                 # [B,5,H,W]
        mask = (batch[1].to(device) > 0).float().unsqueeze(1)
        views.append((name, ((raw - means) / stds) * mask))
    return views

def iter1_pseudo_refine(base_protos, proto_lbls, corrupt_feats, corrupt_views, corrupt_lbls,
                        clf, proj, device, pool_size=500000, val_size=200000, seed=42):
    """Iteration 1 diagnostic (tta_iterations.md): better label-free ASSIGNMENT sources.

    Iteration 0.1 showed gating the existing pseudo-labels cannot reach the oracle
    (rare-class recall starvation); the re-estimate needs better assignments. This
    tests two label-free refinements over the 10kD zero-shot pseudo-labels:
      1. LP-pseudo: the 128D linear probe's assignments (clean-trained, label-free).
      2. Multi-view augmented consensus (MVAC): decode each augmented view (jitter /
         beam-drop / density) with the LP (probability average) and with the 10kD
         prototypes (cosine-softmax average); the averaged assignment is the refined
         pseudo-label.
    All re-estimates use the SAME weighted_mean_update operator and a shared seeded
    pool/val split, so any mIoU difference is purely the assignment source.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    pool_idx = perm[:pool_size]
    val_idx = perm[-val_size:]
    pool_f = corrupt_feats[pool_idx].to(device)
    pool_l = corrupt_lbls[pool_idx].to(device)
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = torch.sign(val_f @ proj)
    w_one = torch.ones(len(pool_f), device=device)

    def decode(protos):
        sims = F.normalize(val_h, p=2, dim=1) @ F.normalize(protos, p=2, dim=1).T
        preds = proto_lbls[sims.argmax(dim=1)]
        return {'acc': float((preds == val_l).float().mean().item()),
                'miou': compute_miou(preds, val_l)}

    def reestimate(pseudo):
        return weighted_mean_update(base_protos, proto_lbls, pool_f, pseudo, w_one, proj, device)

    zs = decode(base_protos)
    oracle = decode(reestimate(pool_l))

    # 10kD zero-shot pseudo-labels on the pool (the baseline assignment source)
    pool_h = torch.sign(pool_f @ proj)
    pool_sims = F.normalize(pool_h, p=2, dim=1) @ F.normalize(base_protos, p=2, dim=1).T
    zs_pseudo = proto_lbls[pool_sims.argmax(dim=1)]
    zs_res = decode(reestimate(zs_pseudo))

    # view features on the pool (aligned to pool base indices)
    pool_views = [v[pool_idx].to(device) for v in corrupt_views]

    # 1. LP-pseudo (base view). NOTE: predict_proba().argmax() returns the COLUMN INDEX,
    # not the class value (misaligned if classes_ omits a rare class); map back through
    # clf.classes_ exactly like clf.predict() does.
    lp_probs = [torch.tensor(clf.predict_proba(vf.cpu().numpy())) for vf in pool_views]
    classes = torch.tensor(clf.classes_)
    lp_pseudo = classes[lp_probs[0].argmax(dim=1)].to(device)
    lp_res = decode(reestimate(lp_pseudo))

    # 2. MVAC-LP: average LP probabilities across views
    avg_lp = torch.stack(lp_probs, dim=0).mean(dim=0)
    mvac_lp_pseudo = classes[avg_lp.argmax(dim=1)].to(device)
    mvac_lp_res = decode(reestimate(mvac_lp_pseudo))

    # 3. MVAC-proto: average cosine-softmax over views in 10kD
    def proto_probs(feat):
        h = F.normalize(torch.sign(feat @ proj), p=2, dim=1)
        s = h @ F.normalize(base_protos, p=2, dim=1).T
        return F.softmax(10.0 * s, dim=1)
    avg_pp = torch.stack([proto_probs(vf) for vf in pool_views], dim=0).mean(dim=0)
    mvac_proto_pseudo = proto_lbls[avg_pp.argmax(dim=1)]
    mvac_proto_res = decode(reestimate(mvac_proto_pseudo))

    pseudo_acc = {
        '10kD_zero_shot': float((pool_l == zs_pseudo).float().mean().item()),
        'LP_base': float((pool_l == lp_pseudo).float().mean().item()),
        'MVAC_LP': float((pool_l == mvac_lp_pseudo).float().mean().item()),
        'MVAC_proto': float((pool_l == mvac_proto_pseudo).float().mean().item()),
    }
    # self-check: is the pool representative? LP accuracy on the val subset and the
    # class-0 (unlabeled) fraction of pool vs val disambiguate a pool-composition
    # artifact from a genuine LP-assignment failure.
    lp_val_preds = torch.tensor(clf.predict(val_f.cpu().numpy())).to(device)
    val_lp_acc = float((lp_val_preds == val_l).float().mean().item())
    pool_norm_mean = float(pool_f.norm(p=2, dim=1).mean().item())
    val_norm_mean = float(val_f.norm(p=2, dim=1).mean().item())

    def class_counts(lbls):
        out = {}
        for c in proto_lbls.tolist():
            out[str(c)] = int((lbls == c).sum().item())
        return out

    return {
        'metrics': {'zero_shot': zs, 'zs_pseudo_reestimate': zs_res, 'LP_pseudo': lp_res,
                    'MVAC_LP': mvac_lp_res, 'MVAC_proto': mvac_proto_res, 'oracle': oracle},
        'pseudo_acc': pseudo_acc,
        'self_check': {
            'pool_class0_frac': float((pool_l == 0).float().mean().item()),
            'val_class0_frac': float((val_l == 0).float().mean().item()),
            'LP_acc_pool': pseudo_acc['LP_base'],
            'LP_acc_val': val_lp_acc,
            'norm_mean_pool': pool_norm_mean,
            'norm_mean_val': val_norm_mean,
            'pool_class_counts': class_counts(pool_l),
            'val_class_counts': class_counts(val_l),
        },
        'views': [name for name, _ in VIEW_CONFIGS],
    }

def iter2_balanced_reestimate(base_protos, proto_lbls, prior_vec, corrupt_feats, corrupt_lbls,
                              proj, device, pool_size=500000, val_size=200000, seed=42,
                              tau=2.0, n_sinkhorn_iters=100):
    """Iteration 2 diagnostic (tta_iterations.md): source-prior-balanced pseudo-assignment
    for the prototype re-estimate (SHOT diversity / Sinkhorn-Knopp, no backprop).

    Iteration 0 showed the oracle's gain is rare-class assignment recall (73k true
    Traffic-sign points, 32 correct pseudo-assignments); argmax pseudo-labels starve the
    rare classes. Sinkhorn forces the re-estimate pool's class marginals to match the
    source class frequencies, guaranteeing rare-class support. Both a hard (argmax of the
    balanced matrix) and a soft (P_bal-weighted prototype mean) re-estimate are evaluated
    against zero-shot, zs-pseudo, and the full-label oracle.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    pool_idx = perm[:pool_size]
    val_idx = perm[-val_size:]
    pool_f = corrupt_feats[pool_idx].to(device)
    pool_l = corrupt_lbls[pool_idx].to(device)
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = torch.sign(val_f @ proj)
    w_one = torch.ones(len(pool_f), device=device)

    def decode(protos):
        sims = F.normalize(val_h, p=2, dim=1) @ F.normalize(protos, p=2, dim=1).T
        preds = proto_lbls[sims.argmax(dim=1)]
        return {'acc': float((preds == val_l).float().mean().item()),
                'miou': compute_miou(preds, val_l)}

    def reestimate(pseudo):
        return weighted_mean_update(base_protos, proto_lbls, pool_f, pseudo, w_one, proj, device)

    zs = decode(base_protos)
    oracle = decode(reestimate(pool_l))

    pool_h = torch.sign(pool_f @ proj)
    sims = F.normalize(pool_h, p=2, dim=1) @ F.normalize(base_protos, p=2, dim=1).T
    zs_pseudo = proto_lbls[sims.argmax(dim=1)]
    zs_res = decode(reestimate(zs_pseudo))

    # Sinkhorn-Knopp: force column marginals toward the source prior over the pool.
    K = len(proto_lbls)
    n = len(pool_f)
    pri = prior_vec.to(device).float()
    pri = pri.clamp(min=1e-6)
    b = (pri / pri.sum()) * n
    # Sinkhorn on exp(tau*sims) with a SMALL temperature: a peaked (near one-hot)
    # P locks the column sums to the argmax marginals and the prior cannot be enforced.
    # exp(0.1-1.0 * shifted sims) keeps enough row spread for Sinkhorn to redistribute.
    sims_shifted = sims - sims.max(dim=1, keepdim=True).values
    results = {}
    best = None
    for tau in [0.1, 0.3, 0.5, 1.0]:
        P = torch.exp(tau * sims_shifted)
        u = torch.ones(n, device=device)
        for _ in range(n_sinkhorn_iters):
            v = b / (P.T @ u + 1e-9)
            u = 1.0 / (P @ v + 1e-9)
        P_bal = P * (u[:, None] * v[None, :])
        bal_pseudo = proto_lbls[P_bal.argmax(dim=1)]
        counts = torch.bincount(bal_pseudo, minlength=K)[:K]
        sup_match = float((counts - b).abs().sum().item()) / (2.0 * n)  # 0 = perfect match to prior
        res = decode(reestimate(bal_pseudo))
        results[f'tau{tau}'] = {'miou': res['miou'], 'acc': res['acc'],
                                'sup_match': sup_match}
        if best is None or sup_match < best['sup_match']:
            best = {'tau': tau, 'miou': res['miou'], 'acc': res['acc'],
                    'sup_match': sup_match, 'counts': counts}

    # use the best-matching tau for the soft re-estimate and support report
    P = torch.exp(best['tau'] * sims_shifted)
    u = torch.ones(n, device=device)
    for _ in range(n_sinkhorn_iters):
        v = b / (P.T @ u + 1e-9)
        u = 1.0 / (P @ v + 1e-9)
    P_bal = P * (u[:, None] * v[None, :])

    # soft re-estimate: prototype_c = normalize(sum_i P_bal[i,c] * sign(z_i @ proj))
    S = torch.zeros(K, proj.shape[1], device=device)
    for c in range(K):
        S[c] = (P_bal[:, c].unsqueeze(1) * pool_h).sum(dim=0)
    protos_soft = F.normalize(S, p=2, dim=1)
    sup_vec = P_bal.sum(dim=0)
    keep_base = sup_vec < 1.0
    protos_soft[keep_base] = F.normalize(base_protos[keep_base], p=2, dim=1)
    bal_soft_res = decode(protos_soft)

    bal_hard_pseudo = proto_lbls[P_bal.argmax(dim=1)]

    # verify the guardrail: balanced assignment's per-class support vs the prior
    counts_zs = {str(c): int((zs_pseudo == c).sum().item()) for c in proto_lbls.tolist()}
    counts_bal = {str(c): int((bal_hard_pseudo == c).sum().item()) for c in proto_lbls.tolist()}
    prior_counts = {str(c): int(round((pri[i] / pri.sum() * n).item()))
                    for i, c in enumerate(proto_lbls.tolist())}

    return {
        'metrics': {'zero_shot': zs, 'zs_pseudo_reestimate': zs_res,
                    'balanced_hard': {'acc': best['acc'], 'miou': best['miou']},
                    'balanced_soft': bal_soft_res,
                    'oracle': oracle},
        'pseudo_acc': {
            'zs': float((pool_l == zs_pseudo).float().mean().item()),
            'balanced_hard': float((pool_l == bal_hard_pseudo).float().mean().item()),
        },
        'sinkhorn': {'per_tau': results, 'best_tau': best['tau'],
                     'sup_match_best': best['sup_match']},
        'support': {'zs': counts_zs, 'balanced': counts_bal, 'prior': prior_counts},
    }

def react_test(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj, device,
               thresholds=(3.0, 4.0, 5.0, 6.0, 8.0, 1e9), val_size=100000, seed=42):
    """ReAct test (Sun et al., NeurIPS 2021): clip the 128D feature norms before the
    HDC projection + Sign() binarization.

    The autopsy's lead discriminator is magnitude inflation (fog/crosstalk 128D norms
    ~7 vs clean ~4.8, 88% in the norm >= 4 poison band), and the binarized decode fails
    (BinCos 0.05-0.08). ReAct tests whether the high-norm artifacts overpower the angular
    structure of the projection: clipping the features to each threshold preserves the
    direction but caps the magnitude, and we measure the frozen-prototype decode and the
    binarized clean<->fog mean cosine per threshold. Zero-training, forward-pass.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    val_idx = perm[-val_size:]
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    norms = torch.norm(val_f, p=2, dim=1)
    rows = []
    for t in thresholds:
        scale = torch.clamp(t / norms, max=1.0).unsqueeze(1)
        clipped = val_f * scale
        h = torch.sign(clipped @ proj)
        sims = F.normalize(h, p=2, dim=1) @ F.normalize(base_protos, p=2, dim=1).T
        preds = proto_lbls[sims.argmax(dim=1)]
        acc = float((preds == val_l).float().mean().item())
        miou = compute_miou(preds, val_l)
        bcs = []
        for i, c in enumerate(proto_lbls.tolist()):
            mask = val_l == c
            if int(mask.sum().item()) >= 500:
                bh = torch.sign(clipped[mask][:20000] @ proj).float().mean(dim=0)
                bcs.append(F.normalize(bh, p=2, dim=0) @ F.normalize(base_protos[i], p=2, dim=0))
        bin_cos = float(torch.stack(bcs).mean().item()) if bcs else 0.0
        rows.append({'threshold': float('inf') if t == 1e9 else float(t),
                     'acc': acc, 'miou': miou, 'bin_cos': bin_cos,
                     'frac_clipped': float((norms > t).float().mean().item()) if t != 1e9 else 0.0})
    return {'rows': rows}

def deep_label_analysis(base_protos, proto_lbls, clean_feats, clean_lbls, corrupt_feats,
                        corrupt_lbls, clf, proj, device, seed=42, max_pts_per_class=30000):
    """Iteration 4 (deep label-information analysis, tta_iterations.md).

    Three parts, all aimed at WHAT the ground-truth labels carry that the features
    and the 10kD prototypes cannot derive:

      A. Feature geometry, clean vs corrupt, per class: centroid cosine shift, norm
         inflation, intra-class tightness, and inter-class absorption (distance to the
         nearest OTHER class centroid, clean vs corrupt). Split classes into survivors
         (corrupt decode keeps them) and collapsers (corrupt decode kills them) to see
         what makes the fragile classes fragile under corruption yet fine under
         supervised clean training.
      B. Pseudo-label error analysis: per-true-class confusion for the 10kD zs and LP
         assignment sources, prototype contamination (precision + cosine to the oracle
         prototype per class), and the per-class IoU impact of the assignment errors on
         the re-estimate.
      C. Recoverability / confidence: does ANY label-free signal (norm, margin, LP
         confidence, cos128, oracle perceptron loss) separate the points the oracle
         rescues (zs-wrong, oracle-right) from the points wrong even under the oracle?
         AUROC ~0.5 on every signal proves the recoverability information is label-only.
    """
    torch.manual_seed(seed)

    # ---- subsample clean per class ----
    clean_parts_f, clean_parts_l = [], []
    for c in proto_lbls.tolist():
        idx = torch.nonzero(clean_lbls == c).flatten()
        if len(idx) > max_pts_per_class:
            idx = idx[torch.randperm(len(idx))[:max_pts_per_class]]
        clean_parts_f.append(clean_feats[idx])
        clean_parts_l.append(clean_lbls[idx])
    clean_f = torch.cat(clean_parts_f, dim=0)
    clean_l = torch.cat(clean_parts_l, dim=0)

    # ---- corrupt pool / val (shared split) ----
    perm = torch.randperm(len(corrupt_feats))
    pool_idx = perm[:500000]
    val_idx = perm[-100000:]
    pool_f = corrupt_feats[pool_idx].to(device)
    pool_l = corrupt_lbls[pool_idx].to(device)
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = torch.sign(val_f @ proj)
    w_one = torch.ones(len(pool_f), device=device)

    def decode(protos):
        sims = F.normalize(val_h, p=2, dim=1) @ F.normalize(protos, p=2, dim=1).T
        preds = proto_lbls[sims.argmax(dim=1)]
        return preds, compute_miou(preds, val_l)

    def per_class_iou(preds):
        present = set(val_l.tolist())
        out = {}
        for c in range(1, 17):
            if c not in present:
                continue
            tp = int(((preds == c) & (val_l == c)).sum().item())
            fp = int(((preds == c) & (val_l != c)).sum().item())
            fn = int(((preds != c) & (val_l == c)).sum().item())
            d = tp + fp + fn
            out[c] = tp / d if d > 0 else 0.0
        return out

    # ---- A. per-class geometry + survivor/collapser split ----
    clean_centers = {}
    corrupt_centers = {}
    geometry = {}
    lp_cls = torch.tensor(clf.predict(clean_f.numpy()))  # LP per-class clean acc
    lp_pool_preds = torch.tensor(clf.predict(pool_f.cpu().numpy())).to(device)

    for c in proto_lbls.tolist():
        cm = clean_l == c
        pm = pool_l == c
        cf_c = clean_f[cm].to(device)
        pf_c = pool_f[pm]
        if len(cf_c) < 200 or len(pf_c) < 200:
            geometry[str(c)] = None
            continue
        c_center = F.normalize(cf_c.mean(dim=0), p=2, dim=0)
        p_center = F.normalize(pf_c.mean(dim=0), p=2, dim=0)
        clean_centers[c] = c_center
        corrupt_centers[c] = p_center
        c_tight = float((F.normalize(cf_c, p=2, dim=1) @ c_center).mean().item())
        p_tight = float((F.normalize(pf_c, p=2, dim=1) @ p_center).mean().item())
        # inter-class absorption: clean/corrupt distance to nearest OTHER clean centroid
        others = [clean_centers[o] for o in clean_centers if o != c]
        if others:
            other_m = F.normalize(torch.stack(others), p=2, dim=1)
            c_d = float((1.0 - other_m @ c_center).min().item())
            p_d = float((1.0 - other_m @ p_center).min().item())
        else:
            c_d = p_d = None
        geometry[str(c)] = {
            'cos_shift': float((c_center @ p_center).item()),
            'norm_clean': float(cf_c.norm(dim=1).mean().item()),
            'norm_corrupt': float(pf_c.norm(dim=1).mean().item()),
            'tight_clean': c_tight, 'tight_corrupt': p_tight,
            'nearest_other_clean_dist': c_d, 'nearest_other_corrupt_dist': p_d,
            'lp_clean_acc': float((lp_cls[cm] == c).float().mean().item()),
            'lp_corrupt_acc': float((lp_pool_preds[pm] == c).float().mean().item()),
        }

    # ---- B. pseudo-label confusion + re-estimate impact ----
    # pool_sims via chunked projection (pool_h at 500k x 10kD = 20GB; compute per chunk).
    base_norm = F.normalize(base_protos, p=2, dim=1)
    sims_chunks = []
    for start in range(0, len(pool_f), 50000):
        hc = torch.sign(pool_f[start:start + 50000] @ proj)
        sims_chunks.append(F.normalize(hc, p=2, dim=1) @ base_norm.T)
    pool_sims = torch.cat(sims_chunks, dim=0)
    zs_pseudo = proto_lbls[pool_sims.argmax(dim=1)]
    protos_oracle = weighted_mean_update(base_protos, proto_lbls, pool_f, pool_l, w_one, proj, device)
    protos_zs = weighted_mean_update(base_protos, proto_lbls, pool_f, zs_pseudo, w_one, proj, device)
    protos_lp = weighted_mean_update(base_protos, proto_lbls, pool_f, lp_pool_preds, w_one, proj, device)

    zs_val_preds, zs_miou = decode(protos_zs)
    lp_val_preds, lp_miou = decode(protos_lp)
    oracle_val_preds, oracle_miou = decode(protos_oracle)
    zs0_val_preds, _ = decode(base_protos)

    confusion = {}
    contamination = {}
    for c in proto_lbls.tolist():
        true_mask = pool_l == c
        n_true = int(true_mask.sum().item())
        if n_true == 0:
            continue
        # top-3 destinations of this class's points under zs
        dest = zs_pseudo[true_mask]
        counts = torch.bincount(dest, minlength=len(proto_lbls))
        top = sorted(zip(proto_lbls.tolist(), counts.tolist()), key=lambda kv: -kv[1])[:3]
        confusion[str(c)] = {'n_true': n_true,
                             'top_dest': [(d, n) for d, n in top if n > 0]}
        # prototype contamination from the zs assignment
        for src_name, src_preds, protos in [('zs', zs_pseudo, protos_zs),
                                           ('lp', lp_pool_preds, protos_lp)]:
            assigned = src_preds == c
            n_assigned = int(assigned.sum().item())
            prec = float(((pool_l[assigned] == c).float().mean().item())) if n_assigned > 0 else 0.0
            idx = (proto_lbls == c).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                cos_proto = float((F.normalize(protos[idx[0]], p=2, dim=0)
                                   @ F.normalize(protos_oracle[idx[0]], p=2, dim=0)).item())
            else:
                cos_proto = 0.0
            contamination.setdefault(src_name, {})[str(c)] = {'prec': prec, 'cos_to_oracle_proto': cos_proto}

    class_impact = {
        'zs_reestimate': per_class_iou(zs_val_preds),
        'lp_reestimate': per_class_iou(lp_val_preds),
        'oracle': per_class_iou(oracle_val_preds),
    }

    # ---- C. recoverability / confidence ----
    recover = {}
    # oracle-recovered = zs-wrong but oracle-right on the val
    both_wrong = (zs0_val_preds != val_l) & (oracle_val_preds != val_l)
    recovered = (zs0_val_preds != val_l) & (oracle_val_preds == val_l)
    if recovered.sum() > 50 and both_wrong.sum() > 50:
        norm = torch.norm(val_f, p=2, dim=1)
        sims10 = F.normalize(val_h, p=2, dim=1) @ F.normalize(base_protos, p=2, dim=1).T
        top2 = torch.topk(sims10, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).clamp(min=0)
        lp_probs = torch.tensor(clf.predict_proba(val_f.cpu().numpy())).to(device)
        lp_conf = lp_probs.max(dim=1).values
        lp_margin = (torch.topk(lp_probs, 2, dim=1).values[:, 0]
                     - torch.topk(lp_probs, 2, dim=1).values[:, 1]).clamp(min=0)
        lp_entropy = -(lp_probs * torch.log(lp_probs.clamp(min=1e-12))).sum(dim=1)
        ti = torch.searchsorted(proto_lbls, val_l)
        cos_true = sims10[torch.arange(len(val_l), device=device), ti]
        loss = (top2.values[:, 0] - cos_true).clamp(min=0)
        # 128D clean-prototype signals (need clean 128D means; recompute from the pool's
        # class assignment is not available here, so use the val's own clean-side geometry
        # via the LP's class-conditional mean is skipped; use cos128 from the 10kD sims' top-1)
        cos128 = top2.values[:, 0]
        # per-class z-scored norm (relative magnitude within the predicted class)
        norm_z = torch.zeros_like(norm)
        for c in proto_lbls.tolist():
            m = zs0_val_preds == c
            if int(m.sum().item()) > 10:
                norm_z[m] = (norm[m] - norm[m].mean()) / (norm[m].std() + 1e-8)
        # kNN local agreement: fraction of k nearest 128D neighbors sharing the zs prediction
        # (subsample for the cdist cost; recovered/stuck balance kept by sampling equally)
        knn_agree = torch.zeros(len(val_f), device=device)
        kn = min(20000, len(val_f))
        torch.manual_seed(0)
        sub = torch.randperm(len(val_f))[:kn]
        sub_f = val_f[sub]
        d = torch.cdist(sub_f, sub_f)
        nb = torch.topk(d, 11, dim=1, largest=False).indices[:, 1:]
        agree = (zs0_val_preds[sub][nb] == zs0_val_preds[sub].unsqueeze(1)).float().mean(dim=1)
        knn_agree[sub] = agree

        signals = [('norm', norm), ('margin', margin), ('cos128', cos128), ('lp_conf', lp_conf),
                   ('lp_margin', lp_margin), ('lp_entropy', lp_entropy),
                   ('norm_z', norm_z), ('knn_agree', knn_agree), ('oracle_loss', loss)]
        y = torch.cat([torch.ones(recovered.sum()), torch.zeros(both_wrong.sum())]).numpy()
        for name, sig in signals:
            x = torch.cat([sig[recovered], sig[both_wrong]]).cpu().numpy()
            if x.std() > 0:
                auc = roc_auc_score(y, x)
            else:
                auc = 0.5
            recover[name] = {
                'auc': float(auc),
                'mean_recovered': float(sig[recovered].mean().item()),
                'mean_stuck': float(sig[both_wrong].mean().item()),
            }
        # combined signal: logistic regression over the label-free signals (oracle labels
        # used only as the diagnostic target; trains on half, evaluates on the other half).
        # A high AUROC here means a JOINT label-free signal separates recovered from stuck.
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        X_all = torch.stack([torch.cat([s[recovered], s[both_wrong]]) for s in
                             [norm, margin, cos128, lp_conf, lp_margin, lp_entropy,
                              norm_z, knn_agree]], dim=1).cpu().numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.5, random_state=0)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_tr, y_tr)
        recover['combined_lr'] = {'auc': float(roc_auc_score(y_te, lr.decision_function(X_te)))}

    return {
        'geometry': geometry,
        'pseudo': {'zs_acc_pool': float((pool_l == zs_pseudo).float().mean().item()),
                   'lp_acc_pool': float((pool_l == lp_pool_preds).float().mean().item()),
                   'zs_miou': zs_miou, 'lp_miou': lp_miou, 'oracle_miou': oracle_miou},
        'confusion': confusion,
        'contamination': contamination,
        'class_impact': class_impact,
        'recover': recover,
    }

def combined_gate_sweep(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, clf, proj, device,
                        pool_size=500000, val_size=100000, seed=42, strengths=(0.5, 1.0, 2.0)):
    """Path-validation sweep (tta_iterations.md): does a COMBINED label-free recoverability
    gate produce full-scene gains on fog/crosstalk without collapsing the other conditions?

    The recoverability battery (deep_label_analysis Part C) showed a JOINT label-free
    signal separates oracle-recovered from oracle-stuck points on both conditions (combined
    AUROC fog 0.799, crosstalk 0.680). This tests whether that joint signal is exploitable:
    it builds FIXED z-scored linear-combination configs (no oracle labels) and, for each,
    weights the prototype re-estimate toward the recoverable points, measuring the
    FULL-SCENE mIoU (the deployable direction). Decode-side retained mIoU is also reported
    as the older diagnostic framing. The all-conditions sweep is the collapse check.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(len(corrupt_feats))
    pool_idx = perm[:pool_size]
    val_idx = perm[-val_size:]
    pool_f = corrupt_feats[pool_idx].to(device)
    pool_l = corrupt_lbls[pool_idx].to(device)
    val_f = corrupt_feats[val_idx].to(device)
    val_l = corrupt_lbls[val_idx].to(device)
    val_h = torch.sign(val_f @ proj)
    w_one = torch.ones(len(pool_f), device=device)

    def decode(protos):
        sims = F.normalize(val_h, p=2, dim=1) @ F.normalize(protos, p=2, dim=1).T
        preds = proto_lbls[sims.argmax(dim=1)]
        return {'acc': float((preds == val_l).float().mean().item()),
                'miou': compute_miou(preds, val_l)}

    # ---- signal battery on the pool (chunked projection to bound memory) ----
    base_norm = F.normalize(base_protos, p=2, dim=1)
    sims = []
    for start in range(0, len(pool_f), 50000):
        hc = torch.sign(pool_f[start:start + 50000] @ proj)
        sims.append(F.normalize(hc, p=2, dim=1) @ base_norm.T)
    sims = torch.cat(sims, dim=0)
    zs_pseudo = proto_lbls[sims.argmax(dim=1)]
    protos_zs = weighted_mean_update(base_protos, proto_lbls, pool_f, zs_pseudo, w_one, proj, device)

    norm = torch.norm(pool_f, p=2, dim=1)
    top2 = torch.topk(sims, 2, dim=1)
    margin = (top2.values[:, 0] - top2.values[:, 1]).clamp(min=0)
    cos128 = top2.values[:, 0]
    lp_probs = torch.tensor(clf.predict_proba(pool_f.cpu().numpy())).to(device)
    lp_conf = lp_probs.max(dim=1).values
    lp_margin = (torch.topk(lp_probs, 2, dim=1).values[:, 0]
                 - torch.topk(lp_probs, 2, dim=1).values[:, 1]).clamp(min=0)
    lp_entropy = -(lp_probs * torch.log(lp_probs.clamp(min=1e-12))).sum(dim=1)
    norm_z = torch.zeros_like(norm)
    for c in proto_lbls.tolist():
        m = zs_pseudo == c
        if int(m.sum().item()) > 10:
            norm_z[m] = (norm[m] - norm[m].mean()) / (norm[m].std() + 1e-8)

    def z(sig):
        return (sig - sig.mean()) / (sig.std() + 1e-8)

    # configs: signed z-scored linear combos (signs from the Part C AUROC means)
    configs = {
        'norm': z(norm_z),
        'lp': z(lp_conf) + z(lp_margin),
        'norm+lp': z(norm_z) + z(lp_conf) + z(lp_margin),
        'full': z(norm_z) + z(lp_conf) + z(lp_margin) - z(cos128) - z(lp_entropy) - z(margin),
        'full_no_norm': z(lp_conf) + z(lp_margin) - z(cos128) - z(lp_entropy) - z(margin),
    }

    results = {}
    for cname, score in configs.items():
        # (a) decode-side retained mIoU at top-25/50/75%
        dec = {}
        for frac in (0.25, 0.50, 0.75):
            k = max(int(len(score) * frac), 1)
            keep = torch.topk(score, k).indices
            preds = proto_lbls[sims[keep].argmax(dim=1)]
            lbl = pool_l[keep]
            dec[f'top{int(frac*100)}'] = compute_miou(preds, lbl)
        # (b) re-estimate-side weighting (full-scene)
        s_min, s_max = score.min(), score.max()
        w_norm = (score - s_min) / (s_max - s_min + 1e-9)
        reest = {}
        for w in strengths:
            weights = 1.0 + w * w_norm
            protos = weighted_mean_update(base_protos, proto_lbls, pool_f, zs_pseudo,
                                          weights, proj, device)
            reest[f'w{w}'] = decode(protos)
        results[cname] = {'decode_retained': dec, 'reestimate': reest}

    zs = decode(base_protos)
    zs_reest = decode(protos_zs)
    oracle = decode(weighted_mean_update(base_protos, proto_lbls, pool_f, pool_l, w_one, proj, device))

    return {
        'metrics': {'zero_shot': zs, 'zs_pseudo_reestimate': zs_reest, 'oracle': oracle},
        'configs': results,
    }
