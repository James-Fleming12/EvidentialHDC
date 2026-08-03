import torch
import torch.nn.functional as F
import numpy as np

NUM_CLASSES = 17
NORM_BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, float('inf'))]

def _band_fracs(norms):
    fracs = {}
    for lo, hi in NORM_BANDS:
        fracs[f'[{lo},{hi})'] = float(((norms >= lo) & (norms < hi)).float().mean().item())
    return fracs

def deep_headroom_diagnostics(clean_feats, clean_lbls, fog_feats, fog_lbls, device,
                              max_pts_per_class=50000):
    """Feature-space diagnostics beyond the headroom metrics.

    1. Class-mean quality (128D): clean vs fog mean norms per class.
       Near-zero fog means => artifact dilution dominates the centroid decode.
    2. Binarized (10kD, seeded) class-mean quality: mean norms + clean<->fog cosine.
       Near-zero binarized fog means => collapsed points randomize HDC prototypes.
    3. Magnitude segregation: fraction of points in norm bands (clean vs fog).
    4. Query-gate feasibility: fog prototype accuracy per norm band (frozen clean
       prototypes) -- the empirical answer to "should we veto low-norm points".
    5. Class-manifold anisotropy: top-eigenvalue/trace ellipticity per class
       (1/128 = isotropic, ~1 = rank-1) for clean vs fog.
    """
    res = {}
    clean_means, fog_means = {}, {}
    for c in range(NUM_CLASSES):
        cm = clean_feats[clean_lbls == c]
        fm = fog_feats[fog_lbls == c]
        if len(cm) > 0:
            clean_means[c] = cm.mean(dim=0)
        if len(fm) > 0:
            fog_means[c] = fm.mean(dim=0)

    # 1. 128D class-mean norms
    cn = {c: clean_means[c].norm().item() for c in clean_means}
    fn = {c: fog_means[c].norm().item() for c in fog_means}
    res['mean_norm_clean'] = float(np.mean(list(cn.values())))
    res['mean_norm_fog'] = float(np.mean(list(fn.values())))
    res['mean_norm_ratio_fog_clean'] = res['mean_norm_fog'] / max(res['mean_norm_clean'], 1e-8)
    res['class_mean_norm_ratio'] = {c: fn[c] / max(cn[c], 1e-8) for c in cn if c in fn}

    # 2. Binarized 10kD class-mean quality (seeded projection)
    torch.manual_seed(42)
    proj = ((torch.rand(128, 10000) > 0.5).float() * 2 - 1).to(device)

    def binarized_mean(feats_subset):
        h = torch.sign(feats_subset @ proj).float()
        return h.mean(dim=0)

    bin_c, bin_f = {}, {}
    for c in cn:
        cm = clean_feats[clean_lbls == c][:max_pts_per_class].to(device)
        bin_c[c] = binarized_mean(cm)
        fm = fog_feats[fog_lbls == c][:max_pts_per_class].to(device)
        if len(fm) > 0:
            bin_f[c] = binarized_mean(fm)
    bcn = {c: bin_c[c].norm().item() for c in bin_c}
    bfn = {c: bin_f[c].norm().item() for c in bin_f}
    res['binarized_mean_norm_clean'] = float(np.mean(list(bcn.values())))
    res['binarized_mean_norm_fog'] = float(np.mean(list(bfn.values())))
    res['binarized_mean_norm_ratio'] = res['binarized_mean_norm_fog'] / max(res['binarized_mean_norm_clean'], 1e-8)
    cos_sims = {}
    for c in bin_c:
        if c in bin_f:
            cos_sims[c] = float(F.cosine_similarity(
                F.normalize(bin_c[c].unsqueeze(0)), F.normalize(bin_f[c].unsqueeze(0))).item())
    res['binarized_mean_cosine_sim_avg'] = float(np.mean(list(cos_sims.values()))) if cos_sims else 0.0

    # 3. Magnitude segregation bands
    clean_norms = clean_feats.norm(dim=1)
    fog_norms = fog_feats.norm(dim=1)
    res['clean_norm_bands'] = _band_fracs(clean_norms)
    res['fog_norm_bands'] = _band_fracs(fog_norms)

    # 4. Query-gate feasibility: fog prototype acc per norm band (chunked)
    proto = torch.stack([F.normalize(clean_means[c], p=2, dim=0) for c in sorted(clean_means)]).to(device)
    plbl = torch.tensor(sorted(clean_means)).to(device)
    band_acc = {}
    for lo, hi in NORM_BANDS:
        m = (fog_norms >= lo) & (fog_norms < hi)
        n = int(m.sum().item())
        if n < 1000:
            band_acc[f'[{lo},{hi})'] = None
            continue
        correct = 0
        idx_all = m.nonzero(as_tuple=True)[0]
        for s in range(0, n, 500000):
            idx = idx_all[s:s + 500000]
            z = F.normalize(fog_feats[idx].to(device), p=2, dim=1)
            preds = plbl[(z @ proto.T).argmax(dim=1)]
            correct += int((preds == fog_lbls[idx].to(device)).sum().item())
        band_acc[f'[{lo},{hi})'] = correct / n
    res['fog_band_proto_acc'] = band_acc

    # 5. Class-manifold anisotropy: ellipticity = lambda_max / trace of class covariance
    def ellipticity(x):
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / len(x)
        eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        tr = eig.sum()
        return float((eig[-1] / (tr + 1e-8)).item()) if tr > 1e-8 else 0.0

    ell = {}
    for c in sorted(clean_means):
        cm = clean_feats[clean_lbls == c][:20000].to(device)
        fm = fog_feats[fog_lbls == c][:20000].to(device)
        row = {'clean': None, 'fog': None}
        if len(cm) >= 500:
            row['clean'] = ellipticity(cm)
        if len(fm) >= 500:
            row['fog'] = ellipticity(fm)
        ell[c] = row
    res['class_ellipticity'] = ell
    cvals = [v['clean'] for v in ell.values() if v['clean'] is not None]
    fvals = [v['fog'] for v in ell.values() if v['fog'] is not None]
    res['ellipticity_avg_clean'] = float(np.mean(cvals)) if cvals else 0.0
    res['ellipticity_avg_fog'] = float(np.mean(fvals)) if fvals else 0.0
    return res

def print_deep_summary(deep):
    print(f"  [Deep] 128D mean-norm ratio (fog/clean): {deep['mean_norm_ratio_fog_clean']:.3f}")
    print(f"  [Deep] binarized mean-norm ratio: {deep['binarized_mean_norm_ratio']:.3f} "
          f"| clean<->fog cos sim: {deep['binarized_mean_cosine_sim_avg']:.3f}")
    print(f"  [Deep] fog norm bands: {deep['fog_norm_bands']}")
    print(f"  [Deep] fog band proto acc (query-gate feasibility): {deep['fog_band_proto_acc']}")
    print(f"  [Deep] ellipticity avg clean/fog: {deep['ellipticity_avg_clean']:.3f} / {deep['ellipticity_avg_fog']:.3f}")
