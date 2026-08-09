"""DGLSS / DGLSS++ consistency losses, adapted to the 128D HDC-input bottleneck.

The two domain-generalization methods of Kim et al. (CVPR 2023, TPAMI 2026) train
a segmentation model on a single dense source domain while augmenting sparsity and
constraining the representation. The original implementations align INTERMEDIATE
voxel features (SIFC/GMSIFC) and scene/local class-prototype correlation matrices
(SCC/LSCC). Here the same constraints are applied to the 128D bottleneck features
(the space the HDC random-projection + sign-binarization decode consumes), so the
only difference from `supcon_vib` is the representation-consistency mechanism.

  - dglss_sifc_loss:  SIFC (DGLSS)  / GMSIFC (DGLSS++, masked)
  - dglss_scc_loss:   SCC  (DGLSS)  / LSCC  (DGLSS++, local cells)

Both take the source and sparse-augmented 128D features plus the input volumes (for
the depth-presence masks that define paired/unpaired positions) and the projection
labels, and return a scalar loss.

Reference: https://github.com/gzgzys9887/DGLSS
"""
import torch
import torch.nn.functional as F
import numpy as np


def get_dglss_view(in_vol, p_range=(0.3, 0.7)):
    """DGLSS / DGLSS++ sparse augmentation: drop whole beam rows (range-view rows)
    at a per-sample rate p ~ U(p_range), dense-to-sparse only, exactly as the papers.
    This is the augmented view the SIFC/GMSIFC alignment and SCC/LSCC operate on."""
    bs, channels, h, w = in_vol.shape
    result = in_vol.clone()
    for b in range(bs):
        p = float(np.random.uniform(p_range[0], p_range[1]))
        num_drop = int(h * p)
        indices = np.random.choice(h, num_drop, replace=False)
        result[b, :, indices, :] = 0
    return result


def single_class_mask(labels):
    """GMSIFC masking (DGLSS++): exclude positions whose 3x3 local neighborhood
    spans multiple classes, so the consistency term only propagates unambiguous
    class features. Mirrors the SIFC-affinity filtering rationale."""
    if labels.dim() == 4 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    lbl_f = labels.float().unsqueeze(1)
    pad = F.pad(lbl_f, (1, 1, 1, 1), mode='replicate')
    local_max = F.max_pool2d(pad, 3, stride=1)
    local_min = -F.max_pool2d(-pad, 3, stride=1)
    pure = (local_max == local_min) & (labels.unsqueeze(1) > 0)
    return pure.squeeze(1)


def dglss_sifc_loss(z8, z8_aug, proj_labels, in_vol, in_vol_aug,
                    masked=False, tau=0.7, max_pts=1500):
    """SIFC (DGLSS) / GMSIFC (DGLSS++) on the 128D bottleneck features.

    Paired positions (present in both views) get an L1 alignment. Unpaired source
    positions are aligned to an affinity + inverse-distance weighted aggregation of
    the augmented view's paired neighbors, where the affinity is computed in the
    SOURCE view and filtered by the cosine threshold tau (and symmetrically for
    unpaired augmented positions), following the paper's Figure 3. GMSIFC masks
    multi-class local neighborhoods. The aggregation is subsampled to bound the
    quadratic cost on the full 2048x64 projection.
    """
    B, C, H, W = z8.shape
    # Adapt the projection labels + depth-presence masks to the feature resolution.
    # This is the identity for the full-res 128D bottleneck and downsamples for the
    # 1/8-res encoder stage (the standard-implementation arm): nearest label mapping
    # and max-pooled presence (a cell is occupied if any source pixel in it is).
    if (proj_labels.shape[-2], proj_labels.shape[-1]) != (H, W):
        labels = F.interpolate(proj_labels.float().unsqueeze(1), size=(H, W),
                               mode='nearest').long().squeeze(1)

        def presence(vol):
            return F.adaptive_max_pool2d((vol[:, 0:1] != 0).float(), (H, W)) > 0
    else:
        labels = proj_labels

        def presence(vol):
            return vol[:, 0:1] != 0

    valid = (labels > 0).unsqueeze(1)                         # (B,1,H,W)
    if masked:
        valid = valid & single_class_mask(labels).unsqueeze(1)
    source_present = presence(in_vol)
    beam_present = presence(in_vol_aug)
    paired = valid & source_present & beam_present
    unpaired_s = valid & source_present & ~beam_present
    unpaired_a = valid & ~source_present & beam_present

    loss = torch.tensor(0.0, device=z8.device)
    if paired.any():
        pm = paired.expand_as(z8)
        loss = loss + F.l1_loss(z8[pm], z8_aug[pm])

    z8f = z8.permute(0, 2, 3, 1).reshape(B, H * W, C)
    z8af = z8_aug.permute(0, 2, 3, 1).reshape(B, H * W, C)
    pf = paired.squeeze(1)
    us = unpaired_s.squeeze(1)
    ua = unpaired_a.squeeze(1)

    agg_losses = []
    for b in range(B):
        # direction 1: unpaired SOURCE <-> agg of paired AUG (affinity in source view)
        # direction 2: unpaired AUG   <-> agg of paired SOURCE (affinity in aug view)
        for (u_flag, f_u_view, f_aff_view, f_agg_view) in [
            (us[b], z8f[b], z8f[b], z8af[b]),
            (ua[b], z8af[b], z8af[b], z8f[b]),
        ]:
            ui = u_flag.nonzero(as_tuple=True)[0]
            pi = pf[b].nonzero(as_tuple=True)[0]
            if len(ui) == 0 or len(pi) == 0:
                continue
            if len(ui) > max_pts:
                ui = ui[torch.randperm(len(ui), device=z8.device)[:max_pts]]
            if len(pi) > max_pts:
                pi = pi[torch.randperm(len(pi), device=z8.device)[:max_pts]]
            f_u = f_u_view[ui]
            f_aff = f_aff_view[pi]
            aff = F.normalize(f_u, p=2, dim=1) @ F.normalize(f_aff, p=2, dim=1).T
            aff = aff * (aff >= tau).float()
            ui_xy = torch.stack([ui % W, ui // W], dim=1).float()
            pi_xy = torch.stack([pi % W, pi // W], dim=1).float()
            inv_dist = 1.0 / (torch.cdist(ui_xy, pi_xy) + 1e-6)
            weights = aff * inv_dist
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            f_agg = weights @ f_agg_view[pi]
            v = weights.sum(dim=1) > 0
            if v.any():
                agg_losses.append(F.l1_loss(f_u[v], f_agg[v]))
    if agg_losses:
        loss = loss + torch.stack(agg_losses).mean()
    return loss


def dglss_scc_loss(z8, z8_aug, proj_labels, in_vol, in_vol_aug,
                   local=False, cell=256, min_pts=5, max_pts=256):
    """SCC (DGLSS) / LSCC (DGLSS++): all-pairs class-prototype correlation consistency.

    SCC pools per-scan GLOBAL class prototypes from both views (the 2B scans of the
    batch) and penalizes the correlation-matrix mismatch between EVERY pair, exactly
    the DGLSS Eq. 5 form. LSCC partitions each view into spatial cells, pools per-cell
    prototypes, applies the same all-pairs consistency between cells (DGLSS++ Eq. 17
    first term), and adds the per-scan contrastive term of Eq. 17 (same-class pull on
    the normalized 128D bottleneck, within each scan).
    """
    B, C, H, W = z8.shape
    cls = proj_labels[proj_labels > 0].unique().tolist()

    # collect (class_order, prototype_matrix): per-scan global (SCC) or per-cell (LSCC),
    # pooling both views (the 2B scans of the batch, per the papers).
    protos_all = []
    for b in range(B):
        if local:
            cells = [(slice(0, H), slice(col, min(col + cell, W))) for col in range(0, W, cell)]
        else:
            cells = [(slice(0, H), slice(0, W))]
        for (rs, cs) in cells:
            for z, vol in ((z8, in_vol), (z8_aug, in_vol_aug)):
                zf = z[b, :, rs, cs].permute(1, 2, 0).reshape(-1, C)
                ls = proj_labels[b, rs, cs].reshape(-1)
                m = (vol[b, 0, rs, cs] != 0).reshape(-1)
                protos, present = {}, []
                for c in cls:
                    mm = (ls == c) & m
                    if mm.sum() >= min_pts:
                        protos[c] = F.normalize(zf[mm].mean(dim=0), p=2, dim=0)
                        present.append(c)
                if len(present) >= 2:
                    order = sorted(present)
                    protos_all.append((order, torch.stack([protos[c] for c in order])))

    loss = torch.tensor(0.0, device=z8.device)
    loss_contr = torch.tensor(0.0, device=z8.device)
    n = 0
    n_scans = 0
    for i in range(len(protos_all)):
        oi, Zi = protos_all[i]
        for j in range(i + 1, len(protos_all)):
            oj, Zj = protos_all[j]
            shared = sorted(set(oi) & set(oj))
            if len(shared) < 2:
                continue
            ioi = {c: k for k, c in enumerate(oi)}
            ioj = {c: k for k, c in enumerate(oj)}
            Zi_s = Zi[[ioi[c] for c in shared]]
            Zj_s = Zj[[ioj[c] for c in shared]]
            loss = loss + ((Zi_s @ Zi_s.T) - (Zj_s @ Zj_s.T)).pow(2).mean()
            n += 1

    if local:
        # DGLSS++ Eq. 17 contrastive term: per-scan InfoNCE on the normalized bottleneck
        # (the metric-learner embedding analog), so the local-region structure is not
        # the only supervision.
        for b in range(B):
            zf = z8[b].permute(1, 2, 0).reshape(-1, C)
            ls = proj_labels[b].reshape(-1)
            zf, ls = zf[ls > 0], ls[ls > 0]
            if len(ls) < 4:
                continue
            if len(ls) > max_pts:
                idx = torch.randperm(len(ls), device=z8.device)[:max_pts]
                zf, ls = zf[idx], ls[idx]
            zn = F.normalize(zf, p=2, dim=1)
            sim = zn @ zn.T
            self_mask = ~torch.eye(len(ls), dtype=torch.bool, device=z8.device)
            pos_mask = (ls.unsqueeze(0) == ls.unsqueeze(1)) & self_mask
            anchors = pos_mask.any(dim=1)
            if not anchors.any():
                continue
            INF = 1e4
            pos_sim = sim[anchors].masked_fill(~pos_mask[anchors], -INF)
            all_sim = sim[anchors].masked_fill(~self_mask[anchors], -INF)
            loss_contr = loss_contr + (torch.logsumexp(all_sim, dim=1)
                                       - torch.logsumexp(pos_sim, dim=1)).mean()
            n_scans += 1

    loss = loss / max(n, 1)
    if local and n_scans > 0:
        loss = loss + loss_contr / n_scans
    return loss
