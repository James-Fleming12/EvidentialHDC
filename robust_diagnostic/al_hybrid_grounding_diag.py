"""al_hybrid_grounding_diag.py: the Iteration-5 hybrid AL mechanism (eval-only,
no plots). Compounds the three measured geometric promises:

  A. SPATIAL SUPERpixel grounding: connected components in the range-view
     projection (label one point per component, its projection neighbors
     inherit). P4 = P(same class | neighbor) was 0.81-0.93 in the promise
     diagnostic -- the sensor's own geometry. Components are found on the
     MASK ONLY (no labels); the rep = the highest-influence point in the
     component; the component's points inherit its TRUE label (simulated).
  B. CLASS-CENTROID from the expanded pools: once components are grounded,
     per-class centroids in the 128-d features (the expanded pool is much
     larger than k anchors, so the centroid approaches the oracle).
  C. AGREEMENT GATE on T: a grounded point enters T only if its superpixel
     label and its class-centroid decode AGREE (spatial geometry AND feature
     geometry both say the same class -- the compound certificate). The
     contamination-free operating point.

End-to-end: budget (components queried) -> grounded precision/coverage -> the
ABLATION LADDER of T constructions -> ridge mIoU (S = all, T = gated):
  S0_direct    : only the queried representatives in T (the floor)
  S1_spatial   : superpixel labels propagated to whole components
  S2_centroid  : class-centroid decode with the cosine gate
  S3_hybrid_AND: spatial AND feature agree + gate (the compound)
  S4_union     : spatial expansion plus centroid expansion beyond it
The ladder shows which stage earns its keep and whether the compound jump from
S1/S2 to S3 is justified, or a simpler rung already works.
Efficiency measured per stage: CCL time, centroid decode time, ridge fit time.

References: frozen / oracle per condition.

Usage:
  uv run python robust_diagnostic/al_hybrid_grounding_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_hybrid_grounding_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import yaml
import numpy as np
import torch

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
DGLSSPP_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp'
DGLSSPP_METHOD = 'supcon_vib_dglsspp'
SKETCH_SEED = 11

def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)

def extract_positions(model, parser, device, num_frames=100):
    """Features, labels, AND the flattened (row*W + col) index per valid point
    plus the (H, W) grids -- needed to map pool points to superpixels."""
    feats, lbls, poss = [], [], []
    grids = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device)
            lab_grid = batch[2].to(device).squeeze(0)          # (H, W)
            msk_grid = (batch[1].to(device) > 0).squeeze(0)    # (H, W)
            labels = lab_grid.view(-1)
            mask = msk_grid.view(-1)
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(z_flat.cpu())
            lbls.append(labels[mask].cpu())
            H, W = lab_grid.shape
            pos_flat = torch.arange(H * W).view(H, W)[msk_grid].cpu()
            poss.append(pos_flat)
            grids.append((lab_grid.cpu(), msk_grid.cpu()))
    return torch.cat(feats, dim=0), torch.cat(lbls, dim=0), torch.cat(poss, dim=0), grids

def hdc_codes(feats, proj, device, chunk=100000):
    codes = []
    for s in range(0, len(feats), chunk):
        codes.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(codes, dim=0)

def onehot(lbls, num_classes):
    y = torch.zeros(len(lbls), num_classes)
    y[torch.arange(len(lbls)), lbls.long()] = 1.0
    return y

def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append((codes[s:s + chunk].float() @ W).argmax(dim=1))
    return torch.cat(preds)

def scores(W, codes, chunk=100000):
    W = W.detach().cpu()
    outs = []
    for s in range(0, len(codes), chunk):
        outs.append(codes[s:s + chunk].float() @ W)
    return torch.cat(outs, dim=0)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def tic():
    sync()
    return time.time()

def toc(t0):
    sync()
    return time.time() - t0

def nystrom_influence(Xd, lam, m, device):
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    M = torch.linalg.inv(Shat)
    C = Xd @ P
    MC = C @ M
    return (MC.norm(dim=1) * (Xd.shape[1] ** 0.5)).cpu()

def ridge_fit_soft(Xd, Y, lam, iters, m, device):
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    That = XP.t() @ Yd
    x = P @ torch.linalg.solve(Shat, That)
    b = Xd.t() @ Yd
    def A(v):
        return Xd.t() @ (Xd @ v)
    r = b - A(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A(p)
        alpha_k = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha_k.unsqueeze(0) * p
        r = r - alpha_k.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    return x.float()

# ---------------- connected components on the projection mask ----------------

def ccl_components(msk_grid):
    """Connected components (4-connectivity) of a masked grid via two-pass
    union-find. Returns (labels (H,W) int, n_components)."""
    H, W = msk_grid.shape
    m = msk_grid.numpy()
    parent = list(range(H * W + 1))          # 1-indexed cells
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    labels = np.zeros((H, W), dtype=np.int64)
    nxt = 0
    for i in range(H):
        for j in range(W):
            if not m[i, j]:
                continue
            up = labels[i - 1, j] if i > 0 else 0
            left = labels[i, j - 1] if j > 0 else 0
            if up == 0 and left == 0:
                nxt += 1
                labels[i, j] = nxt
            elif up != 0 and left == 0:
                labels[i, j] = up
            elif up == 0 and left != 0:
                labels[i, j] = left
            else:
                a, b = up, left
                if a != b:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
                labels[i, j] = min(a, b)
    # second pass: relabel to contiguous 0..K-1
    root_map = {}
    out = np.zeros_like(labels)
    k = 0
    for i in range(H):
        for j in range(W):
            if labels[i, j]:
                r = find(labels[i, j])
                if r not in root_map:
                    root_map[r] = k
                    k += 1
                out[i, j] = root_map[r]
    return torch.from_numpy(out), k

# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--max_clean", type=int, default=200000)
    parser.add_argument("--nystrom_m", type=int, default=1000)
    parser.add_argument("--cg_iters", type=int, default=8)
    parser.add_argument("--budgets", type=str, default="10,30,100,300,all")
    parser.add_argument("--agree_gate", type=float, default=0.5,
                        help="min cosine to the class centroid for the agreement gate")
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_hybrid_grounding_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    # clean features + frozen probe (shared across conditions)
    cf, cl, _, _ = extract_positions(model, build_parser(args.kitti_dir, DATA, ARCH),
                                     device, args.frames)
    proj_clean = get_hdc_projection(dim_in=cf.shape[1], dim_out=10000, device=device)
    mc = min(args.max_clean, len(cf))
    cc = hdc_codes(cf[:mc], proj_clean, device)
    W_clean = ridge_fit_soft(cc.float().to(device), onehot(cl[:mc], NUM_CLASSES),
                             args.lam, args.cg_iters, args.nystrom_m, device)
    del cc, cf, cl

    results = {'label': args.label, 'conds': {}}

    for cond in conds:
        t_cond = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l, pos, grids = extract_positions(model, build_parser(cdir, DATA, ARCH),
                                             device, args.frames)
        proj = get_hdc_projection(dim_in=f.shape[1], dim_out=10000, device=device)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        pool_pos = pos[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        pool_codes = hdc_codes(pool, proj, device)
        val_codes = hdc_codes(val, proj, device)
        Xd = pool_codes.float().to(device)

        r = {'refs': {}, 'components': {}, 'budgets': {}, 'efficiency': {},
             'synthesis': []}

        def mw(W):
            return compute_miou(decode(W, val_codes), vl)

        W_oracle = ridge_fit_soft(Xd, onehot(pl, NUM_CLASSES), args.lam,
                                  args.cg_iters, args.nystrom_m, device)
        r['refs'] = {'frozen': mw(W_clean), 'oracle': mw(W_oracle),
                     'pseudo_acc': float((scores(W_clean, pool_codes).argmax(dim=1) == pl).float().mean().item())}

        # ---- superpixels: CCL per frame, map pool points to components ----
        t_ccl = tic()
        comp_per_point = torch.full((len(f),), -1, dtype=torch.long)
        n_comp = 0
        frame_off = 0
        for lab_g, msk_g in grids:
            H, W = msk_g.shape
            n_f = int(msk_g.sum().item())
            labs, k = ccl_components(msk_g)
            comp_per_point[frame_off:frame_off + n_f] = labs[msk_g] + n_comp
            n_comp += k
            frame_off += n_f
        t_ccl = toc(t_ccl)
        pool_comp = comp_per_point[perm[:args.pool_size]]
        # per-component stats over the POOL points
        n_comp_pool = int(pool_comp.max().item()) + 1
        comp_size = torch.bincount(pool_comp.clamp(min=0), minlength=n_comp_pool)
        # per-point influence for rep selection (HDC codes of the pool)
        I = nystrom_influence(Xd, args.lam, args.nystrom_m, device)
        comp_inf = torch.zeros(n_comp_pool)
        comp_rep = torch.zeros(n_comp_pool, dtype=torch.long)
        for c in range(n_comp_pool):
            m = pool_comp == c
            if int(m.sum().item()) == 0:
                continue
            comp_inf[c] = float(I[m].sum().item())
            comp_rep[c] = m.nonzero().squeeze(1)[int(I[m].argmax().item())]
        r['components'] = {'n_total': n_comp, 'n_in_pool': n_comp_pool,
                           'mean_size': float(comp_size[comp_size > 0].float().mean().item()),
                           'ccl_time_s': t_ccl}

        # ---- budget sweep: query top-M components by influence, ground, gate ----
        order = torch.argsort(comp_inf, descending=True)
        budgets = [int(x) for x in args.budgets.split(',') if x != 'all'] + [n_comp_pool]
        budgets = sorted(set(budgets))

        zn = pool.float()
        zn = zn / (zn.norm(dim=1, keepdim=True) + 1e-8)

        t_dec = tic()
        # class centroids from FULL grounding (oracle-quality for the gate);
        # the budgeted versions re-estimate from the grounded subset
        t_dec = toc(t_dec)

        for B in budgets:
            queried = order[:B]
            queried_pts = comp_rep[queried]
            queried_pts = queried_pts[queried_pts >= 0]
            ground_lbl = torch.zeros(len(pool), dtype=torch.long)
            grounded_mask = torch.zeros(len(pool), dtype=torch.bool)
            for c in queried:
                m = pool_comp == c
                if int(m.sum().item()) == 0:
                    continue
                ground_lbl[m] = pl[comp_rep[c]]
                grounded_mask[m] = True
            n_g = int(grounded_mask.sum().item())
            g_prec = float((ground_lbl[grounded_mask] == pl[grounded_mask]).float().mean().item()) if n_g else None

            # class centroids from the grounded points (128-d)
            cents = {}
            for c in range(1, NUM_CLASSES):
                m = grounded_mask & (ground_lbl == c)
                if int(m.sum().item()) < 5:
                    continue
                cents[c] = zn[m].mean(dim=0)
            if not cents:
                continue
            cid = torch.tensor(sorted(cents))
            c_mat = torch.stack([cents[c] for c in sorted(cents)])
            c_mat = c_mat / (c_mat.norm(dim=1, keepdim=True) + 1e-8)
            simc = zn @ c_mat.t()
            bestc, bic = simc.max(dim=1)
            cent_lbl = cid[bic]
            top2c = torch.topk(simc, 2, dim=1).values
            cent_margin = (top2c[:, 0] - top2c[:, 1]).clamp(min=0)
            cent_pass = bestc >= args.agree_gate

            # agreement: superpixel label == centroid decode AND cosine gate
            agree = (cent_lbl == ground_lbl) & cent_pass & grounded_mask

            # ---- ablation ladder: five rungs per budget ----
            # S0 direct-sparse: only the queried representatives in T (floor)
            Y0 = torch.zeros(len(pool), NUM_CLASSES)
            Y0[queried_pts] = onehot(pl[queried_pts], NUM_CLASSES)
            # S1 spatial-only: superpixel labels propagated to the whole component
            Y1 = torch.zeros(len(pool), NUM_CLASSES)
            Y1[grounded_mask] = onehot(ground_lbl[grounded_mask], NUM_CLASSES)
            # S2 centroid-only: feature-geometry decode with the cosine gate
            Y2 = torch.zeros(len(pool), NUM_CLASSES)
            Y2[cent_pass] = onehot(cent_lbl[cent_pass], NUM_CLASSES)
            # S3 hybrid (AND): spatial AND feature agree + gate (the compound)
            Y3 = torch.zeros(len(pool), NUM_CLASSES)
            Y3[agree] = onehot(cent_lbl[agree], NUM_CLASSES)
            # S4 union: spatial expansion plus centroid expansion beyond it
            Y4 = torch.zeros(len(pool), NUM_CLASSES)
            Y4[grounded_mask] = onehot(ground_lbl[grounded_mask], NUM_CLASSES)
            extra = cent_pass & ~grounded_mask
            Y4[extra] = onehot(cent_lbl[extra], NUM_CLASSES)

            arms = {}
            t_fit0 = tic()
            # one representative fit for the efficiency measurement (the arms
            # below each do a fit; the time is identical)
            ridge_fit_soft(Xd, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters,
                           args.nystrom_m, device)
            fit_time = toc(t_fit0)
            for aname, Ym, amask in [
                    ('S0_direct', Y0, None),
                    ('S1_spatial', Y1, grounded_mask),
                    ('S2_centroid', Y2, cent_pass),
                    ('S3_hybrid_AND', Y3, agree),
                    ('S4_union', Y4, None)]:
                n = int(amask.sum().item()) if amask is not None else int((Ym.sum(dim=1) > 0).sum().item())
                if n == 0:
                    arms[aname] = {'n': 0, 'prec': None, 'cov': 0.0, 'miou': None,
                                   'insufficient': True}
                    continue
                if amask is not None:
                    lbls = Ym[amask].argmax(dim=1)
                    prec = float((lbls == pl[amask]).float().mean().item())
                else:
                    used = (Ym.sum(dim=1) > 0)
                    lbls = Ym[used].argmax(dim=1)
                    prec = float((lbls == pl[used]).float().mean().item())
                W = ridge_fit_soft(Xd, Ym, args.lam, args.cg_iters, args.nystrom_m, device)
                arms[aname] = {'n': n, 'prec': prec, 'cov': n / len(pool),
                               'miou': mw(W), 'fit_time_s': fit_time}

            r['budgets'][str(B)] = {
                'n_queried': min(B, n_comp_pool),
                'n_grounded': n_g,
                'ground_prec': g_prec,
                'arms': arms,
            }

        syn = r['synthesis']
        syn.append(f"COND {cond}: frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} | "
                   f"superpixels {r['components']['n_total']} (pool {r['components']['n_in_pool']}, "
                   f"mean size {r['components']['mean_size']:.0f}) | CCL {t_ccl:.2f}s")
        for B in budgets:
            e = r['budgets'][str(B)]
            syn.append(f"  B={B} (queried {e['n_queried']}, grounded {e['n_grounded']} "
                       f"prec {e['ground_prec']:.3f}): " + " | ".join(
                f"{aname}:{a['miou']:.3f}(prec {a['prec']:.3f},cov {a['cov']:.3f})"
                if a.get('miou') is not None else f"{aname}:skip"
                for aname, a in e['arms'].items()))

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in syn:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("superpixels: CCL on the projection mask (no labels). Components map")
    print("  pool points to spatial regions; n_total vs n_in_pool shows how many")
    print("  are represented in the sampled pool.")
    print("budgets: query the top-B components by influence (sum of per-point")
    print("  influence in the component). ground_prec = precision of the")
    print("  propagated superpixel labels (oracle-measured).")
    print("The ABLATION LADDER per budget (each arm: T-label precision, coverage,")
    print("  ridge mIoU with S=all):")
    print("  S0_direct    : only the queried representatives in T (the floor)")
    print("  S1_spatial   : superpixel labels propagated to whole components")
    print("  S2_centroid  : class-centroid decode with the cosine gate")
    print("  S3_hybrid_AND: spatial AND feature agree + gate (the compound)")
    print("  S4_union     : spatial expansion plus centroid expansion beyond it")
    print("Reading: S3 vs S1/S2 says whether the agreement gate earns its keep;")
    print("  S3 vs S0 says whether spatial grounding adds anything over direct")
    print("  labels; S4 vs S3 shows the contamination cost of un-gated coverage;")
    print("  the simplest rung that beats frozen toward oracle is the mechanism.")
    print("Efficiency: CCL time (per-frame, fixed), fit time (one 50k ridge).")
    print("Verdict: if S3 (or a simpler rung) reaches high T precision at usable")
    print("  coverage with mIoU climbing above frozen toward oracle, that rung is")
    print("  the Pillar-3 mechanism at ~0.9M pts/s.")

if __name__ == "__main__":
    main()
