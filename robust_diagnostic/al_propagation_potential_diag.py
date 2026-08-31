"""al_propagation_potential_diag.py: can AL labels + feature-space structure
approximate the oracle means M*? A POTENTIAL-vs-IMPLEMENTATION diagnostic.

The class-statistics line closed on the finding that the MEAN decoder
(W_mean_oracle = Sigma0^-1 P0 M*) is decision-relevant (+0.50 to +1.15 gc) but
needs the oracle means, which need full pool labels. The open question the user
poses: is there a way to approximate M* from a few AL labels using the structure
of the feature space?

The prior docs (active_iterations.md) establish that the FUNDAMENTAL signals are
real but every IMPLEMENTATION failed:
  - P(same class | feature neighbor) = 0.81-0.93 (128-d) / 1-NN purity 0.45-0.88
    (the signal exists)
  - random 8-32 samples/class estimate means to cos 0.95-0.99 (means estimable)
  - BUT cluster+propagation through 65%-pure clusters poisons T below frozen
    (the implementation failed); diffusion needs 50% anchors; T-synthesis is
    killed by the whitening amplifying residual T error (Iterations 1/2/7/8).
  - The "boundary is pathologically sensitive": means move cos 0.92-0.99 but
    the probe rotates ~90 degrees (Iteration 6).

This diagnostic SEPARATES the two, per direction: the ORACLE signal (what the
structure could support if perfectly estimated) vs the IMPLEMENTATION result
(what a few-label method actually achieves). The point of the separation: if the
oracle signal is high but the implementation is low, the direction has potential
that a better estimator could realize; if the oracle signal is itself low, the
direction is closed regardless of implementation.

PART A -- PROXIMITY as a same-labelness gauge (the fundamental signal).
  A1  P(same class | nearest feature neighbor) at k = {1, 5, 10, 20} in the
      128-d feature space (NOT the saturated HDC code space) on the corrupted
      pool -- the raw label-transfer precision.
  A2  NEAREST-ANCHOR PROPAGATION accuracy/coverage curve: b anchors/class
      (random, oracle-labeled), propagate to ALL pool points via nearest anchor
      in 128-d; report propagation precision and coverage (fraction of pool
      assigned at precision >= tau). This is "can proximity transfer labels?"
  A3  per-class propagation precision -- which classes are transfer-friendly vs
      the known-loose {7, 15, 14}.

PART B -- the PROPAGATED MEAN quality (the class-statistics connection).
  B1  b anchors/class -> propagate -> propagated class means M_prop; report the
      mean error vs M* (raw cos AND whitened error, the Iteration-8 killer)
      AND the resulting mean-decoder gc. Compare to the three references:
      raw 8-point mean (known fail), frozen-pseudo mean (+0.05-0.12),
      W_mean_oracle ceiling (+0.5-1.15).
  B2  The whitened error ||Sigma^-1 (M_prop - M*)||/||Sigma^-1 M*|| -- is the
      propagated mean's residual within what the whitening tolerates?

PART C -- BOUNDARY finding (the "boundaries determine labels of large segments"
signal).
  C1  How much of the pool is near a boundary (low |top-2 margin|), and the
      precision of "label = the frozen probe's predicted side" restricted to
      those boundary points vs the bulk. Is there a boundary in feature space
      that separates classes better than the frozen probe's margin does?
  C2  P(true class = frozen argmax | margin bin): the margin-vs-label structure
      -- where does the frozen probe's boundary actually separate correctly?
      This is the "can boundaries label large segments" fundamental signal.

PART D -- ANCHOR DENSITY / MASS (the Iteration-7/8 hidden failure).
  D1  Mean-estimation curve: b anchors/class, mean cos to M* -- for the anchors
      THEMSELVES vs the PROPAGATED pool. Does propagation over the whole pool
      beat the anchors alone (i.e., does structure multiply the effective
      labeled set)?
  D2  MASS correction: with propagated labels, are the class COUNTS right?
      (Iteration 8's hidden failure: frozen counts were 8-611x wrong.) Report
      the count relative error for propagated labels vs oracle counts.

The verdict rule: for each direction, oracle-signal-high AND implementation-low
=> potential exists, the estimator is the bottleneck (pursue a better estimator).
Oracle-signal-low => the direction is closed regardless.

Usage:
  uv run python robust_diagnostic/al_propagation_potential_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_propagation_potential_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                  labels=data["labels"], color_map=data["color_map"], learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"], sensor=arch["dataset"]["sensor"],
                  max_points=arch["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract_clean(model, parser, device, num_frames=100):
    feats, lbls = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device); labels = batch[2].to(device).view(-1); mask = (batch[1].to(device) > 0).view(-1)
            out = model(in_vol); z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(zf.cpu()); lbls.append(labels[mask].cpu())
    return torch.cat(feats), torch.cat(lbls)


def hdc_codes(feats, proj, device, chunk=100000):
    out = []
    for s in range(0, len(feats), chunk):
        out.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(out)


def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc); y[torch.arange(len(lbls)), lbls.long()] = 1; return y


def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    p = []
    for s in range(0, len(codes), chunk):
        p.append((codes[s:s+chunk].float() @ W).argmax(1))
    return torch.cat(p)


def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)


def ridge_fit_soft(X, Y, lam, iters, m, device):
    X = X.to(device); torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1], max(1, X.shape[0] - 1))
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P; Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device); That = XP.t() @ Yd
    x0 = P @ torch.linalg.solve(Shat, That)
    if X.shape[0] <= 8:
        return x0.float()
    x = x0; b = X.t() @ Yd
    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p); d = (p * Ap).sum(0)
        if not torch.isfinite(d).all() or d.abs().max().item() < 1e-20:
            break
        a = rs / (d + 1e-30); x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap; rsn = (r * r).sum(0); be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p; rs = rsn
        if not torch.isfinite(x).all():
            return x0.float()
    return x.float()


def solve_whitened(X, B, lam, iters, m, device):
    X = X.to(device); B = B.float().to(device)
    torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1], max(1, X.shape[0] - 1))
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    w0 = P @ torch.linalg.solve(Shat, P.t() @ B)
    if B.shape[0] <= 8:
        return w0.float()
    x = w0; b = B
    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p); d = (p * Ap).sum(0)
        if not torch.isfinite(d).all() or d.abs().max().item() < 1e-20:
            break
        a = rs / (d + 1e-30); x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap; rsn = (r * r).sum(0); be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p; rs = rsn
        if not torch.isfinite(x).all():
            return w0.float()
    return x.float()


def class_means(X, y, nc):
    M = torch.zeros(nc, X.shape[1]); C = torch.zeros(nc)
    for c in range(nc):
        m = (y == c)
        if int(m.sum().item()) > 0:
            M[c] = X[m].mean(dim=0)
            C[c] = float(int(m.sum().item()))
    return M, C


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic():
    sync(); return time.time()


def toc(t0):
    sync(); return time.time() - t0


def cos_(a, b):
    return float((a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--pool_size", type=int, default=20000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--max_clean", type=int, default=30000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--b_anchors", type=str, default="1,2,4,8,16")
    ap.add_argument("--k_knn", type=str, default="1,5,10,20")
    ap.add_argument("--tau_gate", type=float, default=0.0, help="distance-gate fraction (0 = no gate)")
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    b_anchors = [int(x) for x in args.b_anchors.split(',')]
    k_sweep = [int(x) for x in args.k_knn.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_anchors': b_anchors,
               'k_knn': k_sweep, 'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    lac = la[ci]
    W0 = ridge_fit_soft(Xc, onehot(lac, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
    M0, C0 = class_means(Xc, lac, NUM_CLASSES)
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        fd, ld = extract_clean(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42); perm = torch.randperm(len(fd))
        pool_f, pl = fd[perm[:args.pool_size]], ld[perm[:args.pool_size]]
        val_f, vl = fd[perm[-args.val_size:]], ld[perm[-args.val_size:]]
        Xp = hdc_codes(pool_f, proj, device).float()
        Xv = hdc_codes(val_f, proj, device).float()
        del val_f, fd, ld
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']
        def gc(mi):
            return (mi - refs['frozen']) / gap if gap > 1e-9 else None

        M_star, C_star = class_means(Xp, pl, NUM_CLASSES)

        # 128-d feature space (the informative space, not the saturated code space)
        pf = F.normalize(pool_f.float(), p=2, dim=1)
        n = len(pf)

        # references for the mean decoder
        W_mean_oracle = solve_whitened(Xp, (M_star * C0.unsqueeze(1)).t().contiguous(),
                                       args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        gc_mean_oracle = gc(mw(W_mean_oracle, Xv, vl))

        # ---- A1. PROXIMITY same-labelness (128-d) ----
        a1 = {}
        chunk = 5000
        for k in k_sweep:
            same = []
            for s in range(0, n, chunk):
                sim = pf[s:s+chunk] @ pf.t()
                topk = torch.topk(sim, k + 1, dim=1)     # +1 to exclude self
                nbr = topk.indices[:, 1:]
                same.append((pl[nbr] == pl[s:s+chunk].unsqueeze(1)).float().mean().item())
            a1[str(k)] = sum(same) / len(same)

        # ---- A2. NEAREST-ANCHOR PROPAGATION (128-d) ----
        a2 = {}
        for b in b_anchors:
            torch.manual_seed(7)
            anchors = []
            for c in range(1, NUM_CLASSES):
                idx = torch.nonzero(pl == c).squeeze(1)
                if len(idx) == 0:
                    continue
                anchors.append(idx[torch.randperm(len(idx))[:min(b, len(idx))]])
            if not anchors:
                continue
            anc = torch.cat(anchors)
            anc_f = pf[anc]
            anc_lab = pl[anc]
            # nearest anchor for every pool point
            sim = pf @ anc_f.t()
            nn = sim.argmax(1)
            prop_lab = anc_lab[nn]
            prec = float((prop_lab == pl).float().mean().item())
            # distance-gated: coverage at precision
            nn_sim = sim.gather(1, nn.unsqueeze(1)).squeeze(1)
            if args.tau_gate > 0:
                q = torch.quantile(nn_sim, 1 - args.tau_gate)
                gated = nn_sim >= q
                prec_g = float((prop_lab[gated] == pl[gated]).float().mean().item()) if int(gated.sum().item()) > 0 else None
                cov_g = float(gated.float().mean().item())
            else:
                prec_g = None; cov_g = None
            a2[str(b)] = {'prec': prec, 'prec_gated': prec_g, 'cov_gated': cov_g,
                          'n_anchors': int(len(anc))}

        # ---- A2b. CENTROID-anchor propagation (the prior best feature promise:
        #      active_iterations Iteration 4 -- B_centroid reached 0.82 precision
        #      with k>=2-4 anchors/class, nearest-anchor alone was weak). ----
        a2b = {}
        for b in b_anchors:
            if b < 2:
                continue
            torch.manual_seed(7)
            means = torch.zeros(NUM_CLASSES, pf.shape[1])
            cnt = torch.zeros(NUM_CLASSES)
            for c in range(1, NUM_CLASSES):
                idx = torch.nonzero(pl == c).squeeze(1)
                if len(idx) == 0:
                    continue
                sub = idx[torch.randperm(len(idx))[:min(b, len(idx))]]
                means[c] = pf[sub].mean(dim=0)
                cnt[c] = float(len(sub))
            cf = F.normalize(means, p=2, dim=1)
            sim = pf @ cf.t()
            nn = sim.argmax(1)
            prop_lab = torch.full((n,), -1)
            for c in range(1, NUM_CLASSES):
                if cnt[c] == 0:
                    continue
                prop_lab[nn == c] = c
            valid = prop_lab >= 0
            prec = float((prop_lab[valid] == pl[valid]).float().mean().item()) if int(valid.sum().item()) > 0 else None
            cov = float(valid.float().mean().item())
            a2b[str(b)] = {'prec': prec, 'cov': cov}

        # ---- A2c. AGREEMENT-gated propagation (Iteration-5 clean-T path: only
        #      propagate where the frozen probe agrees with the anchor's label). ----
        a2c = {}
        Lp_full = Xp.float() @ W0c
        pred_full = Lp_full.argmax(1)
        for b in b_anchors:
            torch.manual_seed(7)
            anc = []
            for c in range(1, NUM_CLASSES):
                idx = torch.nonzero(pl == c).squeeze(1)
                if len(idx) == 0:
                    continue
                anc.append(idx[torch.randperm(len(idx))[:min(b, len(idx))]])
            anc = torch.cat(anc)
            anc_f = pf[anc]; anc_lab = pl[anc]
            nn = (pf @ anc_f.t()).argmax(1)
            prop_lab = anc_lab[nn]
            agree = pred_full == prop_lab
            prec = float((prop_lab[agree] == pl[agree]).float().mean().item()) if int(agree.sum().item()) > 0 else None
            cov = float(agree.float().mean().item())
            a2c[str(b)] = {'prec_gated': prec, 'cov_gated': cov}

        # ---- A4. CLEAN-SOURCE anchor propagation (the MEMORY-BANK idea: use the
        #      free CLEAN training labels as the anchor bank, label the corrupted
        #      pool by nearest clean feature). This is label-free for AL purposes --
        #      the clean labels already exist at training time. ----
        a4 = {}
        for sub_n in [5000, 20000, 50000]:
            torch.manual_seed(11)
            sel = torch.randperm(len(fa))[:sub_n]
            clean_f = F.normalize(fa[sel].float(), p=2, dim=1)
            clean_l = la[sel]
            # nearest clean feature to each pool point (chunked)
            nn_c = []
            for s in range(0, n, 5000):
                sim = pf[s:s+5000] @ clean_f.t()
                nn_c.append(sim.argmax(1))
            nn_c = torch.cat(nn_c)
            prop_lab_c = clean_l[nn_c]
            prec_c = float((prop_lab_c == pl).float().mean().item())
            a4[str(sub_n)] = {'prec': prec_c}

        # ---- A3. per-class propagation precision (b=4) ----
        a3 = {}
        b4 = 4 if 4 in b_anchors else b_anchors[0]
        torch.manual_seed(7)
        anc = []
        for c in range(1, NUM_CLASSES):
            idx = torch.nonzero(pl == c).squeeze(1)
            if len(idx) == 0:
                continue
            anc.append(idx[torch.randperm(len(idx))[:min(b4, len(idx))]])
        anc = torch.cat(anc)
        anc_f = pf[anc]; anc_lab = pl[anc]
        nn = (pf @ anc_f.t()).argmax(1)
        prop_lab = anc_lab[nn]
        for c in range(1, NUM_CLASSES):
            m = (pl == c)
            if int(m.sum().item()) < 50:
                continue
            a3[str(c)] = float((prop_lab[m] == pl[m]).float().mean().item())

        # ---- B1. PROPAGATED MEAN quality (code space) + decoder gc ----
        b1 = {}
        for b in b_anchors:
            torch.manual_seed(7)
            anc = []
            for c in range(1, NUM_CLASSES):
                idx = torch.nonzero(pl == c).squeeze(1)
                if len(idx) == 0:
                    continue
                anc.append(idx[torch.randperm(len(idx))[:min(b, len(idx))]])
            anc = torch.cat(anc)
            anc_f = pf[anc]; anc_lab = pl[anc]
            nn = (pf @ anc_f.t()).argmax(1)
            prop_lab = anc_lab[nn]
            M_prop, C_prop = class_means(Xp, prop_lab, NUM_CLASSES)
            # raw mean cos per class (using true counts for the decoder is a leak;
            # use propagated counts for the decoder B)
            cos_m = []
            for c in range(1, NUM_CLASSES):
                if C_star[c] < 10:
                    continue
                cos_m.append(cos_(M_prop[c], M_star[c]))
            mean_cos = sum(cos_m) / len(cos_m) if cos_m else None
            # whitened error (the Iteration-8 killer)
            B_prop = (M_prop * C0.unsqueeze(1)).t().contiguous()
            W_prop = solve_whitened(Xp, B_prop, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
            W_est_err = float((W_prop - W_mean_oracle).norm().item() /
                              (W_mean_oracle.norm().item() + 1e-12))
            # decoder gc with PROPAGATED counts (honest: no count leak)
            B_prop_c = (M_prop * C_prop.unsqueeze(1)).t().contiguous()
            W_prop_c = solve_whitened(Xp, B_prop_c, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
            gc_prop = gc(mw(W_prop_c, Xv, vl))
            # decoder gc with ORACLE counts (ceiling for propagated means)
            W_prop_oc = solve_whitened(Xp, B_prop, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
            gc_prop_oc = gc(mw(W_prop_oc, Xv, vl))
            # mass correction: count error (Iteration-8 hidden failure)
            count_err = float((C_prop[1:] - C_star[1:]).abs().sum().item() /
                              (C_star[1:].sum().item() + 1e-12))
            # MASS-CALIBRATED propagated mean (the Iteration-8 count-fix lever):
            # rescale the propagated counts by the ANCHOR-observed class
            # proportions, so the decoder uses mass-corrected counts without
            # leaking oracle counts.
            anc_prop = torch.zeros(NUM_CLASSES)
            for c in range(1, NUM_CLASSES):
                anc_prop[c] = float((anc_lab == c).float().mean().item())
            if anc_prop.sum().item() > 1e-9:
                anc_prop = anc_prop / anc_prop.sum().item()
                mass_cal = C_prop.clone()
                # use the anchor-observed proportions scaled to the pool size
                mass_cal[1:] = (anc_prop[1:] * (C_prop.sum().item())) + 1e-6
                B_cal = (M_prop * mass_cal.unsqueeze(1)).t().contiguous()
                W_cal = solve_whitened(Xp, B_cal, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
                gc_cal = gc(mw(W_cal, Xv, vl))
                count_err_cal = float((mass_cal[1:] - C_star[1:]).abs().sum().item() /
                                      (C_star[1:].sum().item() + 1e-12))
            else:
                gc_cal = None; count_err_cal = None
            b1[str(b)] = {'mean_cos': mean_cos, 'W_err_vs_meanoracle': W_est_err,
                          'gc_prop_counts': gc_prop, 'gc_oracle_counts': gc_prop_oc,
                          'gc_mass_cal': gc_cal,
                          'count_err': count_err, 'count_err_cal': count_err_cal}

        # ---- C1/C2. BOUNDARY finding ----
        Lp = Xp.float() @ W0c
        top2 = torch.topk(Lp, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
        pred = Lp.argmax(1)
        # precision of "label = frozen argmax" by margin quantile
        qs = torch.quantile(margin, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        c1 = {}
        for i, q in enumerate(qs):
            m = margin <= q
            c1[f'q{i}'] = {'margin_thresh': float(q.item()),
                           'frac': float(m.float().mean().item()),
                           'prec': float((pred[m] == pl[m]).float().mean().item()) if int(m.sum().item()) > 0 else None}
        # C2: P(true = frozen argmax | margin bin)
        bins = torch.quantile(margin, torch.linspace(0, 1, 6))
        c2 = {}
        for i in range(len(bins) - 1):
            m = (margin >= bins[i]) & (margin <= bins[i+1])
            c2[f'b{i}'] = float((pred[m] == pl[m]).float().mean().item()) if int(m.sum().item()) > 0 else None

        # ---- D1. anchors vs propagated mean (does structure multiply labels?) ----
        d1 = {}
        for b in b_anchors:
            torch.manual_seed(7)
            anc_means = M0.clone(); anc_cnt = torch.zeros(NUM_CLASSES)
            for c in range(1, NUM_CLASSES):
                idx = torch.nonzero(pl == c).squeeze(1)
                if len(idx) == 0:
                    continue
                sub = idx[torch.randperm(len(idx))[:min(b, len(idx))]]
                anc_means[c] = Xp[sub].float().mean(dim=0)
                anc_cnt[c] = float(len(sub))
            # anchor-only mean cos
            cos_a = []
            for c in range(1, NUM_CLASSES):
                if anc_cnt[c] == 0 or C_star[c] < 10:
                    continue
                cos_a.append(cos_(anc_means[c], M_star[c]))
            # propagated mean cos (from b1)
            d1[str(b)] = {'anchor_mean_cos': sum(cos_a) / len(cos_a) if cos_a else None,
                          'prop_mean_cos': b1[str(b)]['mean_cos'] if str(b) in b1 else None}

        cond_res = {'refs': refs, 'gap': float(gap),
                    'gc_mean_oracle': gc_mean_oracle,
                    'A1_proximity': a1, 'A2_propagation': a2,
                    'A2b_centroid': a2b, 'A2c_agreement': a2c,
                    'A3_perclass': a3, 'A4_cleansource': a4,
                    'B1_prop_mean': b1, 'C1_boundary': c1, 'C2_margin_bins': c2,
                    'D1_anchor_vs_prop': d1}
        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, M_star, pool_f, pf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    W_mean_oracle gc {gc_mean_oracle:+.2f}")
        print("    A1 proximity(128d): " + " ".join(f"k{k}:{v:.3f}" for k, v in a1.items()))
        print("    A2 propagation: " + " ".join(
            f"b{b}:prec{v['prec']:.3f}(g{v['prec_gated'] if v['prec_gated'] is not None else float('nan'):.3f}@cov{v['cov_gated'] if v['cov_gated'] is not None else float('nan'):.3f})"
            for b, v in a2.items()))
        print("    A2b centroid: " + " ".join(
            f"b{b}:prec{v['prec'] if v['prec'] is not None else float('nan'):.3f}cov{v['cov']:.3f}" for b, v in a2b.items()))
        print("    A2c agreement: " + " ".join(
            f"b{b}:prec{v['prec_gated'] if v['prec_gated'] is not None else float('nan'):.3f}cov{v['cov_gated']:.3f}" for b, v in a2c.items()))
        print("    A4 clean-source: " + " ".join(f"n{k}:prec{v['prec']:.3f}" for k, v in a4.items()))
        print("    B1 prop-mean: " + " ".join(
            f"b{b}:cos{v['mean_cos']:.2f}Werr{v['W_err_vs_meanoracle']:.1f}"
            f"gcP{v['gc_prop_counts']:+.2f}gcO{v['gc_oracle_counts']:+.2f}gcC{v['gc_mass_cal'] if v['gc_mass_cal'] is not None else float('nan'):+.2f}"
            f"cnt{v['count_err']:.1f}->{v['count_err_cal'] if v['count_err_cal'] is not None else float('nan'):.1f}"
            for b, v in b1.items()))
        print("    C1 boundary: " + " ".join(
            f"{k}:frac{v['frac']:.2f}prec{v['prec']:.3f}" for k, v in c1.items()))
        print("    D1 anchor-vs-prop mean cos: " + " ".join(
            f"b{b}:a{v['anchor_mean_cos']:.2f}p{v['prop_mean_cos']:.2f}" for b, v in d1.items()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("The point: separate the ORACLE signal from the IMPLEMENTATION.")
    print("A1 (proximity precision) and B1's mean_cos / gc_oracle_counts are the")
    print("ORACLE signal -- what the structure could support if estimated well.")
    print("A2/B1-gc_prop_counts are the IMPLEMENTATION -- what few labels +")
    print("nearest-anchor propagation actually achieve.")
    print("  A1 high, A2 low -> proximity signal is real; propagation estimator")
    print("     is the bottleneck (potential exists).")
    print("  B1 gc_oracle_counts ~ gc_mean_oracle -> propagated MEANS + oracle")
    print("     counts reach the ceiling; the COUNT error (Iteration-8 hidden")
    print("     failure) is the bottleneck, not the means.")
    print("  B1 gc_mass_cal vs gc_prop_counts -> does the anchor-proportion count")
    print("     correction (the Iteration-8 lever) fix the count error?")
    print("  A2b (centroid anchors) vs A2 (nearest anchors): the prior Iteration-4")
    print("     promise that k>=2 centroids beat single anchors.")
    print("  A2c (agreement-gated): the Iteration-5 clean-T path -- only propagate")
    print("     where the frozen probe agrees.")
    print("  A4 (clean-source bank, the MEMORY-BANK idea): label-free propagation")
    print("     from the existing clean training labels -- how much structure do we")
    print("     get WITHOUT spending any AL budget?")
    print("  C1/C2: where does the frozen boundary actually separate correctly?")
    print("  D1: does propagation over the pool beat the anchors alone (structure")
    print("     multiplies the effective labeled set)?")


if __name__ == "__main__":
    main()
