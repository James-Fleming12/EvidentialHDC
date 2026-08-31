"""al_class_stats_iter78_diag.py: Iterations 7.5 + 8 in ONE run.

PART A (Iteration 7.5): COVARIANCE-ONLY DECODER CEILING -- the "required before
closing" test from the Iteration-7 review. The Iteration-7 vector decomposition
computed cov_ab RESIDUALLY (cov = dw - mean - prior), so cos_cov ~ 1.0 is partly
tautological. The decisive test is whether the covariance term, built
independently, reproduces the oracle CLASSIFIER (gc and decision agreement).

  The exact additive decomposition (additive in W, since R = W* - W0):
    W0                          frozen
    mean-only    W0 + R_mean    R_mean = Sigma0^-1 P0 (M* - M0)  (W_mean_oracle)
    prior-only   W0 + R_prior   R_prior = Sigma0^-1 (P* - P0) M*
    cov-only     W0 + (W* - W_mean_oracle)   [W* - W_mean_oracle = R_cov + R_prior
                independent, from the ORACLE ridge W* minus the mean decoder]
    mean+prior   W_mean_oracle + R_prior
    mean+cov     W* - R_prior
    full oracle  W*

  Report gc AND dec_agree (fraction of val predictions matching the oracle,
  overall and restricted to the error points) for each decoder.
  DECISIVE: if cov-only gc ~ oracle gc and dec_agree ~ 1.0, the remaining
  problem IS covariance adaptation (closes the class-stats line decisively).
  If cov-only does NOT reproduce the classifier despite the vector accounting,
  there is a subtle issue worth investigating.

PART B (Iteration 8): PAIRWISE LOGIT CORRECTION, ceiling-first. The only tested
positive mechanism from Iteration 7 G (z'_a - z'_b = alpha(z_a-z_b) + beta).
Six ceiling/stability experiments BEFORE building any method:
  B1 per-pair oracle ceiling: fit (alpha,beta) per confused pair on all pool
     points of the pair; report gc + dec_agree. Which pairs help?
  B2 pool vs label estimation: fit (alpha,beta) with b in {1,2,4,8,16} labels
     per class of the pair; does few-label track the oracle?
  B3 random-pair vs confused-pair: same (alpha,beta) fit on RANDOM pairs of the
     same count -- is the gain confusion-concentrated (sparse = good for AL)?
  B4 boundary-conditioned fit: fit (alpha,beta) only on pool points with
     |z_a - z_b| < tau -- more statistically efficient near the boundary.
  B5 global per-class bias: z'_c = z_c + b_c (17 scalars) vs the O(K^2) pairwise
     form -- does the pairwise flexibility actually buy anything?
  B6 SHARED SCALAR (most important): ONE global (alpha,beta) applied to all
     pairs, z'_a - z'_b = alpha(z_a-z_b) + beta. If a tiny parameter count
     captures most of the oracle correction, it is far more compelling than
     17x16 independent pairs -- and the only version a few labels can estimate.

Usage:
  uv run python robust_diagnostic/al_class_stats_iter78_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_class_stats_iter78_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
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


def dec_agree(W, Xv, vl, W_ref, err_only=False):
    """Fraction of val predictions matching the reference decoder."""
    p = decode(W, Xv)
    pr = decode(W_ref, Xv)
    if err_only:
        m = (pr != vl)
        if int(m.sum().item()) == 0:
            return None
        return float((p[m] == pr[m]).float().mean().item())
    return float((p == pr).float().mean().item())


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


def cos_(a, b):
    return float((a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12))


def fit_pair_logit(z_a, z_b, y_in_pair, label_a_b):
    """Fit (alpha, beta): alpha*(z_a-z_b) + beta predicts the a-vs-b label.
    z_a, z_b: tensors. y_in_pair: 1 if a, 0 if b (or -1/+1 target)."""
    m = z_a - z_b
    A = torch.stack([m, torch.ones_like(m)], dim=1)
    sol = torch.linalg.lstsq(A.double(), label_a_b.double().unsqueeze(1)).solution
    return float(sol[0].item()), float(sol[1].item())


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic():
    sync(); return time.time()


def toc(t0):
    sync(); return time.time() - t0


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
    ap.add_argument("--n_pairs", type=int, default=5)
    ap.add_argument("--b_labels", type=str, default="1,2,4,8,16")
    ap.add_argument("--tau", type=float, default=1.0)
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
    b_labels = [int(x) for x in args.b_labels.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'n_pairs': args.n_pairs,
               'b_labels': b_labels, 'tau': args.tau, 'conds': {}}

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
        Ws_c = Ws.detach().cpu()
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']
        def gc(mi):
            return (mi - refs['frozen']) / gap if gap > 1e-9 else None

        M_star, C_star = class_means(Xp, pl, NUM_CLASSES)
        R = (Ws_c - W0c)
        R_norm = R.norm().item()
        p0c = C0 / (C0.sum() + 1e-12)
        p_sc = C_star / (C_star.sum() + 1e-12)

        # ================= PART A: COVARIANCE-ONLY DECODER CEILING ============
        # additive in W (R = W* - W0)
        W_mean_oracle = solve_whitened(Xp, (M_star * C0.unsqueeze(1)).t().contiguous(),
                                       args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        R_mean = (W_mean_oracle - W0c)
        R_prior = solve_whitened(Xp, ((p_sc - p0c).unsqueeze(1) * M_star).t().contiguous(),
                                 args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        # independent covariance: W* - W_mean_oracle = R_cov + R_prior
        # (W* = Sigma*^-1 P* M* is the oracle ridge; W_mean_oracle = Sigma0^-1 P0 M*)
        decoders = {
            'W0': W0c,
            'mean_only': (W0c + R_mean),
            'prior_only': (W0c + R_prior),
            'cov_only': (W0c + (Ws_c - W_mean_oracle)),
            'mean_prior': (W0c + R_mean + R_prior),
            'mean_cov': (W0c + R_mean + (Ws_c - W_mean_oracle)),
            'full_oracle': Ws_c,
        }
        A = {}
        for name, Wd in decoders.items():
            A[name] = {'gc': gc(mw(Wd, Xv, vl)),
                       'dec_agree': dec_agree(Wd, Xv, vl, Ws),
                       'dec_agree_errs': dec_agree(Wd, Xv, vl, Ws, err_only=True)}
        A['R_cov_norm_frac'] = float((Ws_c - W_mean_oracle).norm().item() / (R_norm + 1e-12))
        A['R_mean_norm_frac'] = float(R_mean.norm().item() / (R_norm + 1e-12))

        # ================= PART B: PAIRWISE LOGIT CORRECTION ===================
        # top pairs by oracle pairwise residual
        pair_cands = []
        for a in range(1, NUM_CLASSES):
            for b in range(a + 1, NUM_CLASSES):
                if C_star[a] < 20 or C_star[b] < 20:
                    continue
                d = (Ws_c[:, a] - Ws_c[:, b]) - (W0c[:, a] - W0c[:, b])
                pair_cands.append((a, b, float(d.norm().item())))
        pairs = [(a, b) for a, b, _ in sorted(pair_cands, key=lambda x: -x[2])[:args.n_pairs]]
        torch.manual_seed(99)
        random_pairs = []
        while len(random_pairs) < args.n_pairs:
            a = int(torch.randint(NUM_CLASSES, (1,)).item())
            b = int(torch.randint(NUM_CLASSES, (1,)).item())
            if a == b or a == 0 or b == 0:
                continue
            if C_star[a] < 20 or C_star[b] < 20:
                continue
            if (a, b) in random_pairs or (b, a) in random_pairs:
                continue
            random_pairs.append((a, b))

        Lv = Xv.float() @ W0c
        Lp = Xp.float() @ W0c

        B = {'pairs': [f"{a}-{b}" for a, b in pairs], 'per_pair': {}, 'random_pairs': {},
             'global_bias': {}, 'shared_scalar': {}}

        for (a, b) in pairs:
            # pool points with true label a or b (oracle)
            mask_ab = (pl == a) | (pl == b)
            zi = Lp[mask_ab]
            za = zi[:, a]; zb = zi[:, b]
            tgt = (pl[mask_ab] == a).float() * 1.0 + (pl[mask_ab] == b).float() * (-1.0)
            # B1 oracle ceiling
            alpha_o, beta_o = fit_pair_logit(za, zb, None, tgt)
            pred = Lv.clone()
            corr_o = alpha_o * (Lv[:, a] - Lv[:, b]) + beta_o
            pred[:, a] += corr_o
            pred[:, b] -= corr_o
            entry = {'oracle': {'alpha': alpha_o, 'beta': beta_o,
                                'gc': gc(compute_miou(pred.argmax(1), vl))}}
            # B2 label estimation
            for bb in b_labels:
                za_l = []; zb_l = []; tgt_l = []
                for c, sgn in [(a, 1.0), (b, -1.0)]:
                    idx = torch.nonzero(pl == c).squeeze(1)
                    if len(idx) == 0:
                        continue
                    torch.manual_seed(7 + c)
                    sub = idx[torch.randperm(len(idx))[:min(bb, len(idx))]]
                    zs = Lp[sub]
                    za_l.append(zs[:, a]); zb_l.append(zs[:, b])
                    tgt_l.append(torch.full((len(sub),), sgn))
                if len(za_l) == 0 or len(tgt_l) == 0:
                    entry[f'label{bb}'] = {'gc': None}
                    continue
                za_c = torch.cat(za_l); zb_c = torch.cat(zb_l); tg_c = torch.cat(tgt_l)
                a_l, b_l = fit_pair_logit(za_c, zb_c, None, tg_c)
                pred_l = Lv.clone()
                corr_l = a_l * (Lv[:, a] - Lv[:, b]) + b_l
                pred_l[:, a] += corr_l
                pred_l[:, b] -= corr_l
                entry[f'label{bb}'] = {'gc': gc(compute_miou(pred_l.argmax(1), vl)),
                                       'alpha': a_l, 'beta': b_l}
            # B4 boundary-conditioned oracle fit
            m_ab = Lp[:, a] - Lp[:, b]
            bnd = (pl == a) | (pl == b)
            bnd &= (m_ab.abs() < args.tau)
            if int(bnd.sum().item()) >= 10:
                za_b = Lp[bnd][:, a]; zb_b = Lp[bnd][:, b]
                tgt_b = (pl[bnd] == a).float() * 1.0 + (pl[bnd] == b).float() * (-1.0)
                a_b, b_b = fit_pair_logit(za_b, zb_b, None, tgt_b)
                pred_b = Lv.clone()
                corr_b = a_b * (Lv[:, a] - Lv[:, b]) + b_b
                pred_b[:, a] += corr_b
                pred_b[:, b] -= corr_b
                entry['boundary_oracle'] = {'gc': gc(compute_miou(pred_b.argmax(1), vl)),
                                            'n': int(bnd.sum().item())}
            else:
                entry['boundary_oracle'] = None
            B['per_pair'][f"{a}-{b}"] = entry

        # B3 random pairs (same fit, oracle + b=8 label)
        for (a, b) in random_pairs:
            mask_ab = (pl == a) | (pl == b)
            zi = Lp[mask_ab]
            tgt = (pl[mask_ab] == a).float() * 1.0 + (pl[mask_ab] == b).float() * (-1.0)
            alpha_o, beta_o = fit_pair_logit(zi[:, a], zi[:, b], None, tgt)
            pred = Lv.clone()
            corr_o = alpha_o * (Lv[:, a] - Lv[:, b]) + beta_o
            pred[:, a] += corr_o
            pred[:, b] -= corr_o
            B['random_pairs'][f"{a}-{b}"] = {'gc': gc(compute_miou(pred.argmax(1), vl))}

        # B5 global per-class bias (17 scalars): shift each logit by the
        # median-logit gap between true-c and not-c points (oracle fit)
        gb = torch.zeros(NUM_CLASSES)
        for c in range(1, NUM_CLASSES):
            idx_c = torch.nonzero(pl == c).squeeze(1)
            idx_o = torch.nonzero(pl != c).squeeze(1)
            if len(idx_c) == 0 or len(idx_o) == 0:
                continue
            m_c = float(Lp[idx_c][:, c].median().item())
            m_o = float(Lp[idx_o][:, c].median().item())
            gb[c] = (m_o - m_c) / 2.0
        pred_g = Lv.clone()
        for c in range(1, NUM_CLASSES):
            pred_g[:, c] += gb[c]
        B['global_bias'] = {'gc': gc(compute_miou(pred_g.argmax(1), vl)),
                            'bias': {str(c): round(v, 3) for c, v in gb.items() if v != 0}}

        # B6 shared scalar: one (alpha, beta) across ALL confused pairs
        all_m = []; all_t = []
        for (a, b) in pairs:
            mask_ab = (pl == a) | (pl == b)
            m_p = Lp[mask_ab, a] - Lp[mask_ab, b]
            t_p = (pl[mask_ab] == a).float() * 1.0 + (pl[mask_ab] == b).float() * (-1.0)
            all_m.append(m_p); all_t.append(t_p)
        if all_m:
            am = torch.cat(all_m); at = torch.cat(all_t)
            A_sh = torch.stack([am, torch.ones_like(am)], dim=1)
            sol = torch.linalg.lstsq(A_sh.double(), at.double().unsqueeze(1)).solution
            a_sh = float(sol[0].item()); b_sh = float(sol[1].item())
            # apply to all pairs
            pred_s = Lv.clone()
            for (a, b) in pairs:
                corr_s = a_sh * (Lv[:, a] - Lv[:, b]) + b_sh
                pred_s[:, a] += corr_s
                pred_s[:, b] -= corr_s
            B['shared_scalar'] = {'gc': gc(compute_miou(pred_s.argmax(1), vl)),
                                  'alpha': a_sh, 'beta': b_sh}
        else:
            B['shared_scalar'] = {'gc': None}

        cond_res = {'refs': refs, 'gap': float(gap),
                    'A_ceiling': A, 'B_logit': B}
        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, M_star
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print("    A ceiling:")
        for name in ['W0', 'mean_only', 'prior_only', 'cov_only', 'mean_prior', 'mean_cov', 'full_oracle']:
            v = A[name]
            print(f"      {name:12s} gc {v['gc']:+.2f} da {v['dec_agree']:.3f} dae {v['dec_agree_errs']:.3f}")
        print(f"      R_mean_frac {A['R_mean_norm_frac']:.4f} R_cov_frac {A['R_cov_norm_frac']:.4f}")
        print("    B logit per pair:")
        for k, v in B['per_pair'].items():
            or_g = v['oracle']['gc']
            lb = " ".join(f"b{bb}:{v[f'label{bb}']['gc']:+.2f}" if v.get(f'label{bb}') and v[f'label{bb}']['gc'] is not None else f"b{bb}:NA"
                          for bb in b_labels)
            bnd = v['boundary_oracle']['gc'] if v['boundary_oracle'] else None
            bnd_s = f"{bnd:+.2f}" if bnd is not None else "NA"
            print(f"      {k}: or {or_g:+.2f} | {lb} | bnd {bnd_s}")
        print("    B random pairs: " + " ".join(f"{k}:{v['gc']:+.2f}" for k, v in B['random_pairs'].items()))
        print("    B global_bias gc %+.2f | shared_scalar %s" % (B['global_bias']['gc'], B['shared_scalar']))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("PART A (covariance-only ceiling):")
    print("  cov_only gc ~ oracle gc and dec_agree ~ 1.0 -> the remaining problem")
    print("  IS covariance adaptation (closes the class-stats line decisively).")
    print("  If cov_only does NOT reproduce the classifier, subtle issue to chase.")
    print("  R_cov_frac = ||W* - W_mean_oracle||/||R|| (independent, not residual).")
    print("PART B (logit correction ceiling):")
    print("  B1 oracle per-pair gc -> which pairs actually benefit")
    print("  B2 label-estimation gc vs oracle -> does few-label track?")
    print("  B3 random pairs -> is the gain confusion-concentrated (good for AL)?")
    print("  B4 boundary-conditioned fit -> more statistically efficient?")
    print("  B5 global bias (17 scalars) vs the O(K^2) pairwise form")
    print("  B6 shared scalar (ONE alpha,beta) -> tiny parameter count the key test")


if __name__ == "__main__":
    main()
