"""al_class_stats_iter4_diag.py: Iteration 4 -- push the two green-light findings
from Iteration 3 (class_stats_iters.md), all white-box.

Iteration 3 established: (1) the oracle mean direction is ROBUST (50% corruption
retains 70-88% gc) -- the estimator problem is a precision gap, not structural;
(2) Arm C (few-label C x C confusion correction) is the first positive few-label
mechanism (+0.01 to +0.04). This iteration pushes those two, plus the class-prior
correction.

PART 1 -- REFINED ARM C (confusion correction). Few labels estimate the C x C
matrix Q_cj = P(pseudo = j | true = c); M_corr_c = sum_j Q_cj M_tilde_j. Variants:
  C0 base      empirical Q from the b labels (Iteration-3 form), full rows
  C1 pool-reg  shrink each Q row toward the POOL pseudo-label marginal prior
               Q_cj = (n_c emp_cj + alpha prior_j) / (n_c + alpha) -- fights the
               8-label sample noise using the pool's low-variance prior
  C2 iterated  self-training: correct M, re-pseudo-label the pool with the
               corrected decoder, re-estimate Q from the same labels, iterate
  C3 soft-Q    weight each labeled point's row contribution by its pseudo-class
               confidence (the labels that agree with the pool count more)
  Also reports Q estimation error ||Q_hat - Q_oracle||_F vs the FULL-pool Q.

PART 2 -- ARM B with ALTERNATIVE pool-derived directions (the failure in
Iteration 3 was the noisy direction v = M_pseudo - M0, not the scalar gamma).
For each direction family:
  v_pseudo      M_pseudo_c - M0_c                     (the Iteration-3 failure)
  v_highconf    M_highconf_c - M0_c  (mean of pool points with pseudo-class c
                AND frozen confidence > thresh -- the confident core)
  v_density     M_core_c - M0_c      (mean of the points nearest M_pseudo_c
                within pseudo-class c -- the density core)
gamma_c = <M_lab_c - M0_c, v_c>/||v_c||^2 (labels estimate only the scalar);
M_corr_c = M0_c + gamma_c v_c. Report gc AND the ORACLE direction alignment
cos(v_family_c, M*_c - M0_c) -- WHICH pool-derived direction actually points
toward the true shift (diagnostic-only, tells us if any family is trustworthy).

PART 3 -- CLASS-PRIOR CORRECTION (the residual term never tested). W = Sigma^-1
M^T P with the prior P estimated from the POOL:
  P_pseudo    pseudo-label proportions (label-free)
  P_oracle    true pool class proportions (ceiling)
For same-scan corruptions P* ~ P0, so this is expected to be small -- but it is
the one residual term that has never been measured.

Usage:
  uv run python robust_diagnostic/al_class_stats_iter4_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_class_stats_iter4_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11


def F_normalize(x):
    return torch.nn.functional.normalize(x, p=2, dim=1)


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
    ap.add_argument("--b_per_class", type=str, default="2,4,8")
    ap.add_argument("--k_sweep", type=str, default="3,5")
    ap.add_argument("--alpha_reg", type=str, default="2,8,32")
    ap.add_argument("--conf_thresh", type=float, default=0.5)
    ap.add_argument("--n_iter", type=int, default=2)
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
    b_sweep = [int(x) for x in args.b_per_class.split(',')]
    k_sweep = [int(x) for x in args.k_sweep.split(',')]
    alpha_reg = [float(x) for x in args.alpha_reg.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_per_class': b_sweep,
               'k_sweep': k_sweep, 'alpha_reg': alpha_reg, 'n_iter': args.n_iter,
               'conf_thresh': args.conf_thresh, 'conds': {}}

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
        R = (Ws - W0).detach().cpu().float()

        B_mean_oracle = (M_star * C0.unsqueeze(1)).t().contiguous()
        W_mean_oracle = solve_whitened(Xp, B_mean_oracle, args.lam, args.cg_iters, args.nystrom_m, device)

        # frozen probe on the pool
        Lp = Xp.float() @ W0c
        p0 = torch.softmax(Lp, dim=1)
        pseudo = Lp.argmax(1)
        pseudo_conf = p0.gather(1, pseudo.unsqueeze(1)).squeeze(1)
        M_hard, C_ph = class_means(Xp, pseudo, NUM_CLASSES)
        # pool pseudo-label marginal prior (for C1 regularization)
        prior_j = torch.zeros(NUM_CLASSES)
        for j in range(1, NUM_CLASSES):
            prior_j[j] = float((pseudo == j).float().mean().item())

        # oracle Q from the full pool (diagnostic reference)
        Q_oracle = torch.zeros(NUM_CLASSES, NUM_CLASSES)
        for c in range(1, NUM_CLASSES):
            m = (pl == c)
            if int(m.sum().item()) > 0:
                for j in range(NUM_CLASSES):
                    Q_oracle[c, j] = float((pseudo[m] == j).float().mean().item())

        # suspicious classes (pool evidence, exclude unlabeled 0)
        shift_norm = torch.norm(M_hard - M0, p=2, dim=1)
        suspicious = [int(c) for c in torch.argsort(shift_norm, descending=True) if c != 0]

        # per-class labeled indices
        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}
        Xp_cpu = Xp

        # ---- PART 2: alternative pool-derived directions (need M_highconf, M_core)
        M_highconf = M0.clone()
        M_core = M0.clone()
        for c in range(1, NUM_CLASSES):
            m = (pseudo == c)
            if int(m.sum().item()) < 10:
                continue
            idx = torch.nonzero(m).squeeze(1)
            conf = pseudo_conf[idx]
            hc = idx[conf > args.conf_thresh]
            if len(hc) >= 5:
                M_highconf[c] = Xp_cpu[hc].mean(dim=0)
            # density core: the half of pseudo-c points nearest to M_pseudo_c
            core_sim = F_normalize(Xp_cpu[idx]) @ F_normalize(M_hard[c].unsqueeze(0)).t()
            core = idx[torch.argsort(core_sim[:, 0], descending=True)[:max(len(idx) // 2, 5)]]
            M_core[c] = Xp_cpu[core].mean(dim=0)
        directions = {'pseudo': M_hard - M0, 'highconf': M_highconf - M0, 'density': M_core - M0}

        # oracle direction alignment (diagnostic): cos(v_c, M*_c - M0_c)
        align = {}
        for dname, Dv in directions.items():
            coss = []
            for c in suspicious[:5]:
                a = Dv[c]; b = M_star[c] - M0[c]
                if a.norm().item() < 1e-8 or b.norm().item() < 1e-8:
                    continue
                coss.append(float((a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12)))
            align[dname] = sum(coss) / len(coss) if coss else None

        cond_res = {'refs': refs, 'gap': float(gap),
                    'ladder': {'W0': 0.0,
                               'W_mean_oracle': gc(mw(W_mean_oracle, Xv, vl)),
                               'W*': 1.0},
                    'dir_align': align, 'suspicious': suspicious[:8],
                    'part1': {}, 'part2': {}, 'part3': {}}

        # ================= PART 1: refined ARM C (confusion correction) ========
        for b in b_sweep:
            # labeled points per class (seeded, consistent)
            lab_sub = {}
            for c in range(1, NUM_CLASSES):
                idx = class_idx[c]
                if len(idx) == 0:
                    lab_sub[c] = None
                    continue
                torch.manual_seed(7 + c)
                lab_sub[c] = idx[torch.randperm(len(idx))[:min(b, len(idx))]]
            # empirical Q from labels (full rows)
            Q_emp = torch.zeros(NUM_CLASSES, NUM_CLASSES)
            for c in range(1, NUM_CLASSES):
                if lab_sub[c] is None:
                    continue
                plab = pseudo[lab_sub[c]]
                for j in range(NUM_CLASSES):
                    Q_emp[c, j] = float((plab == j).float().mean().item())
            Q_err = float((Q_emp - Q_oracle).norm().item() /
                          (Q_oracle.norm().item() + 1e-12))

            for K in k_sweep:
                sus = suspicious[:K]
                # C0 base
                M_c = M_hard.clone()
                for c in sus:
                    M_c[c] = sum(Q_emp[c, j] * M_hard[j] for j in range(1, NUM_CLASSES))
                W_c = solve_whitened(Xp, (M_c * C0.unsqueeze(1)).t().contiguous(),
                                     args.lam, args.cg_iters, args.nystrom_m, device)
                cond_res['part1'].setdefault('C0', {}).setdefault(str(b), {})[str(K)] = {
                    'gc': gc(mw(W_c, Xv, vl)), 'Q_err': Q_err}
                # C1 pool-reg
                for alpha in alpha_reg:
                    M_c = M_hard.clone()
                    for c in sus:
                        row = (Q_emp[c] * b + alpha * prior_j) / (b + alpha)
                        row[0] = 0.0
                        row = row / (row.sum() + 1e-12)
                        M_c[c] = sum(row[j] * M_hard[j] for j in range(1, NUM_CLASSES))
                    W_c = solve_whitened(Xp, (M_c * C0.unsqueeze(1)).t().contiguous(),
                                         args.lam, args.cg_iters, args.nystrom_m, device)
                    cond_res['part1'].setdefault('C1', {}).setdefault(str(b), {}) \
                        .setdefault(str(K), {})[str(alpha)] = {'gc': gc(mw(W_c, Xv, vl))}
                # C2 iterated (self-training: re-pseudo-label with corrected W)
                # uses LOCAL copies so later b/K/parts are not contaminated.
                M_cur = M_hard.clone()
                Q_c = Q_emp.clone()
                W_cur = None
                for it in range(args.n_iter):
                    for c in sus:
                        M_cur[c] = sum(Q_c[c, j] * M_hard[j] for j in range(1, NUM_CLASSES))
                    W_cur = solve_whitened(Xp, (M_cur * C0.unsqueeze(1)).t().contiguous(),
                                           args.lam, args.cg_iters, args.nystrom_m, device)
                    # re-pseudo-label the pool with the corrected decoder
                    new_pseudo = (Xp.float() @ W_cur.detach().cpu()).argmax(1)
                    # re-estimate Q from the SAME labels with new pseudo labels
                    for c in range(1, NUM_CLASSES):
                        if lab_sub[c] is None:
                            continue
                        plab = new_pseudo[lab_sub[c]]
                        for j in range(NUM_CLASSES):
                            Q_c[c, j] = float((plab == j).float().mean().item())
                    # update the pseudo-mean table from the new pseudo labels
                    M_cur = class_means(Xp, new_pseudo, NUM_CLASSES)[0]
                cond_res['part1'].setdefault('C2', {}).setdefault(str(b), {})[str(K)] = {
                    'gc': gc(mw(W_cur, Xv, vl))}

        # ================= PART 2: ARM B with alternative directions ===========
        for b in b_sweep:
            obs_means = M0.clone(); obs_counts = torch.zeros(NUM_CLASSES)
            for c in range(1, NUM_CLASSES):
                idx = class_idx[c]
                if len(idx) == 0:
                    continue
                torch.manual_seed(7 + c)
                sub = idx[torch.randperm(len(idx))[:min(b, len(idx))]]
                obs_means[c] = Xp_cpu[sub].mean(dim=0)
                obs_counts[c] = float(len(sub))
            for dname, Dv in directions.items():
                for K in k_sweep:
                    M_c = M0.clone()
                    for c in suspicious[:K]:
                        v_c = Dv[c]
                        vn = v_c.norm().item() ** 2 + 1e-12
                        gam = float(((obs_means[c] - M0[c]) * v_c).sum().item() / vn) \
                            if obs_counts[c] > 0 else 0.0
                        M_c[c] = M0[c] + gam * v_c
                    W_c = solve_whitened(Xp, (M_c * C0.unsqueeze(1)).t().contiguous(),
                                         args.lam, args.cg_iters, args.nystrom_m, device)
                    cond_res['part2'].setdefault(dname, {}).setdefault(str(b), {})[str(K)] = {
                        'gc': gc(mw(W_c, Xv, vl))}

        # ================= PART 3: class-prior correction ======================
        # W = Sigma^-1 M_hard^T P with P from the pool
        # P_pseudo (label-free): pseudo-label proportions
        P_pseudo = prior_j.clone()
        P_pseudo[0] = 0.0
        B_pp = (M_hard * P_pseudo.unsqueeze(1)).t().contiguous()
        W_pp = solve_whitened(Xp, B_pp, args.lam, args.cg_iters, args.nystrom_m, device)
        cond_res['part3']['P_pseudo'] = gc(mw(W_pp, Xv, vl))
        # P_oracle (ceiling): true pool class proportions
        P_or = torch.zeros(NUM_CLASSES)
        for c in range(1, NUM_CLASSES):
            P_or[c] = float((pl == c).float().mean().item())
        B_po = (M_hard * P_or.unsqueeze(1)).t().contiguous()
        W_po = solve_whitened(Xp, B_po, args.lam, args.cg_iters, args.nystrom_m, device)
        cond_res['part3']['P_oracle'] = gc(mw(W_po, Xv, vl))

        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, M_star, Xp_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        l = cond_res['ladder']
        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    ladder W0 {l['W0']:+.2f} W_mean_oracle {l['W_mean_oracle']:+.2f} W* {l['W*']:+.2f}")
        print("    dir_align: " + " ".join(f"{k}:{v:+.2f}" if v is not None else f"{k}:NA"
                                             for k, v in align.items()))
        for b in b_sweep:
            c0 = " ".join(f"K{k}:{cond_res['part1']['C0'][str(b)][str(k)]['gc']:+.2f}" for k in k_sweep)
            c1 = " ".join(f"a{a}:{cond_res['part1']['C1'][str(b)][str(list(k_sweep)[0])][str(a)]['gc']:+.2f}"
                          for a in alpha_reg)
            c2 = " ".join(f"K{k}:{cond_res['part1']['C2'][str(b)][str(k)]['gc']:+.2f}" for k in k_sweep)
            print(f"    b{b}: C0[{c0}] C1[{c1}] C2[{c2}]")
        for dname in directions:
            line = " ".join(f"b{b}:{max(cond_res['part2'][dname][str(b)][str(k)]['gc'] for k in k_sweep):+.2f}"
                            for b in b_sweep)
            print(f"    dir[{dname:8s}] {line}")
        print(f"    part3 P_pseudo {cond_res['part3']['P_pseudo']:+.2f} P_oracle {cond_res['part3']['P_oracle']:+.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("PART 1 (confusion correction): C1 (pool-regularized Q) or C2 (iterated)")
    print("  > C0 (base) -> the 8-label Q noise was the limiter; pool prior /")
    print("  self-training helps. Q_err = ||Q_hat - Q_oracle||/||Q_oracle||.")
    print("PART 2 (directions): dir_align tells WHICH pool direction actually")
    print("  points at the true shift (diagnostic). gc per direction vs the")
    print("  Iteration-3 'pseudo' failure.")
    print("PART 3 (prior): P_oracle ~ W_mean_oracle? -> the prior term matters.")
    print("  If P_pseudo ~ P_oracle gc, the label-free prior correction works.")



if __name__ == "__main__":
    main()
