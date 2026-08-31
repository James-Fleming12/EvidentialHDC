"""al_propagation_improve_diag.py: the improvement-axes diagnostic for the
PROPAGATED-MEAN decoder (the Method section, First implementation).

Iteration 10 + validation established: the propagated-mean decoder gives real
positive gc (+0.045 fog / +0.095 crosstalk mIoU at ~112-125 labels) but the
headroom vs W_mean_oracle is UNKNOWN -- the "oracle-count ceiling" comparison
was count-reference-mismatched. This diagnostic tests the improvement axes
from the doc, organized by which part of the pipeline they improve:

A. THE MEAN ESTIMATOR (the propagated means M_prop):
   A1  M_star x C_prop (TRUE means x PROPAGATED counts) vs M_prop x C_prop --
       holds counts fixed, isolates MEAN quality cleanly (the clean split the
       oracle-count arm muddled). If true-means-with-propagated-counts ~
       W_mean_oracle, the propagated COUNTS are fine and the means are the real
       bottleneck; if far below, the counts are part of the gap.
   A2  128-d mean aggregation: the propagation lives in 128-d but the means are
       aggregated in the 10000-d code space (saturated). Compute the 128-d
       means and report their quality vs the 128-d oracle means, AND the decoder
       using 128-d-projected means.
   A3  Agreement-gated means: aggregate only the points where the frozen probe
       agrees with the propagated label (raises per-point precision to 0.5-0.66).
   A4  Per-class anchor budget: more anchors for the loose classes {7,13,14}.
   A5  Weighted (soft) propagation: softmax-weighted assignment by 128-d
       similarity instead of hard nearest-anchor.

B. THE AL SELECTION (which points to query):
   B1  Confidence vs random anchors (the prior docs: high-confidence points ARE
       the centroid-near representatives -- the free self-selecting rule).
   B2  Mass-stratified anchors (pick in proportion to class mass, so the rare
       classes are not starved).
   B3  Boundary-avoiding anchors (high frozen margin; the bulk transfers, per
       the C1 finding that precision increases with margin).

C. THE UPDATE / DECODER:
   C1  Fractional whitening Sigma^-beta (beta in {0.25, 0.5}): Iteration 6
       showed fractional whitening reduces sensitivity to mean error.
   C2  Update-norm constraint: scale the propagated update toward c * ||R||.
   C3  Mean shrinkage toward the pseudo-mean: M_est = (1-a) M_prop + a M_pseudo.

All compared to the references: W0 (frozen), gcP (the current method), and
W_mean_oracle (the +0.50-1.15 ceiling). DGLSS++ only (fog/crosstalk).

Decisive reads:
  A1 gc ~ W_mean_oracle      -> the propagated COUNTS are the gap, not the means
  A1 gc << W_mean_oracle     -> the MEANS are the (real) bottleneck
  A2/A3/A4/A5 > gcP          -> a better mean estimator exists (improve it)
  B1/B2/B3 > random gc       -> a better anchor selection exists (use it)
  C1/C2/C3 > gcP             -> a better update exists (use it)

Usage:
  uv run python robust_diagnostic/al_propagation_improve_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_propagation_improve_dglsspp.json
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
LOOSE = {7, 13, 14}


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


def topk_eigbasis(X, k, device):
    """Randomized SVD of X -> top-k right singular vectors Q (d x k) and
    singular values. For fractional whitening."""
    X = X.to(device)
    n, d = X.shape
    k = min(k, min(n, d) - 1)
    torch.manual_seed(SKETCH_SEED)
    Omega = torch.randn(d, k + 8, device=device)
    Y = X @ Omega
    Q, _ = torch.linalg.qr(Y)
    Bm = Q.t() @ X
    U, S, Vh = torch.linalg.svd(Bm, full_matrices=False)
    Qe = Vh[:k].t().contiguous()
    sig = S[:k].clamp(min=1e-8)
    return Qe.cpu(), sig.cpu()


def frac_solve(Qe, sig, lam, B, beta):
    """Sigma^-beta B using the top-k eigenbasis. B is d x C on CPU."""
    lamb = (sig ** 2 + lam)
    proj = Qe.t() @ B                      # k x C
    return Qe @ (proj / (lamb ** beta).unsqueeze(1))


def cos_(a, b):
    return float((a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-12))


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
    ap.add_argument("--b_anchors", type=str, default="2,8")
    ap.add_argument("--loose_mult", type=float, default=3.0, help="per-class anchor mult for loose classes A4")
    ap.add_argument("--beta_sweep", type=str, default="0.25,0.5,1.0")
    ap.add_argument("--c_sweep", type=str, default="0.5,1.0,1.5")
    ap.add_argument("--shrink_sweep", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--k_eig", type=int, default=512)
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
    beta_sweep = [float(x) for x in args.beta_sweep.split(',')]
    c_sweep = [float(x) for x in args.c_sweep.split(',')]
    shrink_sweep = [float(x) for x in args.shrink_sweep.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_anchors': b_anchors,
               'loose_mult': args.loose_mult, 'beta_sweep': beta_sweep,
               'c_sweep': c_sweep, 'shrink_sweep': shrink_sweep, 'conds': {}}

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
        W_mean_oracle = solve_whitened(Xp, (M_star * C0.unsqueeze(1)).t().contiguous(),
                                       args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        gc_mean_oracle = gc(mw(W_mean_oracle, Xv, vl))

        pf = F.normalize(pool_f.float(), p=2, dim=1)

        # frozen probe softmax (for confidence selection B1) and margin (B3)
        Lp = Xp.float() @ W0c
        sm = torch.softmax(Lp, dim=1)
        conf = sm.max(dim=1).values
        top2 = torch.topk(Lp, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
        pred = Lp.argmax(1)

        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}

        # fractional whitening eigenbasis
        Qe, sig = topk_eigbasis(Xp, args.k_eig, device)

        cond_res = {'refs': refs, 'gap': float(gap),
                    'gc_mean_oracle': gc_mean_oracle, 'budgets': {}}

        for b in b_anchors:
            # ---- anchor selection variants ----
            # random (the current method)
            torch.manual_seed(7)
            anc_rand = torch.cat([class_idx[c][torch.randperm(len(class_idx[c]))[:min(b, len(class_idx[c]))]]
                                  for c in range(1, NUM_CLASSES) if len(class_idx[c]) > 0])
            # confidence (B1): highest frozen confidence per class
            anc_conf = torch.cat([class_idx[c][torch.argsort(conf[class_idx[c]], descending=True)[:min(b, len(class_idx[c]))]]
                                  for c in range(1, NUM_CLASSES) if len(class_idx[c]) > 0])
            # mass-stratified (B2): pick anchors proportional to class mass
            torch.manual_seed(9)
            mass = torch.tensor([float(len(class_idx[c])) for c in range(1, NUM_CLASSES)])
            anc_mass = []
            total_b = b * (NUM_CLASSES - 1)
            alloc = (mass / mass.sum() * total_b).int().clamp(min=1)
            for i, c in enumerate(range(1, NUM_CLASSES)):
                idx = class_idx[c]
                nb = int(min(alloc[i].item(), len(idx)))
                anc_mass.append(idx[torch.randperm(len(idx))[:nb]])
            anc_mass = torch.cat(anc_mass)
            # boundary-avoiding (B3): highest margin per class
            anc_bnd = torch.cat([class_idx[c][torch.argsort(margin[class_idx[c]], descending=True)[:min(b, len(class_idx[c]))]]
                                 for c in range(1, NUM_CLASSES) if len(class_idx[c]) > 0])
            # per-class budget (A4): loose classes get loose_mult * b
            torch.manual_seed(7)
            anc_loose = []
            for c in range(1, NUM_CLASSES):
                idx = class_idx[c]
                nb = int(b * (args.loose_mult if c in LOOSE else 1.0))
                nb = min(nb, len(idx))
                anc_loose.append(idx[torch.randperm(len(idx))[:nb]])
            anc_loose = torch.cat(anc_loose)

            def propagate(anchors):
                anc_f = pf[anchors]; anc_lab = pl[anchors]
                nn = (pf @ anc_f.t()).argmax(1)
                return anc_lab[nn]

            def decoder(M, C):
                B = (M * C.unsqueeze(1)).t().contiguous()
                W = solve_whitened(Xp, B, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
                return gc(mw(W, Xv, vl))

            prop_rand = propagate(anc_rand)
            M_r, C_r = class_means(Xp, prop_rand, NUM_CLASSES)
            gcP = decoder(M_r, C_r)
            # A1: TRUE means x PROPAGATED counts (the clean split)
            gcA1 = decoder(M_star, C_r)
            # A2: 128-d mean aggregation -> project to code -> decoder
            M128 = torch.zeros(NUM_CLASSES, pf.shape[1])
            for c in range(1, NUM_CLASSES):
                m = prop_rand == c
                if int(m.sum().item()) > 0:
                    M128[c] = pool_f[m].float().mean(dim=0)
            # project 128-d mean into code space (sign of proj)
            M128_code = torch.sign(M128 @ proj.cpu().float())
            gcA2 = decoder(M128_code, C_r)
            # A3: agreement-gated means (propagate only where probe agrees)
            agree = pred == prop_rand
            M_ag = torch.zeros(NUM_CLASSES, Xp.shape[1])
            C_ag = torch.zeros(NUM_CLASSES)
            for c in range(1, NUM_CLASSES):
                m = (prop_rand == c) & agree
                if int(m.sum().item()) > 0:
                    M_ag[c] = Xp[m].float().mean(dim=0)
                    C_ag[c] = float(int(m.sum().item()))
            gcA3 = decoder(M_ag, C_ag)
            # A5: soft/weighted propagation
            anc_f = pf[anc_rand]
            sim = pf @ anc_f.t()
            w = torch.softmax(sim / 0.1, dim=1)          # temperature 0.1
            M_soft = torch.zeros(NUM_CLASSES, Xp.shape[1])
            C_soft = torch.zeros(NUM_CLASSES)
            for c in range(1, NUM_CLASSES):
                mask = pl[anc_rand] == c
                if int(mask.sum().item()) == 0:
                    continue
                wc = w[:, mask].sum(dim=1)
                if wc.sum().item() < 1e-9:
                    continue
                M_soft[c] = (Xp.float() * wc.unsqueeze(1)).sum(dim=0) / wc.sum().item()
                C_soft[c] = float(wc.sum().item())
            gcA5 = decoder(M_soft, C_soft)
            # B: selection variants with the SAME decoder
            M_b1, C_b1 = class_means(Xp, propagate(anc_conf), NUM_CLASSES)
            gcB1 = decoder(M_b1, C_b1)
            M_b2, C_b2 = class_means(Xp, propagate(anc_mass), NUM_CLASSES)
            gcB2 = decoder(M_b2, C_b2)
            M_b3, C_b3 = class_means(Xp, propagate(anc_bnd), NUM_CLASSES)
            gcB3 = decoder(M_b3, C_b3)
            M_a4, C_a4 = class_means(Xp, propagate(anc_loose), NUM_CLASSES)
            gcA4 = decoder(M_a4, C_a4)

            # C: update variants on the current M_r, C_r
            # C1 fractional whitening
            B_r = (M_r * C_r.unsqueeze(1)).t().contiguous()
            gcC1 = {}
            for beta in beta_sweep:
                Wb = frac_solve(Qe, sig, args.lam, B_r, beta).cpu()
                gcC1[str(beta)] = gc(mw(Wb, Xv, vl))
            # C2 update-norm constraint: W0 + c * (W_r - W0) / ||W_r - W0|| * ||R||
            W_r_full = solve_whitened(Xp, B_r, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
            D = (W_r_full - W0c)
            dn = D.norm().item() + 1e-12
            R_norm = (Ws.detach().cpu() - W0c).norm().item()
            gcC2 = {}
            for c_ in c_sweep:
                W_c2 = W0c + (D / dn * R_norm * c_)
                gcC2[str(c_)] = gc(mw(W_c2, Xv, vl))
            # C3 mean shrinkage toward pseudo-mean
            M_pseudo, _ = class_means(Xp, pred, NUM_CLASSES)
            gcC3 = {}
            for a in shrink_sweep:
                M_sh = (1 - a) * M_r + a * M_pseudo
                gcC3[str(a)] = decoder(M_sh, C_r)

            cond_res['budgets'][str(b)] = {
                'gcP': gcP, 'gc_mean_oracle': gc_mean_oracle,
                'A1_true_mean_prop_counts': gcA1,
                'A2_128d_means': gcA2, 'A3_agreement': gcA3,
                'A4_perclass_budget': gcA4, 'A5_soft': gcA5,
                'B1_confidence': gcB1, 'B2_mass': gcB2, 'B3_boundary': gcB3,
                'C1_fractional': gcC1, 'C2_normscale': gcC2, 'C3_shrink': gcC3,
                'n_labels': {'rand': int(len(anc_rand)), 'conf': int(len(anc_conf)),
                             'mass': int(len(anc_mass)), 'bnd': int(len(anc_bnd)),
                             'loose': int(len(anc_loose))}}
            print(f"  b{b}: gcP {gcP:+.2f} | A1 {gcA1:+.2f} A2 {gcA2:+.2f} A3 {gcA3:+.2f} "
                  f"A4 {gcA4:+.2f} A5 {gcA5:+.2f}")
            print("      B1 %+.2f B2 %+.2f B3 %+.2f | " % (gcB1, gcB2, gcB3) +
                  "C1 " + " ".join(f"b{k}:{v:+.2f}" for k, v in gcC1.items()) +
                  " C2 " + " ".join(f"c{k}:{v:+.2f}" for k, v in gcC2.items()) +
                  " C3 " + " ".join(f"a{k}:{v:+.2f}" for k, v in gcC3.items()))

        results['conds'][cond] = cond_res
        del Ws, M_star, pool_f, Qe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    W_mean_oracle gc {gc_mean_oracle:+.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. MEAN ESTIMATOR:")
    print("   A1 (true means x prop counts) ~ W_mean_oracle -> the COUNTS are the")
    print("      gap, not the means; A1 << -> the MEANS are the real bottleneck.")
    print("   A2 (128-d means) / A3 (agreement) / A4 (loose budget) / A5 (soft)")
    print("      vs gcP -> a better mean estimator exists?")
    print("B. AL SELECTION: B1 confidence / B2 mass / B3 boundary vs random gcP.")
    print("C. UPDATE: C1 fractional whitening, C2 norm-constrained step, C3 mean")
    print("   shrinkage toward the pseudo-mean -- any better than gcP?")


if __name__ == "__main__":
    main()
