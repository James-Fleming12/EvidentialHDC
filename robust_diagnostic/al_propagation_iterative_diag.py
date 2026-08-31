"""al_propagation_iterative_diag.py: make MORE use of the labeled points on the
PROPAGATED-MEAN decoder (DGLSS++ fog/crosstalk only).

The two untested levers (targeted acquisition, composition) just closed
(al_propagation_targeted_diag.py): random anchors are unbeaten, the
decision-correction adds nothing on top of propagation. The headroom analysis
is complete: the bottleneck is the propagated MEANS (A1 true means x prop
counts = +0.45-0.81 gc vs gcP +0.26-0.45), every mean-estimator axis is closed
(A2/A3/A5/C1/C3), and the geometry gate says even correct assignments leave
Werr ~1.4-1.5x. The labels currently drive a SINGLE propagation pass. This
diagnostic tests the remaining design lever: use the points more, and denoise
the means.

Arms (all at the SAME labeled set = random anchors, seed 7, b per class):

1. gcP baseline: propagate -> means -> decoder (the current method).
2. ANCHORS-ONLY MEANS: the class means computed from the b labeled points
   THEMSELVES (no propagation, assignment-noise-free; classes are tight in
   128-d so an unbiased b-point mean may be better), at the SAME counts C_prop
   (isolates the mean estimator). The "make direct use of the points" arm.
3. ITERATIVE SELF-TRAINING (the main arm): loop
       W_cur -> decode pool -> confidence-gated pseudo-labels (conf > tau)
       -> new means -> new W_cur
   with the anchor-propagated labels held as the fallback below tau. This is
   "more use" (labels drive every round) AND "denoise the means" (a better
   decoder each round cleans the boundary assignments). Sweeps tau in
   {0.8, 0.95} and also a SOFT variant (weighted by the pseudo-class
   softmax at tau 0.9).
4. TOWARD-CLEAN SHRINKAGE (the inter-class/prior arm): M = (1-a) M_prop + a M0
   (shrink the corrupted propagated means toward the CLEAN means), a in
   {0.25, 0.5}, counts fixed at C_prop. The one form of cross-class structure
   not yet tried (C3 shrank toward the pseudo-mean and failed; the clean means
   are the better prior).

References: W_mean_oracle (+0.72 fog / +0.99 crosstalk) and A1
(true means x prop counts) are the mean-decoder ceilings to grow toward.

Decisive reads:
  iter gc > gcP across rounds            -> the loop is the method; grow toward
                                            A1 / W_mean_oracle
  iter gc plateaus at gcP                -> the single-pass design is not the
                                            bottleneck (means are intrinsically
                                            limited; the story closes)
  gc_anc_only > gcP                      -> propagation assignment noise HURTS;
                                            use the labeled points directly
  shrink a>0 > gcP                       -> the corrupted means are over-fit to
                                            the noisy pool; the clean prior helps

Usage:
  uv run python robust_diagnostic/al_propagation_iterative_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_propagation_iterative_dglsspp.json
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
P3_CLASSES = {11, 13, 14}          # driveable_surface, sidewalk, terrain
P3PAIRS = {(11, 13), (13, 11), (11, 14), (14, 11), (13, 14), (14, 13)}


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


def mean_rel_err(M, M_star, nc, classes):
    """Relative per-class mean error ||M[c]-M_star[c]|| / ||M_star[c]||."""
    d = (M - M_star).norm(dim=1)
    n = M_star.norm(dim=1).clamp(min=1e-8)
    vals = [float((d[c] / n[c]).item()) for c in classes]
    return sum(vals) / len(vals) if vals else None


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
    ap.add_argument("--k_rounds", type=int, default=4)
    ap.add_argument("--tau_sweep", type=str, default="0.8,0.95")
    ap.add_argument("--soft_tau", type=float, default=0.9)
    ap.add_argument("--shrink_sweep", type=str, default="0.25,0.5")
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
    tau_sweep = [float(x) for x in args.tau_sweep.split(',')]
    shrink_sweep = [float(x) for x in args.shrink_sweep.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_anchors': b_anchors,
               'k_rounds': args.k_rounds, 'tau_sweep': tau_sweep, 'shrink_sweep': shrink_sweep,
               'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    lac = la[ci]
    W0 = ridge_fit_soft(Xc, onehot(lac, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    M0_clean, C0 = class_means(Xc, lac, NUM_CLASSES)
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
        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}
        present = [c for c in range(1, NUM_CLASSES) if len(class_idx[c]) > 0]

        def propagate(anchors):
            anc_f = pf[anchors]; anc_lab = pl[anchors]
            nn = (pf @ anc_f.t()).argmax(1)
            return anc_lab[nn]

        def decoder(M, C):
            B = (M * C.unsqueeze(1)).t().contiguous()
            W = solve_whitened(Xp, B, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
            return gc(mw(W, Xv, vl))

        def decoder_W(W):
            return gc(mw(W, Xv, vl))

        cond_res = {'refs': refs, 'gap': float(gap), 'gc_mean_oracle': gc_mean_oracle,
                    'present_classes': present, 'budgets': {}}

        for b in b_anchors:
            # random anchors (the fixed labeled set for ALL arms)
            torch.manual_seed(7)
            anc_rand = torch.cat([class_idx[c][torch.randperm(len(class_idx[c]))[:min(b, len(class_idx[c]))]]
                                  for c in present])
            prop_rand = propagate(anc_rand)
            M_r, C_r = class_means(Xp, prop_rand, NUM_CLASSES)
            gcP = decoder(M_r, C_r)
            # A1 reference: true means x propagated counts (the mean bottleneck)
            gcA1 = decoder(M_star, C_r)
            # arm 2: anchors-only means (the labeled points themselves), counts fixed at C_prop
            M_anc, _ = class_means(Xp[anc_rand], pl[anc_rand], NUM_CLASSES)
            gc_anc = decoder(M_anc, C_r)
            # arm 3: iterative self-training (hard gate + soft variant)
            W_cur = solve_whitened(Xp, (M_r * C_r.unsqueeze(1)).t().contiguous(),
                                   args.lam, args.cg_iters, args.nystrom_m, device).cpu()
            # ---- FAILURE-MODE DIAGNOSTICS (where the method is wrong) ----
            # D1: where does the propagation assignment go wrong?
            prop_acc = float((prop_rand == pl).float().mean().item())
            err_m = prop_rand != pl
            n_err = int(err_m.sum().item())
            pair_code = prop_rand * NUM_CLASSES + pl
            in_p3 = torch.zeros(len(pl), dtype=torch.bool)
            for (a, b) in P3PAIRS:
                in_p3 |= (pair_code == a * NUM_CLASSES + b)
            err_in_p3 = float((err_m & in_p3).float().sum().item()) / n_err if n_err > 0 else None
            # D4 foundation: is the refit actually better on the pool?
            def pool_acc(W):
                return float(((Xp.float() @ W).argmax(1) == pl).float().mean().item())
            pool_acc_w0 = pool_acc(W0)
            p3_present = [c for c in present if c in P3_CLASSES]

            def record(M, W, gc_val, gate_prec=None, gate_cov=None):
                rec = {'gc': gc_val, 'pool_acc': pool_acc(W),
                       'mean_err': mean_rel_err(M, M_star, present),
                       'mean_err_p3': mean_rel_err(M, M_star, p3_present)}
                if gate_prec is not None:
                    rec['gate_prec'] = gate_prec; rec['gate_cov'] = gate_cov
                return rec

            iter_hard = {}
            for tau in tau_sweep:
                Wc = W_cur.clone()
                traj = {'r0': record(M_r, W_cur, gcP)}
                for r in range(1, args.k_rounds + 1):
                    L = Xp.float() @ Wc
                    sm = torch.softmax(L, dim=1)
                    conf = sm.max(dim=1).values
                    pred = L.argmax(1)
                    lab = prop_rand.clone()
                    gate = conf > tau
                    lab[gate] = pred[gate]
                    gate_prec = float((gate & (pred == pl)).float().sum().item() /
                                      max(1.0, float(gate.float().sum().item())))
                    gate_cov = float(gate.float().mean().item())
                    M_new, C_new = class_means(Xp, lab, NUM_CLASSES)
                    Wc = solve_whitened(Xp, (M_new * C_new.unsqueeze(1)).t().contiguous(),
                                        args.lam, args.cg_iters, args.nystrom_m, device).cpu()
                    traj[f'r{r}'] = record(M_new, Wc, decoder_W(Wc),
                                           gate_prec=gate_prec, gate_cov=gate_cov)
                iter_hard[str(tau)] = traj
            # soft variant at soft_tau: weight by the pseudo-class softmax where confident
            Wc = W_cur.clone()
            traj_soft = {'r0': gcP}
            for r in range(1, args.k_rounds + 1):
                L = Xp.float() @ Wc
                sm = torch.softmax(L, dim=1)
                conf = sm.max(dim=1).values
                gate = conf > args.soft_tau
                w = onehot(prop_rand, NUM_CLASSES)
                w[gate] = sm[gate]
                M_new = torch.zeros(NUM_CLASSES, Xp.shape[1]); C_new = torch.zeros(NUM_CLASSES)
                for c in range(1, NUM_CLASSES):
                    wc = w[:, c]
                    if wc.sum().item() < 1e-9:
                        continue
                    M_new[c] = (Xp.float() * wc.unsqueeze(1)).sum(dim=0) / wc.sum().item()
                    C_new[c] = float(wc.sum().item())
                Wc = solve_whitened(Xp, (M_new * C_new.unsqueeze(1)).t().contiguous(),
                                    args.lam, args.cg_iters, args.nystrom_m, device).cpu()
                traj_soft[f'r{r}'] = decoder_W(Wc)
            # arm 4: toward-clean shrinkage, counts fixed at C_prop
            shrink = {}
            for a in shrink_sweep:
                M_sh = (1 - a) * M_r + a * M0_clean
                shrink[str(a)] = decoder(M_sh, C_r)

            cond_res['budgets'][str(b)] = {
                'n_labels': int(len(anc_rand)),
                'gcP': gcP, 'gcA1_true_mean_prop_counts': gcA1,
                'gc_anc_only': gc_anc,
                'prop': {'acc': prop_acc, 'err_in_p3': err_in_p3, 'n_err': n_err},
                'pool_acc_w0': pool_acc_w0,
                'iter_hard': iter_hard, 'iter_soft': traj_soft,
                'shrink': shrink}
            print(f"  b{b}: gcP {gcP:+.2f} | A1 true-mean/prop-count {gcA1:+.2f} | "
                  f"anc-only {gc_anc:+.2f} | shrink " +
                  " ".join(f"a{k}:{v:+.2f}" for k, v in shrink.items()))
            print(f"      prop acc {prop_acc:.2f} | err in p3-pairs {err_in_p3:.2f} "
                  f"({n_err}) | pool_acc w0 {pool_acc_w0:.2f}")
            for tau in tau_sweep:
                tr = iter_hard[str(tau)]
                s = [f"r0:{tr['r0']['gc']:+.2f}(mE {tr['r0']['mean_err']:.2f}/p3 {tr['r0']['mean_err_p3']:.2f})"]
                for r in range(1, args.k_rounds + 1):
                    rec = tr[f'r{r}']
                    s.append(f"r{r}:{rec['gc']:+.2f}(prec {rec['gate_prec']:.2f} "
                             f"cov {rec['gate_cov']:.2f} mE {rec['mean_err']:.2f})")
                print(f"      iter hard tau{tau}: " + " ".join(s))
            print("      iter soft: " + " ".join(f"{k}:{v:+.2f}" for k, v in traj_soft.items()))

        results['conds'][cond] = cond_res
        del Ws, pool_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    W_mean_oracle gc {gc_mean_oracle:+.2f} | present {present}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
print("gcP = baseline. W_mean_oracle / A1 = the mean-decoder ceilings.")
print("iter_hard / iter_soft: does gc grow across rounds r0 -> rK?")
print("  grow toward A1/W_mean_oracle -> the loop is the method")
print("  plateau at gcP               -> single-pass is not the bottleneck")
print("gc_anc_only: anchors-only means at C_prop vs gcP -- is the")
print("  propagation assignment noise hurting the means?")
print("shrink: toward-CLEAN means (the untested prior form).")
print("DIAGNOSTICS (where the method is wrong):")
print("  prop.acc / err_in_p3: is the propagation assignment error in the SAME")
print("    P3 pairs (11-13/11-14) as the frozen errors, or elsewhere?")
print("  pool_acc w0 vs rK: is the refit actually better on the pool (foundation)?")
print("  gate_prec per round vs prop.acc: is confidence the right signal to carry")
print("    across iterations? (loop only denoises if gate_prec > prop.acc)")
print("  mean_err / mean_err_p3 per round: does the loop fix the boundary")
print("    classes or drift elsewhere?")

if __name__ == "__main__":
    main()
