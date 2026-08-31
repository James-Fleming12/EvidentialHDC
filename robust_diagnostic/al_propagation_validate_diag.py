"""al_propagation_validate_diag.py: validation of the propagated-mean result
(Iteration 10) -- DGLSS++ ONLY, the extractor with a real ceiling.

Iteration 10 showed nearest-anchor propagation of few labels produces a positive
mean decoder on dglsspp fog/crosstalk (+0.26 to +0.45 gc, the primary AL
target). Before calling it a method, this run validates it against the five
caveats flagged in the Iteration-10 verdict:

V1. TRUE ORACLE-COUNT ceiling (fix the Iteration-10 gcO bug which used the
    CLEAN counts C0 instead of the oracle counts C_star). Report:
    gc_prop_counts   propagated means x propagated counts (the honest method)
    gc_oracle_counts propagated means x ORACLE counts (how much of the +0.73/
                     +0.99 ceiling the propagated MEANS alone capture)
    gc_clean_counts  propagated means x clean counts (the old buggy arm)
    This splits the mean error from the count error.

V2. CLEAN-SOURCE BANK mean decoder (the memory-bank idea, label-free): aggregate
    the A4 clean-source labels (0.63-0.79 precision) into class means and feed
    the mean decoder. If this works, it is label-free entirely.

V3. INFLUENCE vs RANDOM anchor selection: the prior docs showed influence-ranked
    queries beat random (active_iterations Iteration 1). Test whether the
    propagated-mean result holds/improves with influence-selected anchors.

V4. PER-CLASS propagated-mean breakdown: which classes carry the +0.26-0.45
    (likely the tight majority; the loose {7,15,14} still fail -- the budget
    must target them).

V5. MEAN-vs-COUNT interaction: with the true oracle-count ceiling (V1) and the
    count error reported per budget, is the propagated-mean decoder limited by
    the means or by the counts? (The Iteration-10 result was positive with
    count_err 0.76-0.93 -- the means dominated; this confirms it cleanly.)

Decisive reads:
  V1 gc_oracle_counts ~ W_mean_oracle   -> the propagated MEANS capture the
     ceiling; the count error is the only remaining gap.
  V1 gc_oracle_counts << ceiling         -> the means are still the bottleneck.
  V2 (clean-source, label-free) ~ V1 gc -> the method needs NO labels at all.
  V3 influence > random at every budget  -> use influence selection going forward.
  V5 count_err small at high b + gc_oracle_counts high -> counts are fixable.

DGLSS++ ONLY (the extractor with a meaningful ceiling). Usage:
  uv run python robust_diagnostic/al_propagation_validate_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_propagation_validate_dglsspp.json
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


def nystrom_influence(Xd, lam, m, device):
    """I_i ~= ||(S + lI)^-1 x_i|| in the Nystrom subspace (same sketch as the
    warm start). The magnitude of point i's contribution to W."""
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    M = torch.linalg.inv(Shat)
    C = Xd @ P
    MC = C @ M
    return (MC.norm(dim=1) * (Xd.shape[1] ** 0.5)).cpu()


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
    ap.add_argument("--clean_bank", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=3, help="repeats for the random-selection variance")
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

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b_anchors': b_anchors,
               'clean_bank': args.clean_bank, 'reps': args.reps, 'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    lac = la[ci]
    W0 = ridge_fit_soft(Xc, onehot(lac, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    M0, C0 = class_means(Xc, lac, NUM_CLASSES)
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # clean-source bank (128-d features + labels) for V2
    torch.manual_seed(11)
    cb = torch.randperm(len(fa))[:args.clean_bank]
    clean_f = F.normalize(fa[cb].float(), p=2, dim=1)
    clean_l = la[cb]

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
        n = len(pf)

        # per-class index and influence
        class_idx = {c: torch.nonzero(pl == c).squeeze(1) for c in range(1, NUM_CLASSES)}
        I = nystrom_influence(Xp, args.lam, args.nystrom_m, device)

        cond_res = {'refs': refs, 'gap': float(gap),
                    'gc_mean_oracle': gc_mean_oracle,
                    'budgets': {}}

        # ---- V2. CLEAN-SOURCE BANK mean decoder (label-free, run once) ----
        nn_c = []
        for s in range(0, n, 5000):
            sim = pf[s:s+5000] @ clean_f.t()
            nn_c.append(sim.argmax(1))
        nn_c = torch.cat(nn_c)
        prop_clean = clean_l[nn_c]
        M_clean, C_clean = class_means(Xp, prop_clean, NUM_CLASSES)
        B_cc = (M_clean * C_clean.unsqueeze(1)).t().contiguous()
        W_cc = solve_whitened(Xp, B_cc, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        gc_clean_bank = gc(mw(W_cc, Xv, vl))
        B_cc_or = (M_clean * C_star.unsqueeze(1)).t().contiguous()
        W_cc_or = solve_whitened(Xp, B_cc_or, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
        gc_clean_bank_oracle_counts = gc(mw(W_cc_or, Xv, vl))
        clean_prec = float((prop_clean == pl).float().mean().item())

        for b in b_anchors:
            # repeated random selection (V3 variance) + influence selection
            gcs_rand = []
            gcs_rand_oracle = []
            gcs_infl = []
            gcs_infl_oracle = []
            for rep in range(args.reps):
                torch.manual_seed(7 + rep * 100)
                anc = []
                anc_infl = []
                for c in range(1, NUM_CLASSES):
                    idx = class_idx[c]
                    if len(idx) == 0:
                        continue
                    sub = idx[torch.randperm(len(idx))[:min(b, len(idx))]]
                    anc.append(sub)
                    # influence: top-b points of class c by influence
                    ii = torch.argsort(I[idx], descending=True)[:min(b, len(idx))]
                    anc_infl.append(idx[ii])
                anc = torch.cat(anc); anc_infl = torch.cat(anc_infl)
                anc_f = pf[anc]; anc_lab = pl[anc]
                anc_f_i = pf[anc_infl]; anc_lab_i = pl[anc_infl]
                # random-anchor propagation -> means -> decoder
                nn = (pf @ anc_f.t()).argmax(1)
                prop_lab = anc_lab[nn]
                M_p, C_p = class_means(Xp, prop_lab, NUM_CLASSES)
                B_pc = (M_p * C_p.unsqueeze(1)).t().contiguous()
                W_pc = solve_whitened(Xp, B_pc, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
                gcs_rand.append(gc(mw(W_pc, Xv, vl)))
                B_po = (M_p * C_star.unsqueeze(1)).t().contiguous()
                W_po = solve_whitened(Xp, B_po, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
                gcs_rand_oracle.append(gc(mw(W_po, Xv, vl)))
                # influence-anchor propagation
                nn_i = (pf @ anc_f_i.t()).argmax(1)
                prop_lab_i = anc_lab_i[nn_i]
                M_pi, C_pi = class_means(Xp, prop_lab_i, NUM_CLASSES)
                B_pic = (M_pi * C_pi.unsqueeze(1)).t().contiguous()
                W_pic = solve_whitened(Xp, B_pic, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
                gcs_infl.append(gc(mw(W_pic, Xv, vl)))
                B_pio = (M_pi * C_star.unsqueeze(1)).t().contiguous()
                W_pio = solve_whitened(Xp, B_pio, args.lam, args.cg_iters, args.nystrom_m, device).cpu()
                gcs_infl_oracle.append(gc(mw(W_pio, Xv, vl)))
            # per-class breakdown for this b (use rep 0 random)
            torch.manual_seed(7)
            anc = []
            for c in range(1, NUM_CLASSES):
                idx = class_idx[c]
                if len(idx) == 0:
                    continue
                anc.append(idx[torch.randperm(len(idx))[:min(b, len(idx))]])
            anc = torch.cat(anc)
            anc_f = pf[anc]; anc_lab = pl[anc]
            nn = (pf @ anc_f.t()).argmax(1)
            prop_lab = anc_lab[nn]
            per_class = {}
            for c in range(1, NUM_CLASSES):
                m = (pl == c)
                if int(m.sum().item()) < 50:
                    continue
                per_class[str(c)] = float((prop_lab[m] == pl[m]).float().mean().item())
            count_err = float((torch.bincount(prop_lab.long(), minlength=NUM_CLASSES)[1:].float() -
                               C_star[1:]).abs().sum().item() / (C_star[1:].sum().item() + 1e-12))

            cond_res['budgets'][str(b)] = {
                'rand_gc': gcs_rand, 'rand_gc_mean': sum(gcs_rand) / len(gcs_rand),
                'rand_gc_oracle_counts': gcs_rand_oracle,
                'rand_gc_oracle_mean': sum(gcs_rand_oracle) / len(gcs_rand_oracle),
                'infl_gc': gcs_infl, 'infl_gc_mean': sum(gcs_infl) / len(gcs_infl),
                'infl_gc_oracle_counts': gcs_infl_oracle,
                'infl_gc_oracle_mean': sum(gcs_infl_oracle) / len(gcs_infl_oracle),
                'per_class_prec': per_class, 'count_err': count_err,
                'n_labels': int(len(anc))}
        cond_res['V2_cleansource'] = {'gc_counts': gc_clean_bank,
                                      'gc_oracle_counts': gc_clean_bank_oracle_counts,
                                      'prec': clean_prec}
        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, M_star, pool_f, pf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    W_mean_oracle gc {gc_mean_oracle:+.2f}")
        print(f"    V2 clean-source bank: gc_counts {gc_clean_bank:+.2f} gc_oracle_counts "
              f"{gc_clean_bank_oracle_counts:+.2f} prec {clean_prec:.3f}")
        for b in b_anchors:
            v = cond_res['budgets'][str(b)]
            print(f"    b{b} (n={v['n_labels']}): rand gc {v['rand_gc_mean']:+.2f} "
                  f"oracle-cnt {v['rand_gc_oracle_mean']:+.2f} | infl gc {v['infl_gc_mean']:+.2f} "
                  f"oracle-cnt {v['infl_gc_oracle_mean']:+.2f} | cnt_err {v['count_err']:.2f}")
            # top-3 best per-class prec
            top = sorted(v['per_class_prec'].items(), key=lambda kv: -kv[1])[:3]
            bot = sorted(v['per_class_prec'].items(), key=lambda kv: kv[1])[:3]
            print(f"      per-class best {top} worst {bot}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("V1 (the count-split, now correct): rand_gc_oracle_mean vs rand_gc_mean")
    print("   vs W_mean_oracle. oracle-count ~ ceiling -> the MEANS capture the")
    print("   ceiling; count error is the only gap. << ceiling -> means bottleneck.")
    print("V2 clean-source (label-free): gc_counts ~ V1 -> the method needs NO labels.")
    print("V3 influence vs random: infl_gc_mean > rand_gc_mean -> use influence.")
    print("V4 per-class: which classes carry the gain (tight majority vs 7/15/14).")
    print("V5 count_err at high b + oracle-count gc high -> counts are fixable.")


if __name__ == "__main__":
    main()
