"""al_bank_floor_diag.py: Iteration 2 -- how much supervision is actually needed
to reach a WORKING U?

The problem (from the roadmap): the only thing that yields oracle-quality U is
fitting W_sub on a labeled bank (C30/C31: 56+500 points). The efficient-bank
program claims fewer, better-chosen points reach the same U. We have never
measured the curve: at what bank size (and selection rule) does the bank-learned
U support a trust-region step that closes real gap?

This is the floor measurement that turns the bank from "a workaround we know works
at 500 points" into "a mechanism we know works at N points." It determines the
bank's design (point count, selection, what to store) before we build compression/
streaming machinery.

Setup (per condition, both extractors):
  W0    = frozen clean decoder
  W*    = oracle pool-fit decoder (the target)
  U_or  = top-r of R = W* - W0                    (oracle U, the goal)
  bank  = a labeled subset of the corrupted pool (size N, selection rule s)
  W_sub = ridge fit on the bank
  U_bk  = top-r of (W_sub - W0)                   (the bank-learned U)

Measures:
  align(U_bk, U_or)                     -- how close the bank U is to oracle
  trust-region gc with U_bk:
      G = U_bk^T X_lab^T (Y - X_lab W0) on b direction labels (leverage-in-U_bk)
      W1 = W0 + rho * U_bk * G/||G||, gc = mIoU(W1) - mIoU(W0)
  -- does the bank U support the same step the oracle U supports?

Selection rules (the "how chosen"):
  random          baseline
  per_class       balanced k/class (the C30/C31 rule)
  leverage_oracle top by ||x^T U_or||  (the UPPER BOUND on selection -- uses oracle;
                  if even this cannot reach the floor cheaply, no deployable rule can)
  margin_frozen   low |top2 margin| under W0 (boundary points; deployable)

Bank sizes: sweep {28, 56, 106, 156, 356, 556} (C30/C31 = 556; the efficient claim
is ~56-156 suffices).

Read: the floor is the smallest N where align > ~0.7 AND gc approaches the oracle-U
gc. If the floor is ~500, the bank is not cheap and the efficient claim fails. If
~56-156 with leverage_oracle, the efficient bank is viable (and leverage_oracle is
the ceiling for what a deployable selection could reach).

Usage:
  uv run python robust_diagnostic/al_bank_floor_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_bank_floor_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
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


def right_topk_svd(M, r):
    _, S, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    if not torch.isfinite(S).all():
        S = torch.where(torch.isfinite(S), S, torch.zeros_like(S))
        Vh = torch.where(torch.isfinite(Vh), Vh, torch.zeros_like(Vh))
    return Vh[:r].t().float(), S.float()


def subspace_cos(U_a, U_b, r):
    uh = U_a[:, :r]; uo = U_b[:, :r]
    S = torch.linalg.svdvals((uh.t() @ uo).double())
    return float(S.mean().item())


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
    ap.add_argument("--pool_size", type=int, default=50000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--max_clean", type=int, default=50000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--r_sweep", type=str, default="2,4")
    ap.add_argument("--bank_sizes", type=str, default="28,56,106,156,356,556")
    ap.add_argument("--b_direction", type=int, default=8, help="direction labels for the trust-region step")
    ap.add_argument("--rho_sweep", type=str, default="0.05,0.2,0.8")
    ap.add_argument("--per_class_k", type=int, default=8, help="k/class for the per_class selection")
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
    r_sweep = [int(x) for x in args.r_sweep.split(',')]
    bank_sizes = [int(x) for x in args.bank_sizes.split(',')]
    rho_sweep = [float(x) for x in args.rho_sweep.split(',')]
    rmax = max(r_sweep)

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'r_sweep': r_sweep,
               'bank_sizes': bank_sizes, 'conds': {}}

    # ---- W0: frozen clean decoder ----
    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
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
        pool, pl = fd[perm[:args.pool_size]], ld[perm[:args.pool_size]]
        val, vl = fd[perm[-args.val_size:]], ld[perm[-args.val_size:]]
        Xp = hdc_codes(pool, proj, device).float()
        Xv = hdc_codes(val, proj, device).float()
        del pool, val, fd, ld
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # oracle decoder + oracle U
        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_or, _ = right_topk_svd(R.t(), rmax)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # selection rule candidates. Most are index POOLS (subset by N); per_class
        # is built per N (k = ceil(N/n_classes) per class) so it scales to any N.
        n = len(Xp)
        # leverage_oracle: top by ||x^T U_or||
        lev_or = torch.norm(Xp.float() @ U_or[:, :2], p=2, dim=1)
        order_lever = torch.argsort(lev_or, descending=True)
        # margin_frozen: low |top2 softmax margin| under W0
        sm0 = torch.softmax(Xp.float() @ W0c, dim=1)
        top2 = torch.topk(sm0, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
        order_margin = torch.argsort(margin)                      # lowest margin first
        # random
        torch.manual_seed(7)
        order_random = torch.randperm(n)
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        n_classes = max(len(classes), 1)

        def per_class_order(N):
            k = max(1, int(np.ceil(N / n_classes)))
            order = []
            for c in classes:
                idx = (pl == c).nonzero().squeeze(1)
                if len(idx) >= k:
                    torch.manual_seed(7 + c)
                    order.append(idx[torch.randperm(len(idx))[:k]])
            order = torch.cat(order) if order else order_random[:N]
            if len(order) < N:
                rest = order_random[~torch.isin(order_random, order)][:N - len(order)]
                order = torch.cat([order, rest])
            return order

        selections_pool = {
            'random': order_random,
            'leverage_oracle': order_lever,
            'margin_frozen': order_margin,
        }

        # oracle-U trust-region reference (with the same direction-label budget)
        lev_d = torch.norm(Xp.float() @ U_or[:, :2], p=2, dim=1)
        sel_d = torch.argsort(lev_d, descending=True)[:args.b_direction].long()
        X_d = Xp[sel_d]; Y_d = onehot(pl[sel_d], NUM_CLASSES)
        resid_d = (Y_d.float() - X_d.float() @ W0c)

        cond_res = {'refs': refs, 'gap': float(gap), 'banks': {}}
        for N in bank_sizes:
            entry = {'selections': {}}
            sel_orders = dict(selections_pool)
            sel_orders['per_class'] = per_class_order(N)
            for sname, order in sel_orders.items():
                sel = order[:N].long()
                Xb = Xp[sel]; Yb = onehot(pl[sel], NUM_CLASSES)
                W_sub = ridge_fit_soft(Xb, Yb, args.lam, args.cg_iters, args.nystrom_m, device)
                U_bk, _ = right_topk_svd((W_sub - W0).detach().cpu().t(), rmax)
                s = {'align': {}, 'gc': {}}
                for r in r_sweep:
                    s['align'][str(r)] = subspace_cos(U_bk, U_or, r)
                    # trust-region step with U_bk and the direction labels
                    Ur = U_bk[:, :r]
                    G = (X_d.float() @ Ur).t() @ resid_d
                    Gn = G / (G.norm() + 1e-8)
                    gcs = {}
                    for rho in rho_sweep:
                        W1 = W0c + (Ur @ (rho * Gn))
                        d = mw(W1, Xv, vl) - refs['frozen']
                        gcs[str(rho)] = {'delta': float(d),
                                         'gap_closed': float(d / gap) if gap > 1e-9 else None}
                    s['gc'][str(r)] = gcs
                entry['selections'][sname] = s
            cond_res['banks'][str(N)] = entry

        # oracle-U gc reference (per rank, best over rho)
        oracle_ref = {}
        for r in r_sweep:
            Ur = U_or[:, :r]
            G = (X_d.float() @ Ur).t() @ resid_d
            Gn = G / (G.norm() + 1e-8)
            best = -9
            for rho in rho_sweep:
                W1 = W0c + (Ur @ (rho * Gn))
                d = mw(W1, Xv, vl) - refs['frozen']
                gc = d / gap if gap > 1e-9 else 0.0
                best = max(best, gc)
            oracle_ref[str(r)] = best
        cond_res['oracle_U_gc_best'] = oracle_ref

        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, U_or
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    oracle-U gc (best over rho): " + " ".join(f"r{r}:{v:+.2f}" for r, v in oracle_ref.items()))
        for N in bank_sizes:
            e = cond_res['banks'][str(N)]
            for sname, s in e['selections'].items():
                al = " ".join(f"r{r}:{v:.2f}" for r, v in s['align'].items())
                gc = " ".join(f"r{r}:" + str(max((v['gap_closed'] or -9) for v in s['gc'][str(r)].values()) if s['gc'][str(r)] else None)
                              for r in r_sweep)
                print(f"    N={N:4d} {sname:15s} align({al}) | best gc({gc})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("The floor: the smallest N where align > ~0.7 AND gc approaches oracle-U gc.")
    print("  leverage_oracle is the UPPER BOUND on selection (uses oracle); if even it")
    print("  cannot reach the floor cheaply, no deployable selection rule can.")
    print("  If the floor is ~500 -> bank is not cheap, the efficient claim fails.")
    print("  If ~56-156 -> the efficient bank is viable; the design is then:")
    print("    selection rule (which of random/per_class/margin_frozen approaches the")
    print("    leverage_oracle floor) + streaming sufficient statistics + compression.")


if __name__ == "__main__":
    main()
