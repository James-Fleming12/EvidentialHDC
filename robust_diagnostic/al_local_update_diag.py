"""al_local_update_diag.py: Iteration 2 (Experiment B) -- can a local/conservative
correction be driven by the SAME few acquisition-selected labels, WITHOUT oracle U?

Iteration 1 showed active selection compensates for the lack of U: margin_tta_div
and egl recover +0.29-0.37 gc with a KNOWN (oracle) direction in the first-order
step. The question now is whether the UPDATE ITSELF can be driven by the same few
labels, using LOCAL correction forms that never need oracle U.

Update forms compared (all decode on the val set, all use only the b selected
labels):
  global_oracle   W1 = W0 + rho * U_or * G/||G|| (oracle-U first-order; the
                  Iteration-1 ceiling reference)
  class_bias      z' = z + b, b_c = mean_i[1[y_i=c] - softmax(W0 x_i)_c]
                  (per-class logit bias -- 2C scalars from the few labels)
  prototype       mu_c <- mu_c + (1/|L_c|) sum_{i in L_c, y_i=c}(x_i - mu_c);
                  decode by cosine to the UPDATED prototypes (only classes with
                  evidence move; others keep the clean prototype)
  class_pair      for each labeled point with frozen pred != true, accumulate
                  delta_ab = sum_i s_i x_i (s_i = margin error) for the (true,pred)
                  pair; W0[:,a] += eta*delta, W0[:,b] -= eta*delta (only the
                  implicated class-pair boundary moves)
  local_topK      class_pair restricted to the top-K most-implicated pairs (K=3)

Acquisition rules (the Iteration-1 winners + random baseline):
  random, margin_tta_div, egl        at b in {2,4,8}

Read: if a LOCAL form (class_bias / prototype / class_pair) driven by the same few
labels reaches meaningful gc on fog/crosstalk WITHOUT oracle U, the update itself
is drivable by few labels -- Experiment B succeeds. The healthy conditions
(snow/wet_ground) check the zero-degradation property (P3): a local update must
not hurt them.

Usage:
  uv run python robust_diagnostic/al_local_update_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_local_update_<label>.json
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


def decode_bias(W, b, codes, chunk=100000):
    W = W.detach().cpu()
    p = []
    for s in range(0, len(codes), chunk):
        p.append((codes[s:s+chunk].float() @ W + b).argmax(1))
    return torch.cat(p)

def decode_proto(protos, codes, chunk=100000):
    pn = F.normalize(protos.float(), p=2, dim=1)
    preds = []
    for s in range(0, len(codes), chunk):
        sim = F.normalize(codes[s:s+chunk].float(), p=2, dim=1) @ pn.t()
        preds.append(sim.argmax(1))
    return torch.cat(preds)


def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)


def mwb(b, Xv, vl, W0c):
    return compute_miou(decode_bias(W0c, b, Xv), vl)


def mwproto(protos, Xv, vl):
    return compute_miou(decode_proto(protos, Xv), vl)


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


def farthest_point(feats, cand_idx, b, device):
    cf = F.normalize(feats[cand_idx].float(), p=2, dim=1).to(device)
    torch.manual_seed(3)
    sel = [int(torch.randint(len(cand_idx), (1,)).item())]
    dist = (cf - cf[sel[0]]).norm(dim=1)
    for _ in range(b - 1):
        nxt = int(dist.argmax().item())
        sel.append(nxt)
        d2 = (cf - cf[nxt]).norm(dim=1)
        dist = torch.minimum(dist, d2)
    return cand_idx[torch.tensor(sel)]


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
    ap.add_argument("--r", type=int, default=2, help="rank for the oracle-U reference")
    ap.add_argument("--budgets", type=str, default="2,4,8")
    ap.add_argument("--rho", type=float, default=0.2, help="first-order step (oracle ref)")
    ap.add_argument("--eta", type=float, default=0.05, help="local update step size")
    ap.add_argument("--cand_frac", type=float, default=0.05)
    ap.add_argument("--tta_augs", type=int, default=5)
    ap.add_argument("--topk", type=int, default=3, help="class_pair top-K pairs")
    ap.add_argument("--rules", type=str, default="random,margin_tta_div,egl")
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    budgets = [int(x) for x in args.budgets.split(',')]
    rules = [x.strip() for x in args.rules.split(',') if x.strip()]
    r = args.r

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'r': r, 'budgets': budgets,
               'rules': rules, 'conds': {}}

    # ---- W0 + oracle U (only for the ceiling reference) ----
    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
    # clean prototypes (for the prototype update's reference)
    protos_clean = torch.zeros(NUM_CLASSES, 10000)
    for c in range(1, NUM_CLASSES):
        m = (la[ci] == c)
        if int(m.sum().item()) > 100:
            protos_clean[c] = Xc[m].mean(dim=0)
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
        R = (Ws - W0).detach().cpu().float()
        U_or, _ = right_topk_svd(R.t(), r)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # ---- label-free acquisition signals (Iteration-1 winners) ----
        sm = torch.softmax(Xp.float() @ W0c, dim=1)
        top2 = torch.topk(sm, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
        cand = torch.argsort(margin)[:max(int(args.cand_frac * len(Xp)), 8 * max(budgets))]
        n_cand = len(cand)
        cand_margin = margin[cand]
        # TTA instability on candidates (bit-flip augmentation)
        Xcand = Xp[cand].float()
        draws = []
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xcand) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xcand, Xcand) @ W0c, dim=1))
        tta_var = torch.stack(draws).var(dim=0).mean(dim=1)

        def select(rule, b):
            if rule == 'random':
                torch.manual_seed(7)
                return cand[torch.randperm(n_cand)[:b]]
            if rule == 'margin_tta_div':
                m = cand_margin / (cand_margin.max() + 1e-8)
                v = tta_var / (tta_var.max() + 1e-8)
                score = -m + v
                topM = torch.argsort(score, descending=True)[:8 * b]
                return farthest_point(pool_f, cand[topM], b, device)
            if rule == 'egl':
                cand_sm = sm[cand]
                cand_a = top2.indices[:, 0][cand]; cand_b = top2.indices[:, 1][cand]
                egl = torch.zeros(n_cand)
                for i in range(n_cand):
                    p = cand_sm[i]
                    a = int(cand_a[i].item()); b = int(cand_b[i].item())
                    ea = torch.zeros(NUM_CLASSES); ea[a] = 1.0
                    eb = torch.zeros(NUM_CLASSES); eb[b] = 1.0
                    egl[i] = p[a].item() * (ea - p).norm().item() + p[b].item() * (eb - p).norm().item()
                return cand[torch.argsort(egl, descending=True)[:b]]
            torch.manual_seed(7)
            return cand[torch.randperm(n_cand)[:b]]

        cond_res = {'refs': refs, 'gap': float(gap), 'rules': {}}
        for rule in rules:
            entry = {}
            for b in budgets:
                sel = select(rule, b).long()
                X_lab = Xp[sel]; Y_lab = onehot(pl[sel], NUM_CLASSES)
                y_lab = pl[sel]

                # ---- update forms ----
                res = {}
                # global_oracle (ceiling reference)
                resid = (Y_lab.float() - X_lab.float() @ W0c)
                G = (X_lab.float() @ U_or).t() @ resid
                Gn = G / (G.norm() + 1e-8)
                W1 = W0c + (U_or @ (args.rho * Gn))
                d = mw(W1, Xv, vl) - refs['frozen']
                res['global_oracle'] = {'delta': float(d),
                                        'gap_closed': float(d / gap) if gap > 1e-9 else None}
                # class_bias: b_c = mean_i[1[y=c] - softmax(W0 x)_c]
                p0 = torch.softmax(X_lab.float() @ W0c, dim=1)
                b_vec = (Y_lab.float() - p0).mean(dim=0)
                d_b = mwb(b_vec, Xv, vl, W0c) - refs['frozen']
                res['class_bias'] = {'delta': float(d_b),
                                     'gap_closed': float(d_b / gap) if gap > 1e-9 else None}
                # prototype: update only classes with evidence
                protos = protos_clean.clone()
                n_upd = 0
                for c in range(1, NUM_CLASSES):
                    m = (y_lab == c)
                    if int(m.sum().item()) > 0:
                        protos[c] = protos_clean[c] + args.eta * (X_lab[m].float().mean(dim=0) - protos_clean[c])
                        n_upd += 1
                d_p = mwproto(protos, Xv, vl) - refs['frozen']
                res['prototype'] = {'delta': float(d_p),
                                    'gap_closed': float(d_p / gap) if gap > 1e-9 else None,
                                    'n_classes_updated': n_upd}
                # class_pair: move only the (true,pred) boundaries
                pred0 = sm[sel].argmax(dim=1)
                W_cp = W0c.clone()
                pairs_acc = {}
                for i in range(len(sel)):
                    a = int(y_lab[i].item()); bp = int(pred0[i].item())
                    if a == bp or a == 0 or bp == 0:
                        continue
                    s_i = sm[sel][i, bp].item() - sm[sel][i, a].item()
                    pairs_acc.setdefault((a, bp), 0.0)
                    pairs_acc[(a, bp)] += 1
                    W_cp[:, a] += args.eta * s_i * X_lab[i].float()
                    W_cp[:, bp] -= args.eta * s_i * X_lab[i].float()
                d_cp = mw(W_cp, Xv, vl) - refs['frozen']
                res['class_pair'] = {'delta': float(d_cp),
                                     'gap_closed': float(d_cp / gap) if gap > 1e-9 else None,
                                     'n_pairs': len(pairs_acc)}
                # local_topK: class_pair restricted to top-K most-implicated pairs
                W_tk = W0c.clone()
                if pairs_acc:
                    top = sorted(pairs_acc.items(), key=lambda kv: -kv[1])[:args.topk]
                    top_keys = set(k for k, _ in top)
                    for i in range(len(sel)):
                        a = int(y_lab[i].item()); bp = int(pred0[i].item())
                        if (a, bp) not in top_keys or a == 0 or bp == 0:
                            continue
                        s_i = sm[sel][i, bp].item() - sm[sel][i, a].item()
                        W_tk[:, a] += args.eta * s_i * X_lab[i].float()
                        W_tk[:, bp] -= args.eta * s_i * X_lab[i].float()
                d_tk = mw(W_tk, Xv, vl) - refs['frozen']
                res['local_topK'] = {'delta': float(d_tk),
                                     'gap_closed': float(d_tk / gap) if gap > 1e-9 else None}
                entry[str(b)] = res
            cond_res['rules'][rule] = entry

        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, U_or, pool_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        for rule in rules:
            for b in budgets:
                res = cond_res['rules'][rule][str(b)]
                line = " ".join(f"{k}:{v['gap_closed']:+.2f}" for k, v in res.items() if 'gap_closed' in v)
                print(f"    {rule:13s} b{b}: {line}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Can a LOCAL form (class_bias / prototype / class_pair / local_topK) driven")
    print("by the SAME few labels reach meaningful gc WITHOUT oracle U?")
    print("  global_oracle = the Iteration-1 ceiling reference (uses oracle U).")
    print("  class_bias    = 2C logit scalars; prototype = updated class means;")
    print("  class_pair    = only the (true,pred) boundaries move; local_topK = top-K pairs.")
    print("The healthy conditions (snow/wet_ground) check zero-degradation (P3):")
    print("a local update must NOT hurt them.")


if __name__ == "__main__":
    main()
