"""al_rule_budget_diag.py: the two AL tests on the 21ep ball/spec spaces +
premise diagnostics.

  TEST 1 (query rules): does the confidence rule match influence on the new
  space? C15 found lev_conf_spearman is POSITIVE (+0.09..+0.38) on the ball
  space (negative on the base), so high-confidence points ARE high-leverage
  there. Four rules select k points per class (k=8, oracle counts + V3 control
  variate): influence (the Iteration-1 winner, nystrom_influence), confidence
  (the free rule), random (the Iteration-8 best mean estimator), centroid-near
  (the representative of Iteration 0). Compare the 10-COMB AL delta AND the
  selection-quality diagnostics (mean cos to true mean, per rule): if the
  confidence rule's mean quality is as good as influence's, the free rule is
  validated; if a rule's mean is biased but another is not, the failure is
  selection (fixable), not the premise.

  TEST 2 (k budget): does k=2 (32 labels) hold the k=8 (64-72) result? C15's
  mean-k curve saturates at k=2 (cos 0.93-0.95). Sweep k in {2,4,8} x
  rho in {0.25,0.5,0.75} (control variate weight) with source counts and the
  best rule from Test 1. If the k=2 delta holds, the label cost halves.

  PREMISE DIAGNOSTICS (the "does the feature space still have potential"
  guardrail -- so a negative AL result is attributable, not just failed):
    - frozen / oracle / spectral ceiling: the closeable gap (oracle - frozen).
    - whitened T error at the best config (the Iteration-8 smoking gun).
    - the t_cos -> w_cos -> mIoU chain at the best config (Iteration-7).
    - per-rule mean quality: cos(selected-k mean, true class mean), averaged
      over classes -- selection bias vs the premise, measured separately.
    - lev_conf_spearman re-measured (the C15 signal that motivated Test 1).

  If the AL delta is negative EVERYWHERE but (a) the oracle gap is large and
  (b) the whitened error is small at the best config and (c) mean quality is
  high for all rules -> the failure is in the T_hat mass (counts x mean) or
  the residual anchor, i.e. a DESIGN issue, not the premise. If the whitened
  error is large or the oracle gap is small -> the premise (feature space)
  itself is the limit.

Usage:
  uv run python robust_diagnostic/al_rule_budget_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_rule_budget_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract_features(model, parser, device, num_frames=100):
    feats, lbls = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            out = model(in_vol)
            z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(zf.cpu())
            lbls.append(labels[mask].cpu())
    return torch.cat(feats), torch.cat(lbls)


def hdc_codes(feats, proj, device, chunk=100000):
    out = []
    for s in range(0, len(feats), chunk):
        out.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(out)


def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc)
    y[torch.arange(len(lbls)), lbls.long()] = 1
    return y


def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    p = []
    for s in range(0, len(codes), chunk):
        p.append((codes[s:s + chunk].float() @ W).argmax(1))
    return torch.cat(p)


def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)


def cos_sim(a, b):
    a = a.detach().cpu().float().reshape(-1)
    b = b.detach().cpu().float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def ridge_fit_soft(X, Y, lam, iters, m, device):
    X = X.to(device)
    torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1])
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P
    Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    That = XP.t() @ Yd
    x = P @ torch.linalg.solve(Shat, That)
    b = X.t() @ Yd

    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x)
    p = r.clone()
    rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p)
        a = rs / ((p * Ap).sum(0) + 1e-30)
        x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap
        rsn = (r * r).sum(0)
        be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p
        rs = rsn
    return x.float()


def nystrom_influence(Xd, lam, m, device):
    """I_i ~= ||(S + lI)^-1 x_i|| in the Nystrom subspace (same sketch as the
    warm start): M = (S_hat + lI)^-1 (m x m), c_i = P^T x_i,
    I_i = sqrt(d) * ||M c_i||. The magnitude of point i's contribution to W."""
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
    sync()
    return time.time()


def toc(t0):
    sync()
    return time.time() - t0


def spearman(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 10 else None


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
    ap.add_argument("--max_clean", type=int, default=200000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--ks", type=str, default="2,4,8")
    ap.add_argument("--rhos", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--betas", type=str, default="0.6,0.75")
    ap.add_argument("--etas", type=str, default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="med")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config))
    ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    ks = [int(x) for x in args.ks.split(',')]
    rhos = [float(x) for x in args.rhos.split(',')]
    betas = [float(x) for x in args.betas.split(',')]
    etas = [float(x) for x in args.etas.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)

    results = {'label': args.label, 'method': args.method_b, 'ks': ks, 'rhos': rhos,
               'betas': betas, 'etas': etas, 'conds': {}}

    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]

        proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
        Xc = hdc_codes(fa[ci], proj, device).float()
        Xp = hdc_codes(pool, proj, device).float()
        Xv = hdc_codes(val, proj, device).float()
        Xd = Xp.to(device)
        N = Xp.shape[0]

        W_clean = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam,
                                 args.cg_iters, args.nystrom_m, device)
        W_oracle = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam,
                                  args.cg_iters, args.nystrom_m, device)

        r = {'refs': {}, 'premise': {}, 'test1': {}, 'test2': {}}
        r['refs']['frozen'] = mw(W_clean, Xv, vl)
        r['refs']['oracle'] = mw(W_oracle, Xv, vl)

        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}

        # spectral-exact ceiling
        S = (Xd.t() @ Xd).double() / N
        eigS, U = torch.linalg.eigh(S)
        eigS = eigS.float()
        U = U.float()
        lam_hat = args.lam / N
        sig = (eigS + lam_hat).clamp(min=lam_hat)
        T_or = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            T_or[:, c] = Xp[cls_idx[c]].sum(0)
        m0 = pl == 0
        if int(m0.sum().item()) > 0:
            T_or[:, 0] = Xp[m0].sum(0)
        T_or = T_or / N
        Uc = U.to(device)
        sig_d = sig.to(device)
        UtT_or = Uc.t() @ T_or.to(device)
        W_or_spec = (Uc @ ((1.0 / sig_d).unsqueeze(1) * UtT_or)).cpu().float()
        r['refs']['oracle_spec'] = mw(W_or_spec, Xv, vl)

        # ---- premise diagnostics ----
        prem = r['premise']
        prem['closeable_gap'] = r['refs']['oracle'] - r['refs']['frozen']
        prem['closeable_gap_spec'] = r['refs']['oracle_spec'] - r['refs']['frozen']

        # per-point signals: confidence (frozen probe), influence, residual
        Wf_d = W_clean.to(device)
        logits = Xd.float() @ Wf_d
        conf = logits.softmax(1).max(1).values.cpu()
        I = nystrom_influence(Xd, args.lam, args.nystrom_m, device)
        probs = logits.softmax(1).cpu()
        resid = (onehot(pl, NUM_CLASSES).float() - probs).norm(1, dim=1)
        prem['lev_conf_spearman'] = spearman(I.numpy(), conf.numpy())
        prem['resid_conf_spearman'] = spearman(resid.numpy(), conf.numpy())
        prem['conf_influence_spearman'] = spearman(conf.numpy(), I.numpy())

        # ---- helpers ----
        # clean code means (control variate) + clean freq prior (source counts)
        clean_classes = sorted(set(la[ci].tolist()) & set(range(1, NUM_CLASSES)))
        clean_idx = {c: (la[ci] == c).nonzero().squeeze(1) for c in clean_classes}
        clean_code_means = {}
        for c in clean_classes:
            idx = clean_idx[c]
            if len(idx) > 0:
                clean_code_means[c] = Xc[idx].mean(0)
        tot_clean = len(la[ci])
        clean_freq = {c: int((la[ci] == c).sum().item()) / tot_clean for c in classes}
        source_counts = {c: int(clean_freq.get(c, 0) * N) for c in classes}
        oracle_counts = {c: len(cls_idx[c]) for c in classes}

        # true class means in code space (for mean-quality diagnostics)
        true_means = {c: Xp[cls_idx[c]].mean(0) for c in classes}

        # per-class selection signals (indices within the class)
        def class_signals(c):
            idx = cls_idx[c]
            # influence within class
            I_c = I[idx]
            # confidence within class
            C_c = conf[idx]
            # distance to class centroid (128-d raw features, normalized)
            zn = F.normalize(pool.float(), dim=1)
            mu_c = F.normalize(zn[idx].mean(0).unsqueeze(0), dim=1)[0]
            d_c = 1 - (zn[idx] @ mu_c)
            return I_c, C_c, d_c

        def select_k(c, k, rule, seed=2):
            idx = cls_idx[c]
            I_c, C_c, d_c = class_signals(c)
            if rule == 'influence':
                sel = torch.argsort(I_c, descending=True)[:k]
            elif rule == 'confidence':
                sel = torch.argsort(C_c, descending=True)[:k]
            elif rule == 'random':
                torch.manual_seed(seed)
                sel = torch.randperm(len(idx))[:k]
            elif rule == 'centroid':
                sel = torch.argsort(d_c)[:k]
            return idx[sel]

        def make_T_counts(counts, thresh):
            T = torch.zeros(10000, NUM_CLASSES)
            for c in classes:
                if len(cls_idx[c]) < thresh:
                    continue
                T[:, c] = counts[c]
            return T / N

        def build_T(c, k, rule, counts, rho, thresh):
            # returns the class-c column of T_hat (code space) with the mean
            # from rule-selected k points, control-variate rho toward clean mean
            idx = cls_idx[c]
            if len(idx) < thresh:
                return None
            sel = select_k(c, k, rule)
            mu = Xp[sel].mean(0)
            if rho is not None and c in clean_code_means:
                mu = rho * clean_code_means[c] + (1 - rho) * mu
            return counts[c] * mu / N

        def best_combo(T_full, refs_frozen):
            UtT = Uc.t() @ T_full.to(device)
            Wf = W_clean.detach().cpu()
            best = None
            for beta in betas:
                Wb = (Uc @ (sig_d.pow(-beta).unsqueeze(1) * UtT)).cpu().float()
                for eta in etas:
                    W = Wf + eta * (Wb - Wf)
                    d = mw(W, Xv, vl) - refs_frozen
                    if best is None or d > best['delta']:
                        best = {'beta': beta, 'eta': eta, 'delta': d,
                                'miou': mw(W, Xv, vl)}
            return best

        def T_full_from(c, k, rule, counts, rho, thresh):
            T = make_T_counts(counts, thresh)
            col = build_T(c, k, rule, counts, rho, thresh)
            if col is not None:
                T[:, c] = col
            return T

        # ---- TEST 1: query rules at k=8, oracle counts, V3 (rho=0.5) ----
        t1 = r['test1']
        t1['k'] = 8
        t1['counts'] = 'oracle'
        t1['rho'] = 0.5
        t1['rules'] = {}
        for rule in ['influence', 'confidence', 'random', 'centroid']:
            Tf = make_T_counts(oracle_counts, 50)
            mean_cos = []
            for c in classes:
                col = build_T(c, 8, rule, oracle_counts, 0.5, 50)
                if col is None:
                    continue
                Tf[:, c] = col
                # mean quality: cos(selected mean, true mean) in code space
                sel = select_k(c, 8, rule)
                mu_s = Xp[sel].mean(0)
                mu_t = true_means[c]
                mean_cos.append(cos_sim(mu_s, mu_t))
            best = best_combo(Tf, r['refs']['frozen'])
            t1['rules'][rule] = {'best': best,
                                 'mean_cos': float(np.mean(mean_cos)) if mean_cos else None,
                                 'mean_cos_per_class': {str(c): float(v) for c, v in zip(classes, mean_cos)}}

        # ---- TEST 2: k sweep x rho sweep, best rule from Test 1, source counts ----
        best_rule = max(t1['rules'], key=lambda x: t1['rules'][x]['best']['delta'])
        t2 = r['test2']
        t2['rule'] = best_rule
        t2['counts'] = 'source'
        t2['grid'] = {}
        for k in ks:
            t2['grid'][str(k)] = {}
            for rho in rhos:
                Tf = make_T_counts(source_counts, k)
                for c in classes:
                    col = build_T(c, k, best_rule, source_counts, rho, k)
                    if col is not None:
                        Tf[:, c] = col
                best = best_combo(Tf, r['refs']['frozen'])
                t2['grid'][str(k)][str(rho)] = best

        # ---- the premise verdict at the best config ----
        best_k = max(ks, key=lambda k: max((t2['grid'][str(k)][str(r)]['delta']
                                            for r in rhos)))
        best_rho = max(rhos, key=lambda r: t2['grid'][str(best_k)][str(r)]['delta'])
        cfg = t2['grid'][str(best_k)][str(best_rho)]
        prem['best_config'] = {'k': best_k, 'rho': best_rho, 'rule': best_rule, **cfg}
        # whitened error at the best config: ||(S+lI)^-1 (T_hat - T_or)|| / ||(S+lI)^-1 T_or||
        Tf = make_T_counts(source_counts, best_k)
        for c in classes:
            col = build_T(c, best_k, best_rule, source_counts, best_rho, best_k)
            if col is not None:
                Tf[:, c] = col
        UtT_hat = Uc.t() @ Tf.to(device)
        T_hat_spec = (Uc @ (UtT_hat)).cpu().float()
        # use the spectral solve for both terms (exact)
        W_hat_spec = (Uc @ ((1.0 / sig_d).unsqueeze(1) * UtT_hat)).cpu().float()
        prem['w_cos'] = cos_sim(W_hat_spec, W_or_spec)
        prem['t_cos'] = cos_sim(Tf, T_or)
        # whitened error
        dT = Tf - T_or
        UtT_d = Uc.t() @ dT.to(device)
        dT_w = (Uc @ ((1.0 / sig_d).unsqueeze(1) * UtT_d)).cpu().float()
        T_w = (Uc @ ((1.0 / sig_d).unsqueeze(1) * UtT_or)).cpu().float()
        prem['whitened_error'] = float(dT_w.norm() / (T_w.norm() + 1e-30))

        results['conds'][cond] = r
        del Xc, Xp, Xv, Xd, S, eigS, U, Uc, sig, sig_d, UtT_or, W_or_spec, W_clean, W_oracle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} "
              f"/ spec-ceil {r['refs']['oracle_spec']:.3f} (closeable gap {prem['closeable_gap']:+.3f})")
        print(f"  premise: lev_conf {prem['lev_conf_spearman']:.3f} resid_conf "
              f"{prem['resid_conf_spearman']:.3f}")
        print(f"  TEST1 k=8 (oracle cnt, rho=0.5): " + " | ".join(
            f"{rule}: {v['best']['delta']:+.3f} (mean_cos {v['mean_cos']:.3f})"
            for rule, v in t1['rules'].items()))
        print(f"  TEST2 rule={best_rule} (source cnt):")
        for k in ks:
            row = " ".join(f"r={rho}:{t2['grid'][str(k)][str(rho)]['delta']:+.3f}" for rho in rhos)
            print(f"    k={k}: {row}")
        print(f"  premise best: k={best_k} rho={best_rho} -> {cfg['delta']:+.3f} "
              f"(t_cos {prem['t_cos']:.3f} w_cos {prem['w_cos']:.3f} whit_err {prem['whitened_error']:.2f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("TEST1: does confidence match influence (free rule)? If yes + mean_cos")
    print("similar, the confidence rule is validated on this space.")
    print("TEST2: does k=2 hold k=8? If yes, budget halves to ~32 labels.")
    print("PREMISE: if delta negative everywhere but closeable_gap large and")
    print("whitened_error small at best config and mean_cos high -> design issue")
    print("(fixable). If whitened_error large or gap small -> the space is the limit.")


if __name__ == "__main__":
    main()
