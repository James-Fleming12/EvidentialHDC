"""al_error_loop_diag.py: A5 -- the ERROR-CORRECTION AL LOOP (the recommended
next test in new_iters.md, Level 2 / local decision correction).

The pivot (Iterations 3b + 4) closed every W-update family: the labels cannot
reconstruct R, and neither can the pool. The remaining use of labels is to
DECIDE where the frozen classifier is wrong, and to repair those decisions at
DECODE time -- NOT to re-estimate W.

This diagnostic tests the two mechanisms behind A5, separately:

A. PAIR DISCOVERY -- can a few labels identify the recurring (pred,true) error
   pairs? Reported as confusion precision/recall vs the val-truth error pairs.
   This is the "labels reveal the problem" claim.

B. DECODE-TIME RE-RANKING -- given the identified pairs, can a correction
   applied to the frozen probe's LOGITS (not W) close the gap?
   - pair_bias: per-pair logit offset (z_a -= d, z_b += d for the (a,b) pair).
     This is B3 (logit calibration) generalized to pairs: a decision-rule
     correction that holds for the linear classifier (bias on logits, not U).
   - pair_gate: margin-threshold flip (reassign a->b when the top-2 margin is
     below the label-observed misclassification threshold).
   Ablations: oracle pairs / oracle threshold (ceiling) vs label pairs / label
   threshold (the actual method) vs random pairs (control). The sequential
   acquisition is 2-stage: stage-1 labels (margin_tta_div) discover the pairs,
   stage-2 labels are queried ON those pairs' boundary points (the loop: labels
   reveal the next query).

Decisive reads:
   oracle-pair gc >> 0  -> decode-time pair re-ranking is a real mechanism.
   label-pair gc ~ oracle-pair gc -> labels find the pairs AND the loop works.
   label-pair gc << oracle-pair gc -> the bottleneck is pair discovery.
   random-pair gc ~ label-pair gc -> the correction is just noise.

Usage:
  uv run python robust_diagnostic/al_error_loop_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_error_loop_<label>.json
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
MAX_PAIRS = 4


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


def miou_from_pred(pred, vl):
    return compute_miou(pred, vl)


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


def discover_pairs(pred_lab, true_lab, max_pairs=MAX_PAIRS):
    """Top (a,b) error pairs by confusion count over labeled points."""
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    for p, t in zip(pred_lab.tolist(), true_lab.tolist()):
        if p != t:
            conf[p, t] += 1
    pairs = []
    idx = torch.nonzero(conf > 0)
    order = torch.argsort(conf[idx[:, 0], idx[:, 1]], descending=True)
    for k in order[:max_pairs]:
        a = int(idx[k, 0].item()); b = int(idx[k, 1].item())
        pairs.append((a, b, int(conf[a, b].item())))
    return pairs


def top_pairs_from_confusion(confusion_counts):
    """pairs (a,b) from a dict {(a,b): count} sorted desc."""
    items = sorted(confusion_counts.items(), key=lambda kv: kv[1], reverse=True)
    return [(a, b, c) for (a, b), c in items[:MAX_PAIRS]]


def find_bias_offset(z_ab, y_ab, a, b, d_sweep):
    """Best logit offset d applied to (a,b) over the labeled points of that pair.
    y_ab = 1 if the true label is b (the pair's second class). We seek d so that
    after shifting (logit_a -= d, logit_b += d), 'b wins' matches y_ab."""
    best_d, best_acc = 0.0, -1.0
    for d in d_sweep:
        zc = z_ab.clone()
        zc[:, 0] -= d; zc[:, 1] += d
        pred_b = (zc[:, 0] < zc[:, 1]).long()
        acc = (pred_b == y_ab).float().mean().item()
        if acc > best_acc:
            best_acc, best_d = acc, d
    return best_d, best_acc


def find_gate_threshold(margins_a, is_err):
    """Max margin at which the probe misclassifies (a->b). Below this, flip."""
    err_m = margins_a[is_err]
    if len(err_m) == 0:
        return None
    return float(err_m.max().item())


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
    ap.add_argument("--budgets", type=str, default="2,4,8")
    ap.add_argument("--cand_frac", type=float, default=0.05)
    ap.add_argument("--tta_augs", type=int, default=5)
    ap.add_argument("--d_sweep", type=str, default="0,0.25,0.5,1,2,4")
    ap.add_argument("--max_pairs", type=int, default=MAX_PAIRS)
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
    budgets = [int(x) for x in args.budgets.split(',')]
    bmax = max(budgets)
    d_sweep = [float(x) for x in args.d_sweep.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'budgets': budgets,
               'd_sweep': d_sweep, 'max_pairs': args.max_pairs, 'conds': {}}

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

        # ---- frozen logits / top-2 / margins on pool and val ----
        Lp = Xp.float() @ W0c
        Lv = Xv.float() @ W0c
        pred_p = Lp.argmax(1); pred_v = Lv.argmax(1)
        top2p = torch.topk(Lp, 2, dim=1)
        top2v = torch.topk(Lv, 2, dim=1)
        margin_p = (top2p.values[:, 0] - top2p.values[:, 1]).abs()
        a_p = top2p.indices[:, 0]; b_p = top2p.indices[:, 1]
        margin_v = (top2v.values[:, 0] - top2v.values[:, 1]).abs()
        a_v = top2v.indices[:, 0]; b_v = top2v.indices[:, 1]

        # ---- val-truth error pairs (the ground truth the labels must find) ----
        err_v = (pred_v != vl)
        true_conf = {}
        for a, b, e in zip(a_v[err_v].tolist(), b_v[err_v].tolist(), err_v.tolist()):
            if e:
                key = (a, b)
                true_conf[key] = true_conf.get(key, 0) + 1
        true_pairs = top_pairs_from_confusion(true_conf)

        # ---- stage-1 acquisition: margin_tta_div (Iteration-1 winner) ----
        n_cand = max(int(args.cand_frac * len(Xp)), 8 * bmax)
        cand = torch.argsort(margin_p)[:n_cand]
        cand_margin = margin_p[cand]
        Xcand = Xp[cand].float()
        draws = []
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xcand) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xcand, Xcand) @ W0c, dim=1))
        tta_var = torch.stack(draws).var(dim=0).mean(dim=1)
        m = cand_margin / (cand_margin.max() + 1e-8)
        v = tta_var / (tta_var.max() + 1e-8)
        topM = torch.argsort(-m + v, descending=True)[:8 * bmax]
        stage1 = farthest_point(pool_f, cand[topM], min(bmax, max(budgets)), device).long()

        cond_res = {'refs': refs, 'gap': float(gap),
                    'true_pairs': [(a, b, c) for a, b, c in true_pairs],
                    'budgets': {}}
        for b in budgets:
            b1 = max(1, b // 2)                     # stage-1 discovery labels
            b2 = b - b1                             # stage-2 pair-focused labels
            s1 = stage1[:b1]
            # pair discovery from stage-1 labels
            pred1 = pred_p[s1]; true1 = pl[s1]
            p1_pairs = discover_pairs(pred1, true1, args.max_pairs)
            # stage-2: query ON the discovered pairs' boundary (low margin, top2 in pairs)
            if b2 > 0 and len(p1_pairs) > 0:
                pair_keys = {(a, b) for a, b, _ in p1_pairs}
                in_pair = torch.tensor([(int(a_p[i]), int(b_p[i])) in pair_keys
                                        for i in range(len(Xp))], dtype=torch.bool)
                cand2 = torch.nonzero(in_pair).squeeze(1)
                if len(cand2) > 0:
                    m2 = margin_p[cand2]
                    s2 = cand2[torch.argsort(m2)[:b2]]
                else:
                    s2 = torch.argsort(margin_p)[:b2]
            else:
                s2 = torch.arange(0).long()
            sel = torch.cat([s1, s2]).long()
            y_lab = pl[sel]
            pred_lab = pred_p[sel]

            # ALL labels: final pair set (stage-1 + stage-2 confusion)
            final_pairs = discover_pairs(pred_lab, y_lab, args.max_pairs)

            # ---- decode-time re-ranking over the VAL set ----
            entry = {'n_labels': int(len(sel)),
                     'stage1_n': int(b1), 'stage2_n': int(b2),
                     'discovered_pairs': [(a, b, c) for a, b, c in p1_pairs],
                     'final_pairs': [(a, b, c) for a, b, c in final_pairs]}
            # gc helper
            def gc(mi):
                return (mi - refs['frozen']) / gap if gap > 1e-9 else None

            # --- arm: label-pair bias (the full A5 method) ---
            Lv_c = Lv.clone()
            last_d = 0.0
            for (a, b, _) in final_pairs:
                sel_ab = torch.nonzero((pred_lab == a) | (pred_lab == b)).squeeze(1)
                if len(sel_ab) == 0:
                    continue
                z_ab = Lp[sel_ab][:, [a, b]]
                y_ab = (y_lab[sel_ab] == b).long()
                d_best, _ = find_bias_offset(z_ab, y_ab, a, b, d_sweep)
                last_d = d_best
                Lv_c[:, a] -= d_best
                Lv_c[:, b] += d_best
            entry['pair_bias_label'] = {'gc': gc(miou_from_pred(Lv_c.argmax(1), vl)),
                                        'd': last_d}

            # --- arm: oracle-pair bias (ceiling: the RIGHT pairs, label offsets) ---
            Lv_o = Lv.clone()
            for (a, b, _) in true_pairs:
                sel_ab = torch.nonzero((pred_p == a) | (pred_p == b)).squeeze(1)
                if len(sel_ab) == 0:
                    continue
                z_ab = Lp[sel_ab][:, [a, b]]
                y_ab = (pl[sel_ab] == b).long()
                d_best, _ = find_bias_offset(z_ab, y_ab, a, b, d_sweep)
                Lv_o[:, a] -= d_best
                Lv_o[:, b] += d_best
            entry['pair_bias_oracle_pairs'] = {'gc': gc(miou_from_pred(Lv_o.argmax(1), vl))}

            # --- arm: random-pair bias (control: is the correction just noise?) ---
            Lv_r = Lv.clone()
            torch.manual_seed(11)
            n_pairs_r = len(true_pairs)
            for k in range(n_pairs_r):
                a = int(torch.randint(NUM_CLASSES, (1,)).item())
                b = int(torch.randint(NUM_CLASSES, (1,)).item())
                if a == b:
                    b = (b + 1) % NUM_CLASSES
                sel_ab = torch.nonzero((pred_p == a) | (pred_p == b)).squeeze(1)
                if len(sel_ab) == 0:
                    continue
                z_ab = Lp[sel_ab][:, [a, b]]
                y_ab = (pl[sel_ab] == b).long()
                d_best, _ = find_bias_offset(z_ab, y_ab, a, b, d_sweep)
                Lv_r[:, a] -= d_best
                Lv_r[:, b] += d_best
            entry['pair_bias_random'] = {'gc': gc(miou_from_pred(Lv_r.argmax(1), vl))}

            # --- arm: label-pair gate (margin-threshold flip) ---
            # tau comes from the LABELED points (label-gate arm), and the flip is
            # applied to the val set via the same pair/margin rule.
            Lv_g = Lv.clone()
            lab_margin = margin_p[sel]
            lab_pred = pred_p[sel]
            for (a, b, _) in final_pairs:
                is_a_lab = (lab_pred == a)                       # probe says a (labeled)
                if int(is_a_lab.sum().item()) == 0:
                    continue
                m_a = lab_margin[is_a_lab]
                is_err_a = (y_lab[is_a_lab] != a) | (y_lab[is_a_lab] == b)
                tau = find_gate_threshold(m_a, is_err_a)
                if tau is None:
                    continue
                is_a_pred = (pred_v == a) & (b_v == b)
                flip = is_a_pred & (margin_v <= tau)
                Lv_g[flip, a] = Lv_g[flip, b] - 1.0
            entry['pair_gate_label'] = {'gc': gc(miou_from_pred(Lv_g.argmax(1), vl))}

            # --- arm: oracle-pair gate (ceiling: true pairs, POOL-label thresholds) ---
            Lv_go = Lv.clone()
            for (a, b, _) in true_pairs:
                is_a_pool = (pred_p == a)
                if int(is_a_pool.sum().item()) == 0:
                    continue
                m_a = margin_p[is_a_pool]
                is_err_a = (pl[is_a_pool] != a) | (pl[is_a_pool] == b)
                tau = find_gate_threshold(m_a, is_err_a)
                if tau is None:
                    continue
                is_a_pred = (pred_v == a) & (b_v == b)
                flip = is_a_pred & (margin_v <= tau)
                Lv_go[flip, a] = Lv_go[flip, b] - 1.0
            entry['pair_gate_oracle_pairs'] = {'gc': gc(miou_from_pred(Lv_go.argmax(1), vl))}

            # --- pair-discovery quality vs val truth ---
            disc_keys = {(a, b) for a, b, _ in p1_pairs}
            true_keys = {(a, b) for a, b, _ in true_pairs}
            hit = len(disc_keys & true_keys)
            entry['discovery'] = {'pairs_found': len(disc_keys),
                                  'pairs_true': len(true_keys),
                                  'hit': hit,
                                  'precision': hit / len(disc_keys) if disc_keys else None,
                                  'recall': hit / len(true_keys) if true_keys else None}

            cond_res['budgets'][str(b)] = entry

        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, pool_f, Lp, Lv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    val-truth error pairs: {[(a,b,c) for a,b,c in true_pairs]}")
        for b in budgets:
            e = cond_res['budgets'][str(b)]
            d = e['discovery']
            print(f"    b{b}: pairs found {d['pairs_found']} hit {d['hit']}/{d['pairs_true']} "
                  f"(prec {d['precision']:.2f} rec {d['recall']:.2f}) | "
                  f"bias label {e['pair_bias_label']['gc']:+.2f} oracle {e['pair_bias_oracle_pairs']['gc']:+.2f} "
                  f"random {e['pair_bias_random']['gc']:+.2f} | "
                  f"gate label {e['pair_gate_label']['gc']:+.2f} oracle {e['pair_gate_oracle_pairs']['gc']:+.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("A. DISCOVERY: precision/recall of label-identified (pred,true) error")
    print("   pairs vs the val-truth pairs. This is 'labels reveal the problem'.")
    print("B. DECODE re-ranking (NO W update): pair_bias (logit offset) and")
    print("   pair_gate (margin flip). label vs oracle-pair vs random:")
    print("   label ~ oracle >> 0 -> the error-correction loop works end to end.")
    print("   label << oracle     -> pair discovery is the bottleneck.")
    print("   random ~ label      -> the correction is just noise.")


if __name__ == "__main__":
    main()
