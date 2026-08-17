"""al_diffusion_shift_diag.py: Iteration-2 AL mechanism test (eval-only, no plots).

Iteration 1 showed the query rule works (influence > confidence) but the
cluster + hard-distance-propagation grounding fails (65%-pure clusters propagate
35% wrong labels into T, poisoning the ridge below frozen). Two replacements are
tested here:

  A. GRAPH DIFFUSION with QUERIED anchors: no clustering, no kNN search. Pick K
     anchors by the validated influence rule (class-floored: one anchor per
     class by max influence first, then remaining budget by influence; also pure
     influence, confidence, and random controls), label them TRUE (simulated
     oracle), diffuse through the HDC Hamming-similarity point graph
     Y_diff = (I - a G)^-1 Y_sparse via matrix-free CG (implicit all-pairs, no
     n x n matrix), then the standard ridge S=all, T=Y_diff. Budget curve
     K in {8, 17, 34, 68}; alpha in {0.5, 0.9}.
     Efficiency measured: diffusion solve time + ridge fit time (the whole AL
     loop, no clustering).

  B. PARTIAL SHIFT STRUCTURE robustness: can a few classes' labels estimate the
     corruption shift for ALL classes (skip labels)? Per-class shift
     shift_c = mu_c(corrupted) - proto_c(clean). If the shifts are partially
     aligned, the GLOBAL mean shift estimated from k labeled classes corrects
     the prototypes of the UNLABELED classes too. Sweeps k in {2, 4, 8, 16}:
       - carry-over: apply the global shift estimate to ALL classes' prototypes
         (and to the probe as a per-class decode bias) -- the label-skipping
         arm;
       - per-class-only: correct only the labeled classes (others stay clean) --
         the no-carry-over control;
       - oracle shift: correct every class with its own shift (the structure's
         ceiling).
     Also reports the shift pairwise-cosine matrix statistics (how aligned the
     shifts actually are) and how much the global shift corrects the unlabeled
     classes' prototypes toward their true corrupted means.

Outputs: JSON + log, doc-ready synthesis per condition. No plots.

Usage:
  uv run python robust_diagnostic/al_diffusion_shift_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_diffusion_shift_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import yaml
import numpy as np
import torch

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
DGLSSPP_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp'
DGLSSPP_METHOD = 'supcon_vib_dglsspp'
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
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(z_flat.cpu())
            lbls.append(labels[mask].cpu())
    return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)

def hdc_codes(feats, proj, device, chunk=100000):
    codes = []
    for s in range(0, len(feats), chunk):
        codes.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(codes, dim=0)

def onehot(lbls, num_classes):
    y = torch.zeros(len(lbls), num_classes)
    y[torch.arange(len(lbls)), lbls.long()] = 1.0
    return y

def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append((codes[s:s + chunk].float() @ W).argmax(dim=1))
    return torch.cat(preds)

def scores(W, codes, chunk=100000):
    W = W.detach().cpu()
    outs = []
    for s in range(0, len(codes), chunk):
        outs.append(codes[s:s + chunk].float() @ W)
    return torch.cat(outs, dim=0)

def proto_decode(protos, proto_lbls, codes, chunk=50000):
    protos = protos / (protos.norm(dim=1, keepdim=True) + 1e-8)
    preds = []
    for s in range(0, len(codes), chunk):
        sims = codes[s:s + chunk].float() @ protos.t()
        preds.append(proto_lbls[sims.argmax(dim=1)].cpu())
    return torch.cat(preds)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def tic():
    sync()
    return time.time()

def toc(t0):
    sync()
    return time.time() - t0

# ---------------- Nystrom influence (the validated query signal) ----------------

def nystrom_influence(Xd, lam, m, device):
    """I_i ~= ||(S + lI)^-1 x_i|| in the Nystrom subspace."""
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    M = torch.linalg.inv(Shat)
    C = Xd @ P
    MC = C @ M
    return (MC.norm(dim=1) * (Xd.shape[1] ** 0.5)).cpu()

# ---------------- label diffusion on the point graph ----------------

def diffuse_labels(Xd, Y_sparse, alpha, device, iters=20):
    """Y_diff = (I - a G)^-1 Y_sparse on the normalized Hamming-similarity graph
    A = (X X^T / d + 1 1^T) / 2 in [0,1]; G = D^-1/2 A D^-1/2. Matrix-free CG."""
    n, d = Xd.shape
    D = (Xd @ (Xd.t() @ torch.ones(n, 1, device=device)) / d + n) / 2
    D_inv_sqrt = (D + 1e-8).pow(-0.5).view(-1)
    Yd = Y_sparse.float().to(device)
    def A(z):
        w = D_inv_sqrt.unsqueeze(1) * z
        Aw = (Xd @ (Xd.t() @ w) / d + w.sum(dim=0, keepdim=True).expand(n, -1)) / 2
        Gz = D_inv_sqrt.unsqueeze(1) * Aw
        return z - alpha * Gz
    x = torch.zeros_like(Yd)
    b = Yd
    r = b - A(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A(p)
        alpha_k = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha_k.unsqueeze(0) * p
        r = r - alpha_k.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    return x.float()

# ---------------- the ridge (S_all + T = diffused labels) ----------------

def ridge_fit_soft(Xd, Y, lam, iters, m, device):
    """W = (S_all + lI)^-1 X^T Y with SOFT Y (diffused labels)."""
    torch.manual_seed(SKETCH_SEED)
    P = (torch.rand(Xd.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = Xd @ P
    Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    That = XP.t() @ Yd
    x = P @ torch.linalg.solve(Shat, That)
    b = Xd.t() @ Yd
    def A(v):
        return Xd.t() @ (Xd @ v)
    r = b - A(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A(p)
        alpha_k = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha_k.unsqueeze(0) * p
        r = r - alpha_k.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    return x.float()

# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--max_clean", type=int, default=200000)
    parser.add_argument("--nystrom_m", type=int, default=1000)
    parser.add_argument("--cg_iters", type=int, default=8)
    parser.add_argument("--diffuse_iters", type=int, default=20)
    parser.add_argument("--budgets", type=str, default="8,17,34,68")
    parser.add_argument("--diffuse_alphas", type=str, default="0.5,0.9")
    parser.add_argument("--shift_ks", type=str, default="2,4,8,16")
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_diffusion_shift_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    budgets = [int(x) for x in args.budgets.split(',')]
    d_alphas = [float(x) for x in args.diffuse_alphas.split(',')]
    shift_ks = [int(x) for x in args.shift_ks.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    results = {'label': args.label, 'conds': {}}

    for cond in conds:
        t_cond = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        pool_codes = hdc_codes(pool, proj, device)
        val_codes = hdc_codes(val, proj, device)

        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]
        clean_codes = hdc_codes(fa[ci], proj, device)

        # clean prototypes + frozen probe
        proto_pairs = []
        for c in range(1, NUM_CLASSES):
            m = la[ci] == c
            if int(m.sum().item()) > 0:
                proto_pairs.append((c, clean_codes[m].float().mean(dim=0)))
        proto_ids = torch.tensor([c for c, _ in proto_pairs])
        proto_mat = torch.stack([p for _, p in proto_pairs])
        Xc = clean_codes.float().to(device)
        W_clean = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam,
                                 args.cg_iters, args.nystrom_m, device)
        del clean_codes
        Xd = pool_codes.float().to(device)

        # per-point signals
        sm = torch.softmax(scores(W_clean, pool_codes), dim=1)
        pconf = sm.max(dim=1).values
        ppred = sm.argmax(dim=1)
        pseudo_correct = (ppred == pl)
        I = nystrom_influence(Xd, args.lam, args.nystrom_m, device)

        r = {'refs': {}, 'diffusion': {}, 'shift': {}, 'efficiency': {},
             'signals': {}, 'synthesis': []}

        def mw(W):
            return compute_miou(decode(W, val_codes), vl)

        # references
        W_oracle = ridge_fit_soft(Xd, onehot(pl, NUM_CLASSES), args.lam,
                                  args.cg_iters, args.nystrom_m, device)
        r['refs'] = {
            'frozen': mw(W_clean),
            'oracle': mw(W_oracle),
            'proto_frozen': compute_miou(
                proto_decode(proto_mat, proto_ids, val_codes), vl),
            'pseudo_acc': float(pseudo_correct.float().mean().item()),
        }
        r['signals'] = {'disagree_rate': float((proto_ids[(pool_codes.float() @ proto_mat.t()).argmax(dim=1)] != ppred).float().mean().item())}

        # ---- A. diffusion with queried anchors ----
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        # anchor ranks
        class_max_I = {}
        for c in classes:
            m = pl == c
            if int(m.sum().item()) > 0:
                class_max_I[c] = float(I[m].max().item())
        def anchors_class_floor(K):
            """One anchor per class by max influence (class coverage floor), then
            remaining budget by pure influence."""
            order_classes = sorted(classes, key=lambda c: -class_max_I[c])
            sel = []
            for c in order_classes:
                m = pl == c
                if len(sel) >= K:
                    break
                j = int(I[m].argmax().item())
                sel.append(m.nonzero().squeeze(1)[j])
            if len(sel) < K:
                rest = torch.tensor(sel)
                remain = torch.ones(len(pool), dtype=torch.bool)
                remain[rest] = False
                extra = torch.argsort(I[remain], descending=True)[:K - len(sel)]
                sel.extend(remain.nonzero().squeeze(1)[extra].tolist())
            return torch.tensor(sel[:K])
        def anchors_by(sig, K):
            return torch.argsort(sig, descending=True)[:K]
        torch.manual_seed(3)
        rand_perm = torch.randperm(len(pool))

        t_fit = tic()
        ridge_fit_soft(Xd, onehot(pl, NUM_CLASSES), args.lam,
                       args.cg_iters, args.nystrom_m, device)
        t_fit = toc(t_fit)
        r['efficiency']['ridge_fit_s'] = t_fit

        diff = r['diffusion']
        for K in budgets:
            for aname, idx in [('influence_floor', anchors_class_floor(K)),
                               ('influence', anchors_by(I, K)),
                               ('confidence', anchors_by(pconf, K)),
                               ('random', rand_perm[:K])]:
                Y_sparse = torch.zeros(len(pool), NUM_CLASSES)
                Y_sparse[idx] = onehot(pl[idx], NUM_CLASSES)
                for a in d_alphas:
                    t_d = tic()
                    Y_diff = diffuse_labels(Xd, Y_sparse, a, device, args.diffuse_iters)
                    t_d = toc(t_d)
                    W = ridge_fit_soft(Xd, Y_diff, args.lam, args.cg_iters,
                                       args.nystrom_m, device)
                    diff[f'K{K}_{aname}_a{a}'] = {
                        'miou': mw(W),
                        'anchor_acc': float((ppred[idx] == pl[idx]).float().mean().item()),
                        'diffuse_time_s': t_d,
                        'fit_time_s': t_fit,
                        'total_time_s': t_d + t_fit,
                    }

        # ---- B. partial shift structure ----
        sh = r['shift']
        # per-class shifts: corrupted class mean - clean prototype (code space)
        shifts = {}
        for c in classes:
            m_c = pool_codes[pl == c].float()
            if len(m_c) < 50:
                continue
            mu_c = m_c.mean(dim=0)
            proto_c = proto_mat[proto_ids == c][0]
            shifts[c] = mu_c - proto_c
        cs = sorted(shifts)
        if len(cs) >= 2:
            S = torch.stack([shifts[c] / (shifts[c].norm() + 1e-8) for c in cs])
            Cm = S @ S.t()
            ii = torch.eye(len(cs), dtype=torch.bool)
            sh['shift_pairwise_cos'] = {'mean': float(Cm[~ii].abs().mean().item()),
                                        'std': float(Cm[~ii].abs().std().item()),
                                        'n': len(cs)}
            # global shift from ALL classes (the structure ceiling)
            g_all = torch.stack([shifts[c] for c in cs]).mean(dim=0)
            sh['global_shift_norm'] = float(g_all.norm().item())
            sh['shift_norm_mean'] = float(torch.stack([shifts[c].norm() for c in cs]).mean().item())

        # corrected-prototype decode with the global shift from k labeled classes
        def decode_shifted_protos(g_shift, labeled):
            pm = proto_mat.clone()
            for c in classes:
                if c in labeled:
                    pm[proto_ids == c] = proto_mat[proto_ids == c] + shifts[c]
                else:
                    pm[proto_ids == c] = proto_mat[proto_ids == c] + g_shift
            return compute_miou(proto_decode(pm, proto_ids, val_codes), vl)
        for k in shift_ks:
            k = min(k, len(cs))
            labeled = cs[:k]                       # highest-influence classes first
            g_k = torch.stack([shifts[c] for c in labeled]).mean(dim=0)
            entry = {'k': k, 'labeled_classes': labeled,
                     'carry_over': decode_shifted_protos(g_k, labeled),
                     'per_class_only': decode_shifted_protos(
                         torch.zeros_like(g_k), labeled),
                     'global_shift': float(g_k.norm().item())}
            # how much the global shift corrects UNLABELED classes toward their
            # true corrupted means (cosine before vs after)
            corr_before, corr_after = [], []
            for c in cs:
                if c in labeled:
                    continue
                mu_c = pool_codes[pl == c].float().mean(dim=0)
                proto_c = proto_mat[proto_ids == c][0]
                corr_before.append(float(
                    (proto_c * mu_c).sum().item() /
                    (proto_c.norm().item() * mu_c.norm().item() + 1e-8)))
                pc = proto_c + g_k
                corr_after.append(float(
                    (pc * mu_c).sum().item() /
                    (pc.norm().item() * mu_c.norm().item() + 1e-8)))
            if corr_before:
                entry['unlabeled_cos_before'] = float(np.mean(corr_before))
                entry['unlabeled_cos_after'] = float(np.mean(corr_after))
            sh[f'k{k}'] = entry
        # oracle shift (every class its own): the structure's ceiling
        sh['oracle_shift'] = decode_shifted_protos(
            torch.zeros(Xd.shape[1]), cs)

        # ---- synthesis ----
        syn = r['synthesis']
        syn.append(f"COND {cond}: frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} / "
                   f"proto_frozen {r['refs']['proto_frozen']:.3f} | pseudo acc {r['refs']['pseudo_acc']:.3f}")
        for a in d_alphas:
            line = f"  diffusion a={a}: " + " ".join(
                f"K{K}:{diff[f'K{K}_influence_floor_a{a}']['miou']:.3f}"
                for K in budgets)
            syn.append(line)
        mid = str(budgets[len(budgets) // 2])
        syn.append(f"  anchor rules @K{mid} a0.5: influence_floor "
                   f"{diff[f'K{mid}_influence_floor_a0.5']['miou']:.3f} / influence "
                   f"{diff[f'K{mid}_influence_a0.5']['miou']:.3f} / confidence "
                   f"{diff[f'K{mid}_confidence_a0.5']['miou']:.3f} / random "
                   f"{diff[f'K{mid}_random_a0.5']['miou']:.3f}")
        syn.append(f"  efficiency: diffuse {diff[f'K{mid}_influence_floor_a0.5']['diffuse_time_s']:.3f}s "
                   f"+ fit {t_fit:.3f}s = {diff[f'K{mid}_influence_floor_a0.5']['total_time_s']:.3f}s (no clustering)")
        if 'shift_pairwise_cos' in sh:
            syn.append(f"  shift: pairwise-cos {sh['shift_pairwise_cos']['mean']:.3f} | "
                       f"global norm {sh['global_shift_norm']:.2f} vs mean per-class "
                       f"{sh['shift_norm_mean']:.2f}")
        for k in shift_ks:
            e = sh[f'k{min(k, len(cs))}']
            if 'unlabeled_cos_before' in e:
                syn.append(f"  shift k={e['k']}: carry_over {e['carry_over']:.3f} / "
                           f"per_class_only {e['per_class_only']:.3f} / oracle_shift "
                           f"{sh['oracle_shift']:.3f} | unlabeled cos before "
                           f"{e['unlabeled_cos_before']:.3f} after {e['unlabeled_cos_after']:.3f}")
            else:
                syn.append(f"  shift k={e['k']}: carry_over {e['carry_over']:.3f} / "
                           f"per_class_only {e['per_class_only']:.3f} / oracle_shift "
                           f"{sh['oracle_shift']:.3f} (all classes labeled)")

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in syn:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("A. diffusion: K budget -> mIoU for influence_floor (one anchor per")
    print("   class first, then influence), pure influence, confidence, random.")
    print("   If influence_floor climbs above frozen toward oracle, graph")
    print("   diffusion of queried anchors is the AL mechanism (no clustering,")
    print("   no kNN). Compare vs Iteration-1 grounded_all (which stayed below")
    print("   frozen): diffusion should fix the 35%-wrong propagation.")
    print("B. shift: carry_over applies the GLOBAL shift from k labeled classes to")
    print("   ALL classes' prototypes (the label-skipping arm); per_class_only")
    print("   corrects only the labeled classes; oracle_shift is the ceiling.")
    print("   If carry_over(k) approaches per_class_only(16) quickly, the shift")
    print("   structure lets a few classes' labels skip the rest. unlabeled cos")
    print("   before/after shows how much the global shift moves the unlabeled")
    print("   prototypes toward their true corrupted means.")
    print("efficiency: the whole AL loop is diffuse + fit, no clustering: ~0.2s.")

if __name__ == "__main__":
    main()
