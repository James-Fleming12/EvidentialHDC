"""al_full_dataset_diag.py: FULL-DATASET mIoU for zero-shot W0, ceiling W*, and the
56+500 random-bank AL W_res on every point of every frame of KITTI seq 08
(paper-ready harness).

The 100-frame harness underestimates the recoverable headroom (the "large" tier
raised fog's gap +0.167 -> +0.245 at 200 frames). This run evaluates on the FULL
dataset: every frame of seq 08 for the clean W0 fit source and every corrupted
condition, with the SAME probe machinery as the README harness:

  * zero-shot W0 : exact-ridge fit on <= clean_fit_n clean points (reservoir
                   across ALL clean frames, seed 7).
  * ceiling  W*  : exact-ridge fit on <= pool_cap corrupted-pool points
                   (reservoir across ALL frames of the condition, seed 42).
  * AL W_res     : W0 + U8 C with C fit on the 56+500 random bank (seeds 2/3),
                   oracle U from SVD(W* - W0).
  * VAL          : ALL points of ALL frames (pool points excluded), decoded
                   streaming per-frame in chunks; mIoU accumulated as per-class
                   tp/fp/fn counters (memory flat in dataset size).

This is the paper-ready table: frozen / ceiling / AL on the same full val set.

Usage:
  uv run python robust_diagnostic/al_full_dataset_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label full_ep10 \
    --out robust_diagnostic/logs/al_full_dataset_ep10.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection

NUM_CLASSES = 17
CONDS_ALL = ["fog", "crosstalk", "snow", "wet_ground", "incomplete_echo",
             "beam_missing", "motion_blur", "cross_sensor"]

def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)

def stream_frames(model, parser, device, max_frames=0):
    """Yield (zf, labels, frame_idx) per frame, ALL frames unless max_frames > 0."""
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if max_frames > 0 and i >= max_frames:
                break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            out = model(in_vol)
            z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            yield zf.cpu(), labels[mask].cpu(), i

def reservoir_collect(stream, cap, seed):
    """True reservoir sampling over a frame stream: memory bounded by `cap`,
    uniform over ALL points seen (each point i kept with prob cap/(i+1), slot
    j-1 for the first cap points, random slot after). Returns (feats, lbls,
    keys) with len <= cap; keys[i] = (frame_idx, local_pt_idx)."""
    buf_f = torch.zeros(cap, 128)
    buf_l = torch.zeros(cap, dtype=torch.long)
    buf_k = torch.zeros(cap, 2, dtype=torch.long)
    n = 0
    g = torch.Generator().manual_seed(seed)
    for zf, labels, fi in stream:
        m = len(zf)
        if m == 0:
            continue
        j = n + torch.arange(1, m + 1)                     # global 1-indexed positions
        keep = torch.rand(m, generator=g) < (cap / j.float())
        slot = torch.where(j <= cap, j - 1, torch.randint(0, cap, (m,), generator=g))
        kept = keep.nonzero(as_tuple=True)[0]
        if len(kept):
            buf_f[slot[kept]] = zf[kept]
            buf_l[slot[kept]] = labels[kept]
            buf_k[slot[kept], 0] = fi
            buf_k[slot[kept], 1] = kept
        n += m
        del zf, labels
    n = min(n, cap)
    return buf_f[:n], buf_l[:n], buf_k[:n]

def hdc_codes(feats, proj, device, chunk=100000):
    out = []
    for s in range(0, len(feats), chunk):
        out.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(out)

def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc)
    y[torch.arange(len(lbls)), lbls.long()] = 1
    return y

def ridge_fit_exact(X, Y, lam, device, chunk=50000):
    d = X.shape[1]; nc = Y.shape[1]
    S = torch.zeros(d, d, device=device); T = torch.zeros(d, nc, device=device)
    for s in range(0, len(X), chunk):
        Xc = X[s:s + chunk].to(device); Yc = Y[s:s + chunk].to(device)
        S += Xc.t() @ Xc; T += Xc.t() @ Yc
    return torch.linalg.solve(S + lam * torch.eye(d, device=device), T).float()

def knn_predict(val_feats, bank_feats, bank_labels, k=1, device='cuda', chunk=4096):
    val_n = F.normalize(val_feats.float(), dim=1)
    bank_n = F.normalize(bank_feats.float(), dim=1).to(device)
    preds = []
    for s in range(0, len(val_n), chunk):
        sim = val_n[s:s + chunk].to(device) @ bank_n.t()
        preds.append(bank_labels[sim.argmax(1).cpu()])
    return torch.cat(preds)

class ConfAccum:
    """Streaming per-class tp/fp/fn accumulation (memory flat in dataset size)."""
    def __init__(self, nc=NUM_CLASSES):
        self.tp = torch.zeros(nc); self.fp = torch.zeros(nc); self.fn = torch.zeros(nc)
        self.present = torch.zeros(nc, dtype=torch.bool); self.n = 0
    def update(self, preds, lbls):
        p = preds.long(); l = lbls.long()
        for c in range(1, NUM_CLASSES):
            pc = (p == c); lc = (l == c)
            self.tp[c] += (pc & lc).sum().item()
            self.fp[c] += (pc & ~lc).sum().item()
            self.fn[c] += (~pc & lc).sum().item()
            self.present[c] |= lc.any().item()
        self.n += len(l)
    def miou(self):
        ious = []
        for c in range(1, NUM_CLASSES):
            if not self.present[c]:
                continue
            d = self.tp[c] + self.fp[c] + self.fn[c]
            ious.append(float(self.tp[c] / d) if d > 0 else 0.0)
        return float(np.mean(ious)) if ious else 0.0

def stream_decode_full(model, parser, proj, device, Ws, exclude=None, max_frames=0, chunk=100000):
    """Decode ALL points of all frames with each W in the dict, skipping the
    (frame,pt) pairs in `exclude` (dict frame -> set of local pt idx).
    Returns {name: ConfAccum}."""
    accs = {name: ConfAccum() for name in Ws}
    Wd = {name: W.to(device) for name, W in Ws.items()}
    for zf, labels, fi in stream_frames(model, parser, device, max_frames):
        n = len(zf)
        if n == 0:
            continue
        skip = None
        if exclude and fi in exclude:
            ex = exclude[fi]
            skip = torch.tensor([i in ex for i in range(n)])
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            codes = torch.sign(zf[s:e].to(device) @ proj).float()
            for name, W in Wd.items():
                preds = (codes @ W).argmax(1).cpu()
                lbls = labels[s:e]
                if skip is not None:
                    m = ~skip[s:e]
                    preds, lbls = preds[m], lbls[m]
                accs[name].update(preds, lbls)
            del codes
        del zf, labels
    return accs

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = ALL frames of seq 08")
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--bank_k", type=int, default=8)
    ap.add_argument("--bank_extra", type=int, default=500)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="full_ep10")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)

    # ---- clean W0 fit over ALL clean frames (reservoir, seed 7) ----
    t0 = time.time()
    print("=== clean W0 fit (all clean frames, reservoir %d) ===" % args.clean_fit_n)
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    cf, cl, ck = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                   args.clean_fit_n, 7)
    Xc = hdc_codes(cf, proj, device).float()
    W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
    print(f"  clean: {len(cf)} points, W0 fit done ({time.time()-t0:.0f}s)")
    del cf, cl, ck, Xc
    torch.cuda.empty_cache()

    results = {'label': args.label, 'method': args.method_b, 'max_frames': args.max_frames,
               'clean_fit_n': args.clean_fit_n, 'pool_cap': args.pool_cap, 'conds': {}}
    for cond in conds:
        t0 = time.time()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        cparser = build_parser(cdir, DATA, ARCH)

        # ---- pass 1: reservoir pool + zero-shot decode over ALL frames ----
        print(f"\n=== {cond} (pass 1: pool + zero-shot) ===")
        pf, pl, pk = reservoir_collect(stream_frames(model, cparser, device, args.max_frames),
                                       args.pool_cap, 42)
        print(f"  pool: {len(pf)} points")
        Xp = hdc_codes(pf, proj, device).float()
        Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device)
        print(f"  ceiling W* fit done ({time.time()-t0:.0f}s)")

        # ---- bank 56+500 (same seeds as the README harness) ----
        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}
        lab_idx = []
        for c in classes:
            idx = cls_idx[c]
            if len(idx) < max(50, args.bank_k):
                continue
            torch.manual_seed(2)
            lab_idx.append(idx[torch.randperm(len(idx))[:args.bank_k]])
        lab_idx = torch.cat(lab_idx) if lab_idx else torch.tensor([], dtype=torch.long)
        avail = torch.arange(len(pf))
        mask = torch.ones(len(pf), dtype=torch.bool); mask[lab_idx] = False
        torch.manual_seed(3)
        extra = avail[mask][torch.randperm(len(avail[mask]))[:args.bank_extra]]
        bank_idx = torch.cat([lab_idx, extra])
        print(f"  bank: {len(bank_idx)} points ({len(lab_idx)} true + {len(extra)} random)")

        # ---- W_res with oracle U (r=8) on 56+500 pseudo vs true ----
        R = (Ws - W0).detach().cpu().float()
        U8 = torch.linalg.svd(R.double(), full_matrices=False)[0][:, :args.r].float()
        extra_pred = knn_predict(pf[extra], pf[lab_idx], pl[lab_idx], k=1, device=device)
        X_lab = torch.cat([Xp[lab_idx], Xp[extra]], dim=0)
        Y_pseudo = torch.cat([onehot(pl[lab_idx], NUM_CLASSES), onehot(extra_pred, NUM_CLASSES)], dim=0)
        Y_true = torch.cat([onehot(pl[lab_idx], NUM_CLASSES), onehot(pl[extra], NUM_CLASSES)], dim=0)
        XU = X_lab.to(device).float() @ U8.to(device)
        A = XU.t() @ XU + 1e-6 * torch.eye(args.r, device=device)
        W0d = W0.to(device)
        Cp = torch.linalg.solve(A, XU.t() @ (Y_pseudo.to(device).float() - X_lab.to(device).float() @ W0d)).cpu()
        Ct = torch.linalg.solve(A, XU.t() @ (Y_true.to(device).float() - X_lab.to(device).float() @ W0d)).cpu()
        W_res_pseudo = W0.detach().cpu() + (U8.cpu() @ Cp)
        W_res_true = W0.detach().cpu() + (U8.cpu() @ Ct)

        # ---- pass 2: FULL-dataset decode (all frames, pool points excluded) ----
        from collections import defaultdict
        ex_by_frame = defaultdict(set)
        for f, i in pk.tolist():
            ex_by_frame[f].add(i)
        ex_by_frame = {f: s for f, s in ex_by_frame.items()}
        print(f"  pass 2: full-dataset decode over ALL frames...")
        accs = stream_decode_full(model, cparser, proj, device,
                                  {'frozen': W0.detach().cpu(), 'ceiling': Ws.detach().cpu(),
                                   'W_res_pseudo': W_res_pseudo, 'W_res_true': W_res_true},
                                  exclude=ex_by_frame, max_frames=args.max_frames)
        frozen = accs['frozen'].miou(); ceiling = accs['ceiling'].miou()
        wrp = accs['W_res_pseudo'].miou(); wrt = accs['W_res_true'].miou()
        n_val = accs['frozen'].n
        results['conds'][cond] = {
            'n_pool': len(pf), 'n_val': n_val,
            'frozen': frozen, 'ceiling': ceiling, 'gap': ceiling - frozen,
            'W_res_pseudo': wrp, 'W_res_pseudo_delta': wrp - frozen,
            'W_res_true': wrt, 'W_res_true_delta': wrt - frozen,
            'bank_n': len(bank_idx),
        }
        print(f"  frozen {frozen:.3f} / ceiling {ceiling:.3f} (gap {ceiling-frozen:+.3f}) | "
              f"W_res pseudo {wrp:.3f} ({wrp-frozen:+.3f}) true {wrt:.3f} ({wrt-frozen:+.3f}) | "
              f"n_val {n_val} ({time.time()-t0:.0f}s)")
        del pf, pl, pk, Xp, Ws, R, U8, X_lab, accs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
