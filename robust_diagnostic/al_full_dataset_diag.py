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

def stream_frames(model, parser, device, max_frames=0, progress=None, report=500):
    """Yield (zf, labels, frame_idx) per frame, ALL frames unless max_frames > 0.
    With `progress` (a label string), prints every `report` frames so long
    extraction passes are not silent."""
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if max_frames > 0 and i >= max_frames:
                break
            if progress is not None and i % report == 0:
                print(f"  [{progress}] frame {i}...", flush=True)
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
            buf_l[slot[kept]] = labels[kept].long()
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
    # Solve in float64: some projection variants (sparse_k1) give a rank-deficient
    # code matrix whose fp32 solve is numerically singular; double fixes it.
    A = S.double() + lam * torch.eye(d, dtype=torch.float64, device=device)
    return torch.linalg.solve(A, T.double()).float()

def ridge_fit_balanced(X, Y, counts, lam, device, mode='w', chunk=50000):
    """Class-balanced ridge probes (minority-class robustness diagnostic).
    mode='w'  : per-sample weight w_i = 1/N_{y_i} so every class contributes equal
                total mass to T (fixes the minority T-column under-representation;
                the linear-separability lever).
    mode='lam': per-class ridge lambda_c = lam * N / N_c (the C29 stability
                proposal: MORE shrinkage for the classes with fewer points, so
                their noisy directions do not blow up the fit)."""
    d = X.shape[1]; nc = Y.shape[1]
    N = len(X)
    counts = counts.float().clamp(min=1.0).to(device)
    if mode == 'w':
        # weighted normal equations: (X^T W X + lam I) W = X^T W Y, W_ii = 1/N_{y_i}
        S = torch.zeros(d, d, device=device); T = torch.zeros(d, nc, device=device)
        for s in range(0, len(X), chunk):
            Xc = X[s:s + chunk].to(device); Yc = Y[s:s + chunk].to(device)
            w = (1.0 / counts[Yc.argmax(1)]).unsqueeze(1)
            Xw = Xc * w; Yw = Yc * w
            S += Xw.t() @ Xc; T += Xw.t() @ Yc
        return torch.linalg.solve(S + lam * torch.eye(d, device=device), T).float()
    # mode='lam': per-class regularization, solved via the shared eigensystem
    S = torch.zeros(d, d, device=device); T = torch.zeros(d, nc, device=device)
    for s in range(0, len(X), chunk):
        Xc = X[s:s + chunk].to(device); Yc = Y[s:s + chunk].to(device)
        S += Xc.t() @ Xc; T += Xc.t() @ Yc
    lam_c = lam * (N / counts)                      # (nc,), counts already on device
    evals, evecs = torch.linalg.eigh(S.double())
    # W[:,c] = V ( (V^T T[:,c]) / (evals + lam_c[c]) )
    VT = evecs.t() @ T.double()
    W = evecs @ (VT / (evals.unsqueeze(1) + lam_c.double().unsqueeze(0)))
    return W.float()

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

def stream_decode_full(model, parser, proj, device, decoders, exclude=None, max_frames=0, chunk=100000):
    """Decode ALL points of all frames with each decoder, skipping the
    (frame,pt) pairs in `exclude` (dict frame -> sorted tensor of local pt idx).
    decoders: {name: {'type':'w', 'W': (10000,K)} | {'type':'proto', 'protos':
    (K,10000), 'proto_lbls': (K,)} | {'type':'w_bias', 'W': (10000,K),
    'bias': (K,)}}. Returns {name: ConfAccum}."""
    accs = {name: ConfAccum() for name in decoders}
    prep = {}
    for name, dec in decoders.items():
        if dec['type'] == 'w':
            prep[name] = ('w', dec['W'].to(device))
        elif dec['type'] == 'w_bias':
            prep[name] = ('w_bias', dec['W'].to(device), dec['bias'].to(device))
        else:
            prep[name] = ('proto', dec['protos'].to(device), dec['proto_lbls'].to(device))
    for zf, labels, fi in stream_frames(model, parser, device, max_frames):
        n = len(zf)
        if n == 0:
            continue
        skip = None
        if exclude and fi in exclude:
            ex = exclude[fi]
            pos = torch.searchsorted(ex, torch.arange(n))
            skip = (pos < len(ex)) & (ex[pos.clamp(max=len(ex) - 1)] == torch.arange(n))
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            codes = torch.sign(zf[s:e].to(device) @ proj).float()
            for name, p in prep.items():
                if p[0] == 'w':
                    preds = (codes @ p[1]).argmax(1).cpu()
                elif p[0] == 'w_bias':
                    preds = (codes @ p[1] + p[2]).argmax(1).cpu()
                else:
                    sims = F.normalize(codes, p=2, dim=1) @ p[1].t()
                    preds = p[2][sims.argmax(1)].cpu()
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

def build_prototypes(codes, lbls, num_classes=NUM_CLASSES, device='cuda'):
    """Class prototypes from binarized codes: per-class mean, L2-normalized
    (mirrors oracle_core.build_hdc_prototypes but on codes)."""
    protos = torch.zeros(num_classes, codes.shape[1], device=device)
    counts = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        mask = lbls.to(device) == c
        if mask.sum() > 0:
            protos[c] += codes.to(device)[mask].sum(dim=0)
            counts[c] += mask.sum()
    for c in range(num_classes):
        if counts[c] > 0:
            protos[c] /= counts[c]
    return F.normalize(protos, p=2, dim=1)

def build_nuscenes_parser(root, data, arch):
    """NuScenes parser with the 32-beam projection (vs KITTI's 64-beam): NuScenes
    scans must be projected into H=32 x W=1024 to keep point density (the
    unsup_kitti-nuscenes.py setup)."""
    sensor = arch["dataset"]["sensor"].copy()
    sensor["fov_up"] = 10.0
    sensor["fov_down"] = -30.0
    sensor["img_prop"] = sensor["img_prop"].copy()
    sensor["img_prop"]["height"] = 32
    sensor["img_prop"]["width"] = 1024
    return Parser(root=root, train_sequences=data["split"]["valid"],
                  valid_sequences=data["split"]["valid"], test_sequences=None,
                  labels=data["labels"], color_map=data.get("color_map", {}),
                  learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"],
                  sensor=sensor, max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)

def eval_target_condition(model, parser, proj, device, W0, protos_clean, args,
                          label="target", cond_name=None, bal=None):
    """Shared per-target eval: reservoir pool (seed 42) -> ceiling W* + pool
    protos, 56+500 bank (seeds 2/3) -> W_res pseudo/true (oracle U r=8), then a
    FULL streaming decode of every point (pool excluded) with R4 (frozen /
    ceiling / W_res) and R1 (frozen / ceiling) decoders. Returns the metrics dict.
    `label` is used for print lines; `cond_name` names the JSON key.
    `bal` (optional dict) adds the class-balanced probe variants (minority-class
    diagnostic): 'W0_w'/'W0_lam' balanced clean probes, 'clean_counts'. When
    given, the pool-fit balanced probes are also fit and all are decoded in the
    same pass."""
    from collections import defaultdict
    t0 = time.time()
    print(f"\n=== [{label}] {cond_name or 'target'} (pass 1: pool) ===")
    pf, pl, pk = reservoir_collect(stream_frames(model, parser, device, args.max_frames),
                                   args.pool_cap, 42)
    print(f"  pool: {len(pf)} points")
    Xp = hdc_codes(pf, proj, device).float()
    Ws = ridge_fit_exact(Xp, onehot(pl, NUM_CLASSES), args.lam, device)
    protos_pool = build_prototypes(Xp, pl, device=device)
    bal_extra = {}
    if bal is not None:
        pool_counts = torch.bincount(pl.long(), minlength=NUM_CLASSES).float()
        print(f"  balanced probes (pool counts: "
              f"{[int(c) for c in pool_counts if c > 0][:6]}... )")
        Ws_w = ridge_fit_balanced(Xp, onehot(pl, NUM_CLASSES), pool_counts,
                                  args.lam, device, mode='w')
        Ws_lam = ridge_fit_balanced(Xp, onehot(pl, NUM_CLASSES), pool_counts,
                                    args.lam, device, mode='lam')
        # logit-prior bias: tau * log(N_c / N) on the frozen and pool probes
        tau = getattr(args, 'bal_tau', 1.0)
        log_prior_pool = tau * torch.log(pool_counts.clamp(min=1) / len(pf))
        log_prior_clean = tau * torch.log(bal['clean_counts'].clamp(min=1)
                                          / bal['clean_counts'].sum())
        bal_extra = {
            'W0_w': bal['W0_w'], 'W0_lam': bal['W0_lam'],
            'Ws_w': Ws_w, 'Ws_lam': Ws_lam,
            'logit_prior_clean': log_prior_clean, 'logit_prior_pool': log_prior_pool,
        }
    print(f"  ceiling W* + pool protos done ({time.time()-t0:.0f}s)")

    # bank 56+500 (same seeds as the README harness)
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

    # W_res with oracle U (r=8) on 56+500 pseudo vs true
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

    # pass 2: FULL decode (all frames, pool points excluded)
    ex_by_frame = defaultdict(list)
    for f, i in pk.tolist():
        ex_by_frame[f].append(i)
    ex_by_frame = {f: torch.tensor(sorted(s), dtype=torch.long) for f, s in ex_by_frame.items()}
    print(f"  pass 2: full decode over ALL frames...")
    decoders = {
        'linear_frozen': {'type': 'w', 'W': W0.detach().cpu()},
        'linear_ceiling': {'type': 'w', 'W': Ws.detach().cpu()},
        'linear_W_res_pseudo': {'type': 'w', 'W': W_res_pseudo},
        'linear_W_res_true': {'type': 'w', 'W': W_res_true},
        'proto_frozen': {'type': 'proto', 'protos': protos_clean.cpu(),
                         'proto_lbls': torch.arange(NUM_CLASSES)},
        'proto_ceiling': {'type': 'proto', 'protos': protos_pool.cpu(),
                          'proto_lbls': torch.arange(NUM_CLASSES)},
    }
    for k in ('W0_w', 'W0_lam', 'Ws_w', 'Ws_lam'):
        if k in bal_extra:
            decoders[f'linear_{k}'] = {'type': 'w', 'W': bal_extra[k].detach().cpu()}
    if 'logit_prior_clean' in bal_extra:
        decoders['linear_frozen_logit'] = {'type': 'w_bias', 'W': W0.detach().cpu(),
                                           'bias': bal_extra['logit_prior_clean']}
        decoders['linear_ceiling_logit'] = {'type': 'w_bias', 'W': Ws.detach().cpu(),
                                            'bias': bal_extra['logit_prior_pool']}
    accs = stream_decode_full(model, parser, proj, device, decoders,
                              exclude=ex_by_frame, max_frames=args.max_frames)
    n_val = accs['linear_frozen'].n
    m = {k: accs[k].miou() for k in decoders}
    out = {
        'n_pool': len(pf), 'n_val': n_val,
        'linear_frozen': m['linear_frozen'], 'linear_ceiling': m['linear_ceiling'],
        'linear_gap': m['linear_ceiling'] - m['linear_frozen'],
        'linear_W_res_pseudo': m['linear_W_res_pseudo'],
        'linear_W_res_pseudo_delta': m['linear_W_res_pseudo'] - m['linear_frozen'],
        'linear_W_res_true': m['linear_W_res_true'],
        'linear_W_res_true_delta': m['linear_W_res_true'] - m['linear_frozen'],
        'proto_frozen': m['proto_frozen'], 'proto_ceiling': m['proto_ceiling'],
        'proto_gap': m['proto_ceiling'] - m['proto_frozen'],
        'bank_n': len(bank_idx),
    }
    for k in ('W0_w', 'W0_lam', 'Ws_w', 'Ws_lam'):
        if k in bal_extra:
            out[f'linear_{k}'] = m[f'linear_{k}']
            out[f'linear_{k}_delta'] = m[f'linear_{k}'] - m['linear_frozen']
    for k in ('linear_frozen_logit', 'linear_ceiling_logit'):
        if k in m:
            out[k] = m[k]
            out[k + '_delta'] = m[k] - (m['linear_frozen'] if k.endswith('frozen_logit')
                                        else m['linear_ceiling'])
    print(f"  [R4] frozen {m['linear_frozen']:.3f} / ceiling {m['linear_ceiling']:.3f} "
          f"(gap {m['linear_ceiling']-m['linear_frozen']:+.3f}) | "
          f"W_res pseudo {m['linear_W_res_pseudo']:.3f} "
          f"({m['linear_W_res_pseudo']-m['linear_frozen']:+.3f}) "
          f"true {m['linear_W_res_true']:.3f} "
          f"({m['linear_W_res_true']-m['linear_frozen']:+.3f})")
    if 'linear_W0_w' in m:
        print(f"  [BAL] frozen w {m['linear_W0_w']:.3f} ({m['linear_W0_w']-m['linear_frozen']:+.3f}) "
              f"lam {m['linear_W0_lam']:.3f} ({m['linear_W0_lam']-m['linear_frozen']:+.3f}) | "
              f"ceiling w {m['linear_Ws_w']:.3f} ({m['linear_Ws_w']-m['linear_frozen']:+.3f}) "
              f"lam {m['linear_Ws_lam']:.3f} ({m['linear_Ws_lam']-m['linear_frozen']:+.3f}) | "
              f"logit frozen {m['linear_frozen_logit']:.3f} "
              f"({m['linear_frozen_logit']-m['linear_frozen']:+.3f}) "
              f"ceiling {m['linear_ceiling_logit']:.3f} "
              f"({m['linear_ceiling_logit']-m['linear_ceiling']:+.3f})")
    print(f"  [R1] frozen {m['proto_frozen']:.3f} / ceiling {m['proto_ceiling']:.3f} "
          f"(gap {m['proto_ceiling']-m['proto_frozen']:+.3f}) | n_val {n_val} "
          f"({time.time()-t0:.0f}s)")
    del pf, pl, pk, Xp, Ws, R, U8, X_lab, accs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = ALL frames of seq 08")
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--pool_cap", type=int, default=400000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--bank_k", type=int, default=8)
    ap.add_argument("--bank_extra", type=int, default=500)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--nusc_dir", type=str, default="/mnt/alpha/jmfleming/nuscenes_kitti",
                    help="NuScenes dataset in KITTI format (32-beam)")
    ap.add_argument("--nusc_labels", type=str, default="config/labels/nuscenes_new.yaml")
    ap.add_argument("--nusc", type=int, default=1,
                    help="1 = also evaluate the extractor's frozen/ceiling/AL on NuScenes")
    ap.add_argument("--proj_dim", type=int, default=10000,
                    help="HDC projection dimension (default 10000; code-2000 peaks "
                         "per tta Iteration 2 -- this lets the full harness verify "
                         "the projection-dim effect at full scale)")
    ap.add_argument("--nusc_c_dir", type=str, default="",
                    help="NuScenes-C root in KITTI format (e.g. "
                         ".../nuscenes_c_kitti). When set, also evaluate each "
                         "<cond>/<sev> under it, stored at "
                         "extractors[<label>]['nuscenes_c']['<cond>/<sev>'].")
    ap.add_argument("--nusc_c_conds", type=str, default="",
                    help="comma-separated conditions under --nusc_c_dir "
                         "(default: all 8)")
    ap.add_argument("--nusc_c_sevs", type=str, default="heavy,moderate,light",
                    help="comma-separated severities under --nusc_c_dir")
    ap.add_argument("--bal", type=int, default=1,
                    help="1 = also fit+decode the class-balanced probe variants "
                         "(per-sample w=1/N_c, per-class lam~1/N_c, logit prior)")
    ap.add_argument("--bal_tau", type=float, default=1.0,
                    help="logit-prior strength tau * log(N_c / N) for the bal logit decoders")
    ap.add_argument("--extractors", type=str,
                    default="cov_ep10:supcon_vib_dglsspp_inputin_in_chan:"
                            "robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/"
                            "supcon_vib_dglsspp_inputin_in_chan,"
                            "cov_ep21:supcon_vib_dglsspp_inputin_in_chan:"
                            "robust_diagnostic/logs/med_supcon_vib_dglsspp_inputin_in_chan/"
                            "supcon_vib_dglsspp_inputin_in_chan,"
                            "dglsspp:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp,"
                            "robust:supcon_vib_dglsspp_corsupcon:"
                            "robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon",
                    help="comma-separated label:method:path triplets")
    ap.add_argument("--label", type=str, default="full_ep10")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    if conds == ['none']:  # sentinel: skip KITTI-C conditions entirely
        conds = []
    extractors = [tuple(e.strip().split(':')) for e in args.extractors.split(',') if e.strip()]
    proj = get_hdc_projection(dim_in=128, dim_out=args.proj_dim, device=device)

    results = {'label': args.label, 'max_frames': args.max_frames,
               'clean_fit_n': args.clean_fit_n, 'pool_cap': args.pool_cap,
               'extractors': {lab: {'method': method, 'conds': {}}
                              for lab, method, _ in extractors}}

    # ---- pass 1 (per extractor): clean fit + all KITTI-C conditions ----
    # Keep (model, W0, protos_clean) per extractor for the NuScenes pass later.
    nusc_ready = {}
    for lab, method, path in extractors:
        print(f"\n{'='*80}\n=== extractor {lab} ({method}, {path}) ===\n{'='*80}")
        # Deep-copy ARCH per extractor: GenTrainer mutates ARCH["train"]["twobranch"]
        # in place (norm / input_in / norm_channels for the method), so a shared dict
        # leaks one method's input-normalization config into the NEXT extractor's model
        # construction. E.g. cov-shift (inputin_in_chan) before DGLSS++ built DGLSS++
        # with input_in=True -- wrong architecture (6.786436M vs 6.796804M params) and
        # a partial strict=False checkpoint load. Without the copy, the DGLSS++/Robust
        # columns in the README measured DGLSS++-with-cov-shift-normalization.
        import copy as _copy
        arch = _copy.deepcopy(ARCH)
        trainer = GenTrainer(arch, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)

        # ---- clean W0 + clean prototypes over ALL clean frames (reservoir, seed 7) ----
        t0 = time.time()
        print(f"=== [{lab}] clean fit (all clean frames, reservoir {args.clean_fit_n}) ===")
        cf, cl, ck = reservoir_collect(stream_frames(model, clean_parser, device, args.max_frames),
                                       args.clean_fit_n, 7)
        Xc = hdc_codes(cf, proj, device).float()
        W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device)
        protos_clean = build_prototypes(Xc, cl, device=device)
        bal = None
        if args.bal:
            clean_counts = torch.bincount(cl.long(), minlength=NUM_CLASSES).float()
            W0_w = ridge_fit_balanced(Xc, onehot(cl, NUM_CLASSES), clean_counts,
                                      args.lam, device, mode='w')
            W0_lam = ridge_fit_balanced(Xc, onehot(cl, NUM_CLASSES), clean_counts,
                                        args.lam, device, mode='lam')
            bal = {'W0_w': W0_w, 'W0_lam': W0_lam, 'clean_counts': clean_counts}
            print(f"  balanced W0 (w / lam) done ({time.time()-t0:.0f}s)")
        print(f"  clean: {len(cf)} points, W0 + clean protos done ({time.time()-t0:.0f}s)")
        del cf, cl, ck, Xc
        torch.cuda.empty_cache()
        nusc_ready[lab] = (model, W0, protos_clean, bal)

        for cond in conds:
            t0 = time.time()
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            cparser = build_parser(cdir, DATA, ARCH)
            r = eval_target_condition(model, cparser, proj, device, W0, protos_clean,
                                      args, label=lab, cond_name=cond, bal=bal)
            results['extractors'][lab]['conds'][cond] = r

        # ---- checkpoint after each extractor so a crash keeps the completed results ----
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\n[checkpoint] extractor {lab} KITTI-C done, saved to {args.out}")

    # ---- pass 2: NuScenes cross-dataset transfer, sequentially after ALL KITTI-C ----
    if args.nusc:
        for lab, method, path in extractors:
            model, W0, protos_clean, bal = nusc_ready[lab]
            print(f"\n{'='*80}\n=== [{lab}] NuScenes cross-dataset transfer ===\n{'='*80}")
            if not os.path.isdir(args.nusc_dir):
                print(f"  WARNING: nusc_dir {args.nusc_dir} not found, skipping")
            else:
                nusc_data = yaml.safe_load(open(args.nusc_labels))
                nusc_parser = build_nuscenes_parser(args.nusc_dir, nusc_data, ARCH)
                results['extractors'][lab]['nuscenes'] = eval_target_condition(
                    model, nusc_parser, proj, device, W0, protos_clean,
                    args, label=lab, cond_name='nuscenes', bal=bal)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
            print(f"\n[checkpoint] extractor {lab} NuScenes done, saved to {args.out}")

    # ---- pass 3: NuScenes-C per condition/severity (loop reuses the clean fit) ----
    if args.nusc_c_dir:
        nusc_c_conds = [c.strip() for c in args.nusc_c_conds.split(',') if c.strip()]
        if not nusc_c_conds:
            nusc_c_conds = CONDS_ALL
        nusc_c_sevs = [s.strip() for s in args.nusc_c_sevs.split(',') if s.strip()]
        nusc_c_data = yaml.safe_load(open(args.nusc_labels))
        for lab, method, path in extractors:
            model, W0, protos_clean, bal = nusc_ready[lab]
            res_c = results['extractors'][lab].setdefault('nuscenes_c', {})
            for cond in nusc_c_conds:
                for sev in nusc_c_sevs:
                    d = os.path.join(args.nusc_c_dir, cond, sev)
                    if not os.path.isdir(d):
                        print(f"  WARNING: {d} not found, skipping")
                        continue
                    print(f"\n{'='*80}\n=== [{lab}] NuScenes-C {cond}/{sev} ===\n{'='*80}")
                    nusc_parser = build_nuscenes_parser(d, nusc_c_data, ARCH)
                    res_c[f'{cond}/{sev}'] = eval_target_condition(
                        model, nusc_parser, proj, device, W0, protos_clean,
                        args, label=lab, cond_name=f'nuscenes_c_{cond}_{sev}', bal=bal)
                    os.makedirs(os.path.dirname(args.out), exist_ok=True)
                    with open(args.out, 'w') as fh:
                        json.dump(results, fh, indent=2, default=float)
                    print(f"\n[checkpoint] extractor {lab} NuScenes-C {cond}/{sev} done, "
                          f"saved to {args.out}")

    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
