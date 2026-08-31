"""lp_three_decoder_diag.py: the three/four-decoder numbers, FULL-HARNESS protocol
(paper-realistic; docs/lin_probe_training/validation.md).

Matches the README / al_full_dataset_diag.py protocol exactly:
  - CLEAN FIT: reservoir over ALL clean frames (seed 7), cap 200k
    (--clean_fit_n); spectral-exact ridge (ridge_fit_exact, lam 1e-3).
  - EVAL: FULL streaming decode of EVERY point of EVERY frame of seq 08
    (~300M points/condition), max_frames=0 = all frames.
  - SEVERITY: default heavy (the README tables are heavy); pass
    --sevs light,moderate,heavy for the 3-severity mean.
  - The clean-fit reservoir points are EXCLUDED from the clean eval.

Four decoders, all zero-shot (fit on clean only):
  no_hdc      the model's own trained 1x1 conv head (argmax of its softmax)
  proto       mean binarized code per class, cosine decode (the README R1)
  linear      ridge probe on the binarized codes (the README R4)
  raw_linear  the SAME ridge probe on the RAW 128-d features (input space is
              the only change -> answers 'does the HDC projection help or
              hurt the linear classifier' on this encoder)

Per-class IoU is accumulated streaming (ConfAccum, class 0 + absent excluded),
so the numbers are directly comparable to the README tables.

Usage:
  uv run python robust_diagnostic/lp_three_decoder_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp \
    --out robust_diagnostic/logs/lp_three_decoder_dglsspp.json
  # 3-severity average (AL-arc reporting) instead of heavy-only:
  #   --sevs light,moderate,heavy
"""
import os, sys, time, argparse, json, yaml
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection

NUM_CLASSES = 17
CONDS_ALL = ["fog", "crosstalk", "snow", "wet_ground", "incomplete_echo",
             "beam_missing", "motion_blur", "cross_sensor"]


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                  labels=data["labels"], color_map=data["color_map"], learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"], sensor=arch["dataset"]["sensor"],
                  max_points=arch["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)


def stream_full(model, parser, device, max_frames=0, progress=None, report=500):
    """Yield (zf, head_pred, labels, frame_idx) per frame, ALL frames unless
    max_frames > 0. head_pred = the model's OWN head argmax (no-HDC)."""
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
            pred = out[0]
            z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask].cpu()
            ph = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])[mask].argmax(1).cpu()
            yield zf, ph, labels[mask].cpu(), i


def reservoir_collect(stream, cap, seed):
    """True reservoir sampling over a frame stream (harness copy): memory
    bounded by cap, uniform over ALL points seen."""
    buf_f = torch.zeros(cap, 128)
    buf_l = torch.zeros(cap, dtype=torch.long)
    buf_k = torch.zeros(cap, 2, dtype=torch.long)
    n = 0
    g = torch.Generator().manual_seed(seed)
    for zf, ph, labels, fi in stream:
        m = len(zf)
        if m == 0:
            continue
        j = n + torch.arange(1, m + 1)
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
    y = torch.zeros(len(lbls), nc); y[torch.arange(len(lbls)), lbls.long()] = 1; return y


def ridge_fit_exact(X, Y, lam, device, chunk=50000):
    d = X.shape[1]; nc = Y.shape[1]
    S = torch.zeros(d, d, device=device); T = torch.zeros(d, nc, device=device)
    for s in range(0, len(X), chunk):
        Xc = X[s:s + chunk].to(device); Yc = Y[s:s + chunk].to(device)
        S += Xc.t() @ Xc; T += Xc.t() @ Yc
    A = S.double() + lam * torch.eye(d, dtype=torch.float64, device=device)
    return torch.linalg.solve(A, T.double()).float()


def build_prototypes(codes, lbls, nc=NUM_CLASSES):
    protos = torch.zeros(nc, codes.shape[1]); counts = torch.zeros(nc)
    for c in range(nc):
        m = lbls == c
        if int(m.sum().item()) > 0:
            protos[c] = codes[m].float().mean(dim=0)
            counts[c] = float(int(m.sum().item()))
    return F.normalize(protos, p=2, dim=1), counts


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
        return float(sum(ious) / len(ious)) if ious else 0.0

    def per_class(self):
        ious = {}
        for c in range(1, NUM_CLASSES):
            if not self.present[c]:
                continue
            d = self.tp[c] + self.fp[c] + self.fn[c]
            ious[str(c)] = float(self.tp[c] / d) if d > 0 else 0.0
        return ious


def stream_decode_four(model, parser, proj, device, decoders, exclude=None, max_frames=0, chunk=100000):
    """Stream-decode EVERY point with each decoder; returns {name: ConfAccum}.
    decoders: {'no_hdc': {'type':'head'}, 'proto': {'type':'proto',
    'protos':(K,D)}, 'linear': {'type':'w','W':(D,K)},
    'raw_linear': {'type':'raw_w','W':(128,K)}}."""
    accs = {name: ConfAccum() for name in decoders}
    prep = {}
    for name, dec in decoders.items():
        t = dec['type']
        if t == 'w':
            prep[name] = ('w', dec['W'].to(device))
        elif t == 'raw_w':
            prep[name] = ('raw_w', dec['W'].to(device))
        elif t == 'proto':
            prep[name] = ('proto', dec['protos'].to(device))
        else:
            prep[name] = ('head',)
    for zf, ph, labels, fi in stream_full(model, parser, device, max_frames, progress="decode"):
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
            zch = zf[s:e].to(device)
            for name, p in prep.items():
                if p[0] == 'head':
                    preds = ph[s:e]
                elif p[0] == 'w':
                    preds = (codes @ p[1]).argmax(1).cpu()
                elif p[0] == 'raw_w':
                    preds = (zch @ p[1]).argmax(1).cpu()
                else:
                    sims = F.normalize(codes, p=2, dim=1) @ p[1].t()
                    preds = sims.argmax(1).cpu()
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
    ap.add_argument("--max_frames", type=int, default=0, help="0 = ALL frames of seq 08 (paper protocol)")
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--proj_dim", type=int, default=10000)
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--sevs", type=str, default="heavy", help="comma-separated; default heavy = README protocol")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    sevs = [s.strip() for s in args.sevs.split(',') if s.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    proj = get_hdc_projection(dim_in=128, dim_out=args.proj_dim, device=device)
    results = {'label': args.label, 'method': args.method_b, 'sevs': sevs,
               'clean_fit_n': args.clean_fit_n, 'max_frames': args.max_frames,
               'lam': args.lam, 'proj_dim': args.proj_dim, 'conds': {}}

    # ---- clean fit: reservoir over ALL clean frames (seed 7, cap clean_fit_n) ----
    t0 = tic()
    print(f"=== clean fit (reservoir {args.clean_fit_n}) ===")
    cf, cl, ck = reservoir_collect(stream_full(model, clean_parser, device, args.max_frames, progress="clean"),
                                   args.clean_fit_n, 7)
    print(f"  clean reservoir: {len(cf)} points ({toc(t0):.0f}s)")
    Xc = hdc_codes(cf, proj, device).float()
    W0 = ridge_fit_exact(Xc, onehot(cl, NUM_CLASSES), args.lam, device).detach().cpu()
    protos, _ = build_prototypes(Xc, cl)
    protos = protos.float()
    W_raw = ridge_fit_exact(cf.float().to(device), onehot(cl, NUM_CLASSES).to(device),
                            args.lam, device).detach().cpu()
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  W0 (code), protos, W_raw (128-d) fit done ({toc(t0):.0f}s)")

    # exclude the clean-fit reservoir from the clean eval (no train-point optimism)
    ex_clean = defaultdict(list)
    for f, i in ck.tolist():
        ex_clean[f].append(i)
    ex_clean = {f: torch.tensor(sorted(s), dtype=torch.long) for f, s in ex_clean.items()}

    decoders = {
        'no_hdc': {'type': 'head'},
        'proto': {'type': 'proto', 'protos': protos},
        'linear': {'type': 'w', 'W': W0},
        'raw_linear': {'type': 'raw_w', 'W': W_raw},
    }

    def eval_stream(parser, name, exclude=None):
        t1 = tic()
        accs = stream_decode_four(model, parser, proj, device, decoders,
                                  exclude=exclude, max_frames=args.max_frames)
        out = {'n': accs['no_hdc'].n}
        for k, a in accs.items():
            out[k] = a.miou()
            out[k + '_per_class'] = a.per_class()
        print(f"  {name}: no-hdc {out['no_hdc']:.3f} | proto {out['proto']:.3f} | "
              f"linear {out['linear']:.3f} | raw-lin {out['raw_linear']:.3f} | n {out['n']} ({toc(t1):.0f}s)")
        return out

    clean_res = eval_stream(clean_parser, "clean", exclude=ex_clean)
    results['clean'] = clean_res

    for cond in conds:
        cond_res = {'sevs': {}}
        for sev in sevs:
            cdir = os.path.join(args.kittic_dir, cond, sev)
            if not os.path.exists(cdir):
                print(f"  [{cond}/{sev}] dir missing, skipped")
                continue
            cond_res['sevs'][sev] = eval_stream(build_parser(cdir, DATA, ARCH), f"{cond}/{sev}")
        ev = [v for v in cond_res['sevs'].values()]
        if ev:
            for k in ('no_hdc', 'proto', 'linear', 'raw_linear'):
                cond_res[k + '_sev_mean'] = float(sum(v[k] for v in ev) / len(ev))
        results['conds'][cond] = cond_res
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out} ({toc(t0):.0f}s)")
    print("\n=== THREE-DECODER READ (full-harness protocol) ===")
    print("mIoU_linear vs mIoU_raw_linear = 'does the HDC projection help/hurt")
    print("   the linear classifier' on THIS encoder, same fitter + fit set.")
    print("mIoU_linear (R4) vs mIoU_proto (R1) vs mIoU_no_hdc per condition;")
    print("   full-dataset eval (~300M pts/cond), 200k clean reservoir fit,")
    print("   spectral-exact ridge -- directly comparable to the README tables.")


if __name__ == "__main__":
    main()
