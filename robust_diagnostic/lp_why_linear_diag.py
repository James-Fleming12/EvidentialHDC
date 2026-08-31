"""lp_why_linear_diag.py: WHY does the HDC linear classifier beat the prototype,
FULL-HARNESS protocol (paper-realistic; docs/lin_probe_training/validation.md).

Same protocol as the README / al_full_dataset_diag.py:
  - CLEAN FIT: reservoir over ALL clean frames (seed 7), cap 200k;
    spectral-exact ridge (ridge_fit_exact, lam 1e-3).
  - EVAL: FULL streaming decode of EVERY point of EVERY frame (~300M
    points/condition), default severity heavy (--sevs for the 3-sev mean).
  - GEOMETRY (P1-P4): computed on a reservoir of the condition stream
    (--geo_res, seed 3), representative not first-slice-biased.

Reports per condition/severity:
  mIoU_linear (code probe) / mIoU_proto / mIoU_raw_linear (128-d probe) /
  mIoU_no_hdc, per-class gap, and where the code probe and the prototype
  disagree, P(linear right | disagree) vs P(proto right | disagree).
  gap_code_minus_raw = does the HDC projection HELP (+)/HURT (-) the linear
  classifier on this encoder (the README-consistent answer).

Feature-space diagnostics on the condition reservoir:
  P1 isotropy of the 128-d features (participation ratio, top-5 variance)
  P2 code diversity (dead-coordinate fraction, mean pairwise code cosine)
  P3 prototype centroid separation (off-diagonal mean cosine in 10000-d)
  P4 per-class mean shift clean->cond and within-class dispersion
  P5 clean reference: the linear-vs-proto gap on clean (static space property
     vs corruption collapse)

Usage:
  uv run python robust_diagnostic/lp_why_linear_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp \
    --out robust_diagnostic/logs/lp_why_linear_dglsspp.json
"""
import os, sys, time, argparse, json, yaml
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
    buf_f = torch.zeros(cap, 128)
    buf_l = torch.zeros(cap, dtype=torch.long)
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
        n += m
        del zf, labels
    n = min(n, cap)
    return buf_f[:n], buf_l[:n]


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
    def __init__(self, nc, fixed=False):
        self.nc = nc
        self.fixed = fixed
        self.tp = torch.zeros(nc); self.fp = torch.zeros(nc); self.fn = torch.zeros(nc)
        self.present = torch.zeros(nc, dtype=torch.bool); self.n = 0

    def update(self, preds, lbls):
        p = preds.long(); l = lbls.long()
        for c in range(1, self.nc):
            pc = (p == c); lc = (l == c)
            self.tp[c] += (pc & lc).sum().item()
            self.fp[c] += (pc & ~lc).sum().item()
            self.fn[c] += (~pc & lc).sum().item()
            self.present[c] |= lc.any().item()
        self.n += len(l)

    def miou(self):
        ious = []
        for c in range(1, self.nc):
            if (not self.fixed) and (not self.present[c]):
                continue
            d = self.tp[c] + self.fp[c] + self.fn[c]
            ious.append(float(self.tp[c] / d) if d > 0 else 0.0)
        return float(sum(ious) / len(ious)) if ious else 0.0

    def per_class(self):
        ious = {}
        for c in range(1, self.nc):
            if (not self.fixed) and (not self.present[c]):
                continue
            d = self.tp[c] + self.fp[c] + self.fn[c]
            ious[str(c)] = float(self.tp[c] / d) if d > 0 else 0.0
        return ious


class DisagreeAccum:
    """lin vs proto disagreement, accumulated streaming."""
    def __init__(self):
        self.n = 0; self.n_dis = 0
        self.lin_right_dis = 0; self.pro_right_dis = 0; self.both_wrong_dis = 0

    def update(self, lin_p, pro_p, lbls):
        dis = lin_p != pro_p
        lin_r = lin_p == lbls; pro_r = pro_p == lbls
        self.n_dis += int(dis.sum().item())
        self.lin_right_dis += int((dis & lin_r).sum().item())
        self.pro_right_dis += int((dis & pro_r).sum().item())
        self.both_wrong_dis += int(((dis & ~lin_r) & ~pro_r).sum().item())
        self.n += len(lbls)

    def summary(self):
        if self.n_dis == 0:
            return {'n_disagree': 0}
        return {'n_disagree': self.n_dis,
                'P_linear_right_given_disagree': self.lin_right_dis / self.n_dis,
                'P_proto_right_given_disagree': self.pro_right_dis / self.n_dis,
                'P_both_wrong_given_disagree': self.both_wrong_dis / self.n_dis}


def stream_decode_why(model, parser, proj, device, decoders, exclude=None, max_frames=0, chunk=100000, nc=17):
    accs = {name: ConfAccum(nc) for name in decoders}
    dis = DisagreeAccum()
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
            chunk_preds = {}
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
                chunk_preds[name] = (preds, lbls)
            dis.update(chunk_preds['linear'][0], chunk_preds['proto'][0], chunk_preds['linear'][1])
            for name, (preds, lbls) in chunk_preds.items():
                accs[name].update(preds, lbls)
            del codes
        del zf, labels
    return accs, dis


def isotropy(z):
    zc = z - z.mean(0)
    S = (zc.t() @ zc) / len(zc)
    ev = torch.linalg.eigvalsh(S.double()).clamp(min=0)
    s = ev.sum().item()
    pr = (s * s) / ((ev ** 2).sum().item()) if s > 1e-12 else None
    top5 = ev[-5:].sum().item() / s if s > 1e-12 else None
    return pr, top5


def dead_coords(codes):
    if len(codes) < 2:
        return None
    const = (codes > 0).all(0) | (codes < 0).all(0)
    return float(const.float().mean().item())


def code_div(codes, seed=1, sub=5000):
    n = min(sub, len(codes))
    torch.manual_seed(seed)
    idx = torch.randperm(len(codes))[:n]
    nz = F.normalize(codes[idx].float(), dim=1)
    return float((nz @ nz.t()).mean().item())


def offdiag_mean_cos(means):
    mn = F.normalize(means.float(), dim=1)
    G = mn @ mn.t()
    mask = ~torch.eye(G.shape[0], dtype=torch.bool)
    return float(G[mask].mean().item())


def class_mean_shift(clean_means, cond_means, classes):
    shifts = {}
    for c in classes:
        a = F.normalize(clean_means[c].float().unsqueeze(0), dim=1)
        b = F.normalize(cond_means[c].float().unsqueeze(0), dim=1)
        shifts[str(c)] = float((1 - (a @ b.t())).item())
    return shifts


def class_dispersion(codes, lbls, means, classes, sub=10000):
    disp = {}
    for c in classes:
        m = lbls == c
        if int(m.sum().item()) == 0:
            continue
        pts = codes[m][:sub].float()
        nz = F.normalize(pts, dim=1)
        mc = F.normalize(means[c].float().unsqueeze(0), dim=1)
        disp[str(c)] = float((nz @ mc.t()).mean().item())
    return disp


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
    ap.add_argument("--map19", type=int, default=0,
                    help="1 = evaluate on GeoID's exact 19-class map (semantic-kitti-19.yaml), "
                         "fixed-19 mIoU convention, no-HDC decoder dropped (merged head)")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = ALL frames of seq 08 (paper protocol)")
    ap.add_argument("--clean_fit_n", type=int, default=200000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--proj_dim", type=int, default=10000)
    ap.add_argument("--geo_res", type=int, default=200000, help="reservoir cap for the geometry diagnostics")
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--sevs", type=str, default="heavy", help="comma-separated; default heavy = README protocol")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.map19:
        args.config = "config/labels/semantic-kitti-19.yaml"
    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    NC = len(DATA["learning_map_inv"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} | config {args.config} | NC {NC} | map19 {args.map19}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    sevs = [s.strip() for s in args.sevs.split(',') if s.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    proj = get_hdc_projection(dim_in=128, dim_out=args.proj_dim, device=device)
    results = {'label': args.label, 'method': args.method_b, 'sevs': sevs,
               'config': args.config, 'map19': args.map19, 'nc': NC,
               'clean_fit_n': args.clean_fit_n, 'max_frames': args.max_frames,
               'lam': args.lam, 'proj_dim': args.proj_dim, 'geo_res': args.geo_res,
               'conds': {}}

    t0 = tic()
    print(f"=== clean fit (reservoir {args.clean_fit_n}) ===")
    cf, cl = reservoir_collect(stream_full(model, clean_parser, device, args.max_frames, progress="clean"),
                               args.clean_fit_n, 7)
    print(f"  clean reservoir: {len(cf)} points ({toc(t0):.0f}s)")
    Xc = hdc_codes(cf, proj, device).float()
    W0 = ridge_fit_exact(Xc, onehot(cl, NC), args.lam, device).detach().cpu()
    protos, _ = build_prototypes(Xc, cl, NC)
    protos = protos.float()
    W_raw = ridge_fit_exact(cf.float().to(device), onehot(cl, NC).to(device),
                            args.lam, device).detach().cpu()
    # clean class means in the CODE space (same space as the condition means
    # used by mean_shift / proto_pair_cos / dispersion below)
    present_classes = sorted(set(cl.tolist()) & set(range(1, NC)))
    clean_means = torch.zeros(NC, Xc.shape[1])
    for c in present_classes:
        m = cl == c
        if int(m.sum().item()) > 0:
            clean_means[c] = Xc[m].float().mean(dim=0)
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    decoders = {
        'proto': {'type': 'proto', 'protos': protos},
        'linear': {'type': 'w', 'W': W0},
        'raw_linear': {'type': 'raw_w', 'W': W_raw},
    }
    if not args.map19:
        decoders['no_hdc'] = {'type': 'head'}

    def geo(feats, lbls):
        Xg = hdc_codes(feats, proj, device).float()
        cond_means = torch.zeros(NC, Xg.shape[1])
        for c in present_classes:
            m = lbls == c
            if int(m.sum().item()) > 0:
                cond_means[c] = Xg[m].float().mean(dim=0)
        pr, top5 = isotropy(feats)
        return {'participation_ratio': pr, 'top5_var_frac': top5,
                'dead_coords': dead_coords(Xg), 'code_pair_cos': code_div(Xg),
                'proto_pair_cos': offdiag_mean_cos(cond_means),
                'mean_shift': class_mean_shift(clean_means, cond_means, present_classes),
                'dispersion': class_dispersion(Xg, lbls, cond_means, present_classes)}

    def eval_stream(parser, name, exclude=None, with_geo=False):
        t1 = tic()
        accs, dis = stream_decode_why(model, parser, proj, device, decoders,
                                      exclude=exclude, max_frames=args.max_frames, nc=NC)
        out = {'n': accs['linear'].n}
        for k, a in accs.items():
            out[k] = a.miou()
            out[k + '_per_class'] = a.per_class()
        out['gap_linear_minus_proto'] = out['linear'] - out['proto']
        out['gap_code_minus_raw'] = out['linear'] - out['raw_linear']
        out['disagree'] = dis.summary()
        pc_l = out['linear_per_class']; pc_p = out['proto_per_class']
        out['per_class_gap'] = {k: pc_l.get(k, 0.0) - pc_p.get(k, 0.0) for k in pc_l}
        if with_geo:
            print(f"  [{name}] geometry reservoir...")
            gf, gl = reservoir_collect(stream_full(model, parser, device, args.max_frames, progress="geo"),
                                       args.geo_res, 3)
            out['geo'] = geo(gf, gl)
            del gf, gl
        head_s = f"no-hdc {out['no_hdc']:.3f} | " if not args.map19 else ""
        print(f"  {name}: {head_s}proto {out['proto']:.3f} | "
              f"linear {out['linear']:.3f} | raw-lin {out['raw_linear']:.3f} | "
              f"gap-lin-proto {out['gap_linear_minus_proto']:+.3f} | "
              f"gap-code-raw {out['gap_code_minus_raw']:+.3f} | disagree {out['disagree']['n_disagree']} ({toc(t1):.0f}s)")
        return out

    clean_res = eval_stream(clean_parser, "clean", with_geo=True)
    results['clean'] = clean_res

    for cond in conds:
        cond_res = {'sevs': {}}
        for sev in sevs:
            cdir = os.path.join(args.kittic_dir, cond, sev)
            if not os.path.exists(cdir):
                print(f"  [{cond}/{sev}] dir missing, skipped")
                continue
            cond_res['sevs'][sev] = eval_stream(build_parser(cdir, DATA, ARCH), f"{cond}/{sev}", with_geo=True)
        ev = [v for v in cond_res['sevs'].values()]
        if ev:
            keys = ('proto', 'linear', 'raw_linear', 'gap_linear_minus_proto', 'gap_code_minus_raw')
            if not args.map19:
                keys = ('no_hdc',) + keys
            for k in keys:
                cond_res[k + '_sev_mean'] = float(sum(v[k] for v in ev) / len(ev))
        results['conds'][cond] = cond_res
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out} ({toc(t0):.0f}s)")
    print("\n=== WHY-LINEAR READ (full-harness protocol) ===")
    print("gap_code_minus_raw: does the HDC projection HELP (+)/HURT (-) the")
    print("   linear classifier per condition (README-consistent protocol).")
    print("gap_linear_minus_proto + disagreement: what the probe recovers that")
    print("   the prototype throws away, and P(linear right | disagree).")
    print("geo: isotropy / code diversity / centroid separation / mean shift +")
    print("   dispersion on a representative reservoir; P5 clean gap tells if")
    print("   the linear-vs-proto gap is a static space property or corruption.")


if __name__ == "__main__":
    main()
