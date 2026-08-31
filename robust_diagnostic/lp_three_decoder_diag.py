"""lp_three_decoder_diag.py: the three-decoder numbers for the linear-probe
training work (docs/lin_probe_training/validation.md).

Evaluates DGLSS++ (or any frozen extractor) on KITTI-C with THREE classifiers,
all reading the same frozen 128-d features:
  no-HDC    the model's own trained 1x1 conv head (softmax over z8)
  prototype mean binarized code per class, cosine decode (the HDC prototype
            classifier)
  linear    ridge probe W = (X^T X + lam I)^-1 X^T Y on the binarized codes,
            argmax decode (the HDC linear classifier)

Both HDC decoders are fit on CLEAN features only (zero-shot protocol) and
evaluated on each condition at each severity (default light/moderate/heavy),
with the 3-severity mean per condition and the clean eval. Per-class IoU is
reported so the gap can be inspected. This establishes the reference numbers
for the "linear classifier consistently outperforms on every condition" claim.

Usage:
  uv run python robust_diagnostic/lp_three_decoder_diag.py \
    --path_b robust_diagnostic/logs/supcon_vib_dglsspp \
    --method_b supcon_vib_dglsspp --label dglsspp \
    --out robust_diagnostic/logs/lp_three_decoder_dglsspp.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
CONDS_ALL = ["fog", "crosstalk", "snow", "wet_ground", "incomplete_echo",
             "beam_missing", "motion_blur", "cross_sensor"]


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                  labels=data["labels"], color_map=data["color_map"], learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"], sensor=arch["dataset"]["sensor"],
                  max_points=arch["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract_full(model, parser, device, num_frames=100):
    """Frozen features z8, the model's OWN softmax (no-HDC head), and labels."""
    feats, preds, lbls = [], [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device); labels = batch[2].to(device).view(-1); mask = (batch[1].to(device) > 0).view(-1)
            out = model(in_vol)
            pred = out[0]                             # softmax of the trained 1x1 conv head
            z8 = out[2] if len(out) == 3 else out[1]  # the 128-d bottleneck
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            pf = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])[mask]
            feats.append(zf.cpu()); preds.append(pf.cpu()); lbls.append(labels[mask].cpu())
    return torch.cat(feats), torch.cat(preds), torch.cat(lbls)


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


def per_class_iou(preds, lbls, nc=NUM_CLASSES):
    present = set(lbls.tolist())
    ious = {}
    for c in range(1, nc):
        if c not in present:
            continue
        tp = int(((preds == c) & (lbls == c)).sum().item())
        fp = int(((preds == c) & (lbls != c)).sum().item())
        fn = int(((preds != c) & (lbls == c)).sum().item())
        denom = tp + fp + fn
        ious[str(c)] = tp / denom if denom > 0 else 0.0
    return ious


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
    ap.add_argument("--fit_clean", type=int, default=30000, help="cap on clean points used to fit linear+proto")
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--proj_dim", type=int, default=10000)
    ap.add_argument("--conds", type=str, default=",".join(CONDS_ALL))
    ap.add_argument("--sevs", type=str, default="light,moderate,heavy")
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
    cf, cp, cl = extract_full(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=cf.shape[1], dim_out=args.proj_dim, device=device)
    results = {'label': args.label, 'method': args.method_b, 'sevs': sevs,
               'fit_clean': args.fit_clean, 'val_size': args.val_size, 'lam': args.lam,
               'proj_dim': args.proj_dim, 'conds': {}}

    # split clean: first fit_clean points fit both HDC decoders, last val_size eval clean
    cf_fit = cf[:args.fit_clean]; cl_fit = cl[:args.fit_clean]
    Xc = hdc_codes(cf_fit, proj, device).float()
    W = ridge_fit_exact(Xc, onehot(cl_fit, NUM_CLASSES), args.lam, device).detach().cpu()
    protos, _ = build_prototypes(Xc, cl_fit)
    protos = protos.float()
    # the RAW 128-d ridge probe: same fitter, same fit set, input space only change.
    # Answers "does the HDC projection help or hurt the linear classifier" cleanly.
    W_raw = ridge_fit_exact(cf_fit.to(device).float(), onehot(cl_fit, NUM_CLASSES).to(device),
                            args.lam, device).detach().cpu()
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def eval_set(feats, preds, lbls, name):
        Xv = hdc_codes(feats[:args.val_size], proj, device).float()
        lv = lbls[:args.val_size]
        pv = preds[:args.val_size]
        zv = feats[:args.val_size].float()
        r = {'mIoU_no_hdc': compute_miou(pv.argmax(1), lv),
             'mIoU_proto': compute_miou((Xv @ protos.t()).argmax(1), lv),
             'mIoU_linear': compute_miou((Xv @ W).argmax(1), lv),
             'mIoU_raw_linear': compute_miou((zv @ W_raw).argmax(1), lv)}
        r['per_class_no_hdc'] = per_class_iou(pv.argmax(1), lv)
        r['per_class_proto'] = per_class_iou((Xv @ protos.t()).argmax(1), lv)
        r['per_class_linear'] = per_class_iou((Xv @ W).argmax(1), lv)
        r['per_class_raw_linear'] = per_class_iou((zv @ W_raw).argmax(1), lv)
        r['n'] = int(len(lv))
        print(f"  {name}: no-hdc {r['mIoU_no_hdc']:.3f} | proto {r['mIoU_proto']:.3f} | "
              f"lin {r['mIoU_linear']:.3f} | raw-lin {r['mIoU_raw_linear']:.3f}")
        del Xv
        return r

    t0 = tic()
    clean_res = eval_set(cf, cp, cl, "clean")
    results['clean'] = clean_res

    for cond in conds:
        cond_res = {'sevs': {}}
        for sev in sevs:
            cdir = os.path.join(args.kittic_dir, cond, sev)
            if not os.path.exists(cdir):
                print(f"  [{cond}/{sev}] dir missing, skipped")
                continue
            fd, pd, ld = extract_full(model, build_parser(cdir, DATA, ARCH), device, args.frames)
            cond_res['sevs'][sev] = eval_set(fd, pd, ld, f"{cond}/{sev}")
            del fd, pd, ld
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ev = [v for v in cond_res['sevs'].values()]
        if ev:
            for k in ('mIoU_no_hdc', 'mIoU_proto', 'mIoU_linear', 'mIoU_raw_linear'):
                cond_res[k + '_3sev_mean'] = float(sum(v[k] for v in ev) / len(ev))
        results['conds'][cond] = cond_res

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out} ({toc(t0):.0f}s)")
    print("\n=== THREE-DECODER READ ===")
    print("mIoU_no_hdc = the model's own trained head (no HDC).")
    print("mIoU_proto = mean binarized code per class, cosine decode.")
    print("mIoU_linear = ridge probe on the binarized codes.")
    print("mIoU_raw_linear = ridge probe on the RAW 128-d features (same fitter,")
    print("  same fit set; the input space is the only change).")
    print("  mIoU_linear vs mIoU_raw_linear answers 'does the HDC projection help")
    print("  or hurt the linear classifier' on THIS encoder.")
    print("Both HDC decoders fit on clean only (zero-shot). 3-sev mean per condition.")


if __name__ == "__main__":
    main()
