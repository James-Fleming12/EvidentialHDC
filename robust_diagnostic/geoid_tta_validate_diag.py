"""geoid_tta_validate_diag.py: does GeoID's TTA mechanism work on OUR range-view
network? (docs/causal_rep_learning.md, docs/inv_rep_learning.md)

GeoID's test-time training is NOT semantic pseudo-labelling. It is:
  1. inject synthetic displaced points (geoid_displace) -> augmented cloud,
  2. BiUPF: using the FROZEN source inlier scores c_src, retain real points with
     c_src >= tau_r and synthetic points with c_src <= 1 - tau_r,
  3. a few gradient steps of the inlier-discrimination BCE on the retained set,
     updating the shared ENCODER + the geoid head (the SEG head is kept frozen).

This validates whether that mechanism produces a usable TTA signal on the
range-view encoder, and whether it beats the frozen zero-shot. It is
parameterized (--path_b / --method / --arch) so it can run on a 6.8M geoid
model now and on the ~38M geoid-cenet38 (senet-2048p-w38) later for a direct
capacity comparison.

Metrics: the model's OWN seg head mIoU (the decoder GeoID adapts and evaluates),
frozen vs adapted, per condition/severity, plus the BiUPF retained fraction.
The seg head is NOT updated; only the encoder + geoid head are, so any gain is
the TTA's feature-space effect.

Usage:
  uv run python robust_diagnostic/geoid_tta_validate_diag.py \
    --path_b robust_diagnostic/logs/geoid_cenet38_19cls --method geoid \
    --arch config/arch/senet-2048p-w38.yml --conds fog,crosstalk,wet_ground \
    --out robust_diagnostic/logs/geoid_tta_validate.json
"""
import os, sys, time, argparse, json, yaml, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer

NUM_CLASSES = 19  # fixed-19 map (GeoID's convention; updated from the config)
CONDS_ALL = ["fog", "crosstalk", "snow", "wet_ground", "incomplete_echo",
             "beam_missing", "motion_blur", "cross_sensor"]


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                  labels=data["labels"], color_map=data["color_map"], learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"], sensor=arch["dataset"]["sensor"],
                  max_points=arch["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)


def stream_inputs(parser, device, max_frames=0, progress=None, report=100):
    """Yield (in_vol_cpu, mask, labels_masked, frame_idx) per frame, ALL frames
    unless max_frames > 0. labels are masked to the valid points to match the
    seg predictions (which are masked)."""
    for i, batch in enumerate(parser.get_train_set()):
        if max_frames > 0 and i >= max_frames:
            break
        if progress is not None and i % report == 0:
            print(f"  [{progress}] frame {i}...", flush=True)
        in_vol = batch[0].to(device).cpu()
        mask = (batch[1].to(device) > 0).view(-1).cpu()
        labels = batch[2].to(device).view(-1).cpu()
        yield in_vol, mask, labels[mask], i


class ConfAccum:
    """Per-class tp/fp/fn for the fixed-19 mean (absent classes count as 0)."""
    def __init__(self, nc):
        self.nc = nc
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
            d = self.tp[c] + self.fp[c] + self.fn[c]
            ious.append(float(self.tp[c] / d) if d > 0 else 0.0)
        return float(sum(ious) / len(ious)) if ious else 0.0


def seg_pred(model, in_vol, mask, device):
    """Model's own seg head argmax at the valid points."""
    model.eval()
    with torch.no_grad():
        out = model(in_vol.to(device))
        pred = out[0]  # softmax (B, C, H, W)
    pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])[mask]
    return pred.argmax(1).cpu()


def inlier_scores(model, in_vol, device):
    """Sigmoid of the geoid head logits on the given volume."""
    model.eval()
    with torch.no_grad():
        model(in_vol.to(device))
        logits = model._geoid_logits
    return torch.sigmoid(logits).cpu()


def biupf_retain(c_src, geo_lbl, tau_r):
    """GeoID BiUPF: keep real (geo_lbl==1) with c_src>=tau_r, synthetic
    (geo_lbl==0) with c_src<=1-tau_r; empty (-1) excluded."""
    real = (geo_lbl == 1) & (c_src >= tau_r)
    syn = (geo_lbl == 0) & (c_src <= (1.0 - tau_r))
    return real | syn


def tta_step(model, optimizer, scaler, aug, geo_lbl, retain, device):
    """One inlier-BCE gradient step on the retained points (encoder + geoid head)."""
    model.train()
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        model(aug.to(device))
        logits = model._geoid_logits
        logits_p = logits.permute(0, 2, 3, 1).reshape(-1)
        lbl_p = geo_lbl.permute(0, 2, 3, 1).reshape(-1).to(device)
        retain_p = retain.permute(0, 2, 3, 1).reshape(-1).to(device)
        sel = retain_p & (lbl_p >= 0)
        if not sel.any():
            return
        target = (lbl_p[sel] > 0).float().to(logits_p.dtype)
        loss = F.binary_cross_entropy_with_logits(logits_p[sel], target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def pseudo_step(model, optimizer, scaler, in_vol, mask, tau_p, device):
    """Pseudo-label self-training step: CE on the model's own confident
    predictions (conf > tau_p), updating the whole model. Works on ANY
    segmentation model (no geoid head needed)."""
    model.train()
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        out = model(in_vol.to(device))
        pred = out[0]
        m = mask.to(device)
        sm = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])[m]
        conf = sm.max(1).values
        gate = conf > tau_p
        if not gate.any():
            return
        pseudo = sm.argmax(1)[gate]
        loss = F.nll_loss(torch.log(sm[gate].clamp(min=1e-8)), pseudo)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


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
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-19.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p-w38.yml")
    ap.add_argument("--path_b", type=str, required=True, help="geoid checkpoint dir")
    ap.add_argument("--method", type=str, default="geoid", help="geoid or supcon_vib_geoid")
    ap.add_argument("--tta_mode", type=str, default="geoid",
                    help="geoid = GeoID inlier-BCE + BiUPF (needs a geoid head); "
                         "pseudo = confidence-gated pseudo-label self-training (any model)")
    ap.add_argument("--tau_p", type=float, default=0.9, help="pseudo-label confidence gate")
    ap.add_argument("--label", type=str, default="geoid")
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--sevs", type=str, default="heavy")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames (eval); TTA bounded by --tta_frames")
    ap.add_argument("--tta_frames", type=int, default=100, help="first N frames used for the TTA update")
    ap.add_argument("--tta_steps", type=int, default=3, help="gradient steps per scan")
    ap.add_argument("--tta_lr", type=float, default=1e-3, help="TTA learning rate (GeoID uses 0.001)")
    ap.add_argument("--tau_r", type=float, default=0.6, help="BiUPF reliability threshold (real: c_src>=tau_r, synth: c_src<=1-tau_r)")
    ap.add_argument("--p", type=float, default=0.05, help="synthetic displacement fraction")
    ap.add_argument("--val_size", type=int, default=200000, help="cap on eval points per condition")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    NC = len(DATA["learning_map_inv"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} | config {args.config} | NC {NC} | method {args.method}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    sevs = [s.strip() for s in args.sevs.split(',') if s.strip()]

    # load the geoid model + its displacement helper via the trainer
    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method)
    model = trainer.model
    model.eval()
    if args.tta_mode == 'geoid':
        # frozen source model for BiUPF (GeoID keeps the source encoder + geoid head frozen for the gate)
        frozen_model = copy.deepcopy(model)
        frozen_model.eval()
        for p in frozen_model.parameters():
            p.requires_grad_(False)
        # optimizer updates encoder + geoid head, NOT the seg head (GeoID keeps Phi_s frozen)
        frozen_p = set()
        for n, p in model.named_parameters():
            if 'semantic_output' in n or 'aux_head' in n:
                frozen_p.add(id(p))
        params = [p for n, p in model.named_parameters() if id(p) not in frozen_p]
    else:
        frozen_model = None
        params = [p for p in model.parameters()]
    optimizer = torch.optim.Adam(params, lr=args.tta_lr)
    scaler = torch.amp.GradScaler('cuda')
    print(f"  mode {args.tta_mode} | trainable params: {sum(p.numel() for p in params)/1e6:.2f}M")

    results = {'label': args.label, 'method': args.method, 'tta_mode': args.tta_mode, 'nc': NC,
               'tta_frames': args.tta_frames, 'tta_steps': args.tta_steps,
               'tta_lr': args.tta_lr, 'tau_r': args.tau_r, 'tau_p': args.tau_p, 'p': args.p,
               'conds': {}}

    for cond in conds:
        cond_res = {'sevs': {}}
        for sev in sevs:
            cdir = os.path.join(args.kittic_dir, cond, sev)
            if not os.path.exists(cdir):
                print(f"  [{cond}/{sev}] dir missing, skipped")
                continue
            parser = build_parser(cdir, DATA, ARCH)
            t0 = tic()
            # pass 1: frozen eval on the val slice (frames >= tta_frames)
            acc_f = ConfAccum(NC)
            for in_vol, mask, labels, i in stream_inputs(parser, device, args.max_frames, progress="frozen"):
                if i >= args.tta_frames:
                    preds = seg_pred(model, in_vol, mask, device)
                    acc_f.update(preds[:args.val_size], labels[:args.val_size])
            frozen = acc_f.miou()
            print(f"  [{cond}/{sev}] frozen seg mIoU {frozen:.3f} ({toc(t0):.0f}s)")

            # pass 2: TTA on the first tta_frames
            retained_tot = 0.0; retained_n = 0
            for in_vol, mask, labels, i in stream_inputs(parser, device, args.tta_frames, progress="tta"):
                if args.tta_mode == 'geoid':
                    aug, geo_lbl = trainer.geoid_displace(in_vol, min_dist=1, max_dist=3, p=args.p)
                    c_src = inlier_scores(frozen_model, aug, device)
                    retain = biupf_retain(c_src, geo_lbl, args.tau_r)
                    retained_tot += float(retain.float().mean().item()); retained_n += 1
                    for _ in range(args.tta_steps):
                        tta_step(model, optimizer, scaler, aug, geo_lbl, retain, device)
                else:
                    for _ in range(args.tta_steps):
                        pseudo_step(model, optimizer, scaler, in_vol, mask, args.tau_p, device)
            model.eval()
            retained_frac = retained_tot / max(1, retained_n) if args.tta_mode == 'geoid' else None
            print(f"  [{cond}/{sev}] TTA {args.tta_frames} frames x {args.tta_steps} steps "
                  f"(mode {args.tta_mode}" + (f", retained {retained_frac:.2f}" if retained_frac is not None else "") + f") ({toc(t0):.0f}s)")

            # pass 3: adapted eval on the val slice
            acc_a = ConfAccum(NC)
            for in_vol, mask, labels, i in stream_inputs(parser, device, args.max_frames, progress="adapted"):
                if i >= args.tta_frames:
                    preds = seg_pred(model, in_vol, mask, device)
                    acc_a.update(preds[:args.val_size], labels[:args.val_size])
            adapted = acc_a.miou()
            print(f"  [{cond}/{sev}] adapted seg mIoU {adapted:.3f} (delta {adapted - frozen:+.3f}) ({toc(t0):.0f}s)")

            cond_res['sevs'][sev] = {'frozen': frozen, 'adapted': adapted,
                                     'delta': adapted - frozen, 'retained_frac': retained_frac}
        results['conds'][cond] = cond_res
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(results, fh, indent=2, default=float)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("delta = adapted - frozen (seg-head mIoU, fixed-19 mean).")
    if args.tta_mode == 'geoid':
        print("positive delta -> GeoID's TTA mechanism (inlier-BCE + BiUPF) works on our")
        print("range-view encoder. delta ~ 0 or negative + low retained_frac -> the")
        print("inlier signal / BiUPF gate is not usable on our features.")
    else:
        print("positive delta -> confidence-gated pseudo-label self-training helps on our")
        print("features. delta ~ 0 or negative -> the pseudo-label signal is not usable")
        print("(consistent with the AL-arc label-free-gate finding).")
    print("Re-run with --path_b <geoid-cenet38 ckpt> --arch senet-2048p-w38.yml --tta_mode geoid")
    print("for the capacity-matched GeoID-TTA comparison.")


if __name__ == "__main__":
    main()
