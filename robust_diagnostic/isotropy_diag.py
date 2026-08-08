"""isotropy_diag.py: small-scale train + isotropy analysis of the HDC-input space.

Tests the working hypothesis documented in docs/robust_details.md: DGLSS / DGLSS++
collapse the HDC decode because their correlation-consistency constraints allow an
ANISOTROPIC (highly directional, low-rank) 128D bottleneck, which saturates the HDC
sign-projection pathway; our SupCon-uniformity space stays angularly ISOTROPIC and
survives it.

Arms (all at equal budget, VIB-free for the DGLSS methods):
  - supcon_vib:             SupCon + VIB (our method)
  - vib:                    VIB only (isolates VIB's isotropy contribution)
  - supcon_vib_dglss:       SIFC + SCC on the 128D bottleneck
  - supcon_vib_dglsspp:     GMSIFC + LSCC on the 128D bottleneck
  - supcon_vib_dglss_enc:   paper-faithful attachment: SIFC on encoder stage x_4,
                            SCC on the decoded bottleneck

Pipeline per arm:
  1. Train at small scale (cutoff * epochs) on sequence 08.
  2. Extract the 128D bottleneck features (the exact space the HDC random projection
     + sign binarization consumes) on clean + all 8 SemanticKITTI-C conditions.
  3. Isotropy analysis on each feature set:
       - participation ratio, top-5 variance fraction, log10 condition number
         (covariance spectrum: PR ~ 128 = isotropic, PR ~ k = low-rank/anisotropic)
       - mean |cos| over pairs (angular uniformity on the unit sphere)
       - HDC dead-coordinate fraction + mean pairwise Hamming distance over the REAL
         Bernoulli random projection (the collapse mechanism: dead-frac -> 1 and
         Hamming -> 0 when a low-rank space with a dominant mean saturates the codes)
  4. HDC decode sanity: prototypes built on clean, zero-shot mIoU on every condition
     (the ground-truth "does the HDC model collapse" measure).

Default scale is tuned for ~10-12 hours on the medium GPU (10 epochs at 10% data for
all arms). The DGLSS methods are slower than supcon_vib because of the affinity
aggregation.

Usage:
  uv run python robust_diagnostic/isotropy_diag.py
  uv run python robust_diagnostic/isotropy_diag.py --methods supcon_vib --epochs 2 --cutoff 0.05
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

CONDS = ['fog', 'crosstalk', 'snow', 'wet_ground', 'incomplete_echo',
         'beam_missing', 'motion_blur', 'cross_sensor']
# The comparison arms. supcon_vib is the SupCon+VIB reference; vib isolates VIB (the
# isotropy-attribution control); the three DGLSS arms are VIB-free:
#   - supcon_vib_dglss:      SIFC + SCC on the 128D bottleneck (the HDC-input space)
#   - supcon_vib_dglsspp:    GMSIFC + LSCC on the bottleneck
#   - supcon_vib_dglss_enc:  the paper-faithful attachment: SIFC on the deepest encoder
#                            stage x_4, SCC on the decoded bottleneck
METHODS = ['supcon_vib', 'vib', 'supcon_vib_dglss', 'supcon_vib_dglsspp', 'supcon_vib_dglss_enc']


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract_features(model, parser, device, num_frames=50):
    """Returns (feat, lbl) 128D bottleneck features per frame."""
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


def isotropy_metrics(z, proj, n_cov=50000, n_cos=20000, n_ham=20000, seed=42):
    """Anisotropy spectrum + HDC sign-code diversity of the 128D bottleneck features.

    The spectrum (PR / top5 / condition number) is computed on the CENTERED features
    and measures the effective rank of the space. The HDC-diversity metrics (dead
    coordinate fraction, pairwise Hamming) are computed on the RAW features, exactly
    as the HDC random projection + sign binarization consumes them: a low-rank space
    combined with a dominant shared mean direction saturates most sign coordinates
    (dead-coordinate fraction -> 1, Hamming -> 0), which is the observed DGLSS
    collapse mechanism. The mean-fraction reports how dominant that shared direction is.
    """
    z = z[:n_cov].to(proj.device)
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / len(zc)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)          # ascending
    tr = eig.sum()
    pr = float((tr ** 2) / (eig ** 2).sum()) if tr > 0 else 0.0
    top5 = float(eig[-5:].sum() / tr) if tr > 0 else 0.0
    cond = float((eig[-1] / (eig[0] + 1e-9)).item()) if eig[-1] > 0 else 0.0

    torch.manual_seed(seed)
    u = F.normalize(zc, p=2, dim=1)
    uu = u[torch.randperm(len(u))[:n_cos]]
    sim = (uu @ uu.T).abs()
    off = sim.numel() - uu.shape[0]
    mean_cos = float(((sim.sum() - uu.shape[0]) / off).item())

    mean_frac = float((z.mean(dim=0).norm() / z.norm(dim=1).mean()).item())

    codes = torch.sign(z @ proj)                             # raw features, real HDC pathway
    frac_pos = (codes > 0).float().mean(dim=0)
    dead = float(((frac_pos < 0.05) | (frac_pos > 0.95)).float().mean().item())

    torch.manual_seed(seed)
    a = codes[torch.randperm(len(codes))[:n_ham]]
    b = codes[torch.randperm(len(codes))[:n_ham]]
    hamming = float((1.0 - (a == b).float().mean(dim=1)).mean().item())

    return {'n': int(len(z)), 'pr': pr, 'top5_frac': top5, 'log10_cond': float(np.log10(cond + 1e-9)),
            'mean_abs_cos': mean_cos, 'mean_frac': mean_frac,
            'hdc_dead_frac': dead, 'hdc_hamming': hamming}


def proto_miou(feats, lbls, base_protos, proto_lbls, proj, device):
    feats_d = feats.to(device)
    protos = F.normalize(base_protos, p=2, dim=1)
    sims = []
    for start in range(0, len(feats_d), 50000):
        hc = F.normalize(torch.sign(feats_d[start:start + 50000] @ proj), p=2, dim=1)
        sims.append(hc @ protos.T)
    sims = torch.cat(sims, dim=0)
    return compute_miou(proto_lbls[sims.argmax(dim=1)], lbls.to(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--log_dir", type=str, default="robust_diagnostic/logs")
    parser.add_argument("--epochs", type=int, default=12,
                        help="epochs per method at --cutoff data (~10h total for the 3 methods)")
    parser.add_argument("--cutoff", type=float, default=0.1, help="fraction of training data per epoch")
    parser.add_argument("--frames", type=int, default=50, help="frames per condition for evaluation")
    parser.add_argument("--conditions", type=str, default="",
                        help="comma-separated subset; default = all 8")
    parser.add_argument("--ref_load_path", type=str, default="",
                        help="optional existing checkpoint (e.g. logs/med_pretrain_supcon_vib) "
                             "to analyze without retraining")
    parser.add_argument("--ref_method", type=str, default="supcon_vib")
    args = parser.parse_args()

    DATA = yaml_load(args.config)
    ARCH = yaml_load(args.arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    conds = [c.strip() for c in args.conditions.split(',')] if args.conditions else CONDS
    os.makedirs(args.log_dir, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    if args.ref_load_path:
        methods.append('__ref__')
    results = {}

    for method in methods:
        if method == '__ref__':
            load_path, mname = args.ref_load_path, args.ref_method
            print(f"\n{'='*80}\n=== Analyzing reference checkpoint {mname} ({load_path}) ===\n{'='*80}")
            trainer = GenTrainer(ARCH, DATA, args.kitti_dir, load_path, path=load_path, method=mname)
            label = f"{mname}[ref]"
        else:
            log_dir = os.path.join(args.log_dir, method)
            os.makedirs(log_dir, exist_ok=True)
            print(f"\n{'='*80}\n=== Training {method} (cutoff {args.cutoff}, {args.epochs} epochs) ===\n{'='*80}")
            trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, method=method,
                                 cutoff_percent=args.cutoff)
            trainer.train(epochs=args.epochs)
            label = method

        model = trainer.model
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        print("Extracting clean...")
        clean_f, clean_l = extract_features(model, clean_parser, device, args.frames)
        print(f"  clean n {len(clean_f)}")

        clf = LogisticRegression(max_iter=1000)
        fit_n = min(100000, len(clean_f))
        clf.fit(clean_f[:fit_n].numpy(), clean_l[:fit_n].numpy())
        clean_lp = float(clf.score(clean_f[:fit_n].numpy(), clean_l[:fit_n].numpy()))

        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)

        rows = {}
        sets = [('clean', clean_parser)]
        for cond in conds:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            sets.append((cond, build_parser(cdir, DATA, ARCH)))

        header = (f"{'cond':<16} {'LP':>7} {'PR':>6} {'top5':>6} {'logcond':>8} "
                  f"{'|cos|':>6} {'meanF':>6} {'deadF':>6} {'hamm':>6} {'HDC-ioU':>8}")
        print(header)
        for name, p in sets:
            f, l = extract_features(model, p, device, args.frames)
            f = f.to(device)
            m = isotropy_metrics(f.cpu(), proj)
            lp = float(clf.score(f[:200000].cpu().numpy(), l[:200000].numpy()))
            hdc = proto_miou(f, l, base_protos, proto_lbls, proj, device)
            rows[name] = {**m, 'lp_acc': lp, 'hdc_zs_miou': hdc}
            print(f"{name:<16} {lp:>7.4f} {m['pr']:>6.1f} {m['top5_frac']:>6.3f} "
                  f"{m['log10_cond']:>8.2f} {m['mean_abs_cos']:>6.3f} "
                  f"{m['mean_frac']:>6.3f} {m['hdc_dead_frac']:>6.3f} "
                  f"{m['hdc_hamming']:>6.3f} {hdc:>8.4f}")
        results[label] = {'clean_lp': clean_lp, 'conditions': rows}

    out_path = os.path.join(args.log_dir, 'isotropy_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


def yaml_load(path):
    import yaml
    return yaml.safe_load(open(path, 'r'))


if __name__ == "__main__":
    main()
