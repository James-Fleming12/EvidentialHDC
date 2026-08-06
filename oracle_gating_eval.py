"""oracle_gating_eval.py: evaluation orchestrator.

HDC prototype evaluation and test-time adaptation (TTA) diagnostics for the
EvidentialHDC pipeline. The heavy lifting lives in two modules:
  - modules/oracle_core.py      : frozen-prototype decode, gated EMA ladder, autopsy
  - modules/tta_diagnostics.py  : the TTA iteration diagnostics (tta_iterations.md)
This file owns the CLI, data extraction, and the per-condition dispatch.
"""
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.HDC_utils import GATE_CFG

from modules.oracle_core import (
    get_hdc_projection,
    build_hdc_prototypes,
    condition_autopsy,
    evaluate_oracle_gating,
)
from modules.tta_diagnostics import (
    gate_sweep,
    prototype_rebalance,
    prior_correction_sweep,
    tta_oracle_decode,
    oracle_pool_sweep,
    iteration0_label_info,
    iteration0_update_diag,
    build_mv_views,
    iter1_pseudo_refine,
    iter2_balanced_reestimate,
    react_test,
    deep_label_analysis,
    VIEW_CONFIGS,
)

CORRUPTIONS = [
    'fog', 'snow', 'wet_ground', 'incomplete_echo', 
    'crosstalk', 'beam_missing', 'motion_blur', 'cross_sensor'
]

def main():
    parser = argparse.ArgumentParser("./oracle_gating_eval.py")
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--load_path", type=str, default="logs/med_pretrain_supcon_vib",
                        help="Dir containing a trained GenTrainer checkpoint (e.g. logs/micro_pretrain/supcon_vib)")
    parser.add_argument("--method", type=str, default="supcon_vib")
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=1000000,
                        help="Adaptation pool size (seeded uniform sample over all frames). "
                             "Must be large (Phase 13: a 20k pool is ~250x too small to "
                             "refine 10kD prototypes whose base comes from millions of points; "
                             "the update is a vectorized weighted class-mean, so large pools are cheap).")
    parser.add_argument("--u_th", type=float, default=GATE_CFG["u_th"])
    parser.add_argument("--u_coef", type=float, default=GATE_CFG["u_coef"])
    parser.add_argument("--z_th", type=float, default=GATE_CFG["z_th"])
    parser.add_argument("--z_coef", type=float, default=GATE_CFG["z_coef"])
    parser.add_argument("--corruptions", type=str, default="",
                        help="Comma-separated subset of the 8 corruptions (default: all)")
    parser.add_argument("--gate_sweep", action="store_true",
                        help="Run the in-memory artifact-gate sweep (Phase 23) and skip the "
                             "ladder: extract per-point signals once, sweep threshold space, "
                             "report acc/mIoU Pareto bands + oracle-loss bound + per-class IoU.")
    parser.add_argument("--rebalance", action="store_true",
                        help="Run update-side prototype rebalancing: the label-free gate "
                             "selects which points recompute each class prototype, and the "
                             "rebalanced prototypes are evaluated over the FULL scene. "
                             "Distinct from decode-side gating (retained-subset mIoU).")
    parser.add_argument("--prior_sweep", action="store_true",
                        help="Run the decision-level source-prior correction test "
                             "(README sec 5.2): score = kappa*cos + tau*log(pi_c) over "
                             "(tau, kappa) configs, full-scene acc + mIoU, "
                             "prediction-only (never in the gate or updates).")
    parser.add_argument("--tta_oracle", action="store_true",
                        help="Run the TTA battery + prototype-oracle bounds: naive EMA, "
                             "soft-dual-weight EMA, and BN-stat alignment (self-supervised), "
                             "plus full-label and artifact-free oracle prototypes from the "
                             "corrupted pool (true labels, multiple artifact filter configs), "
                             "all full-scene acc + mIoU.")
    parser.add_argument("--oracle_pool_sweep", action="store_true",
                        help="Reconcile the Phase 24.9 full-label oracle pool-size "
                             "discrepancy: full-label prototypes from the corrupted pool at "
                             "pool sizes 200k / 500k / 1M, same shared val subset, "
                             "full-scene acc + mIoU.")
    parser.add_argument("--iter0_label_info", action="store_true",
                        help="Iteration 0 diagnostic (tta_iterations.md): quantify what "
                             "information the labels give over the label-free TTA methods. "
                             "Per-class pool pseudo-label precision/recall + contamination "
                             "source, and per-class val IoU for zero-shot / naive EMA / "
                             "full-label oracle.")
    parser.add_argument("--iter0_update_diag", action="store_true",
                        help="Iteration 0 diagnostic: HOW the prototypes should be updated. "
                             "Distinguishes a gating problem (correct pseudo-assigned points "
                             "are informative but drowned out) from an overrun problem "
                             "(minority classes swamped by majority artifacts). Per-class "
                             "cosine of the naive / correct-subset prototype to the oracle "
                             "prototype, and val mIoU for zero-shot / naive / correct-subset "
                             "(perfect-gating bound) / full-label oracle.")
    parser.add_argument("--iter1_pseudo_refine", action="store_true",
                        help="Iteration 1 diagnostic: better label-free ASSIGNMENT sources "
                             "for the prototype re-estimate. Tests the 128D linear-probe "
                             "pseudo-labels and Multi-View Augmented Consensus (MVAC, "
                             "canonical D3CTTA-style views: point-cloud scale / yaw / "
                             "pitch / dropout, LP-probability and 10kD cosine-softmax "
                             "averages) against the 10kD zero-shot pseudo-labels and the "
                             "full-label oracle.")
    parser.add_argument("--iter2_balanced_reestimate", action="store_true",
                        help="Iteration 2 diagnostic: source-prior-balanced pseudo-assignment "
                             "(Sinkhorn-Knopp, SHOT diversity guardrail, no backprop) for the "
                             "prototype re-estimate. Forces the re-estimate pool's class "
                             "marginals to match the source frequencies, countering the "
                             "rare-class recall starvation (Iteration 0). Reports hard and "
                             "soft balanced re-estimates vs zero-shot / zs-pseudo / oracle.")
    parser.add_argument("--react_test", action="store_true",
                        help="ReAct test (Sun et al., NeurIPS 2021): clip the 128D feature "
                             "norms at thresholds [3,4,5,6,8,inf] before the HDC projection "
                             "+ Sign() binarization. Tests whether the fog/crosstalk magnitude "
                             "inflation overpowers the angular structure of the binarized "
                             "prototypes (BinCos 0.05-0.08). Reports frozen-prototype decode "
                             "acc/mIoU and the binarized clean<->fog mean cosine per threshold.")
    parser.add_argument("--deep_label_analysis", action="store_true",
                        help="Iteration 4 deep analysis: what the ground-truth labels carry "
                             "that the features and prototypes cannot derive. Per-class clean "
                             "vs corrupt geometry (centroid shift, norm inflation, tightness, "
                             "inter-class absorption) with a survivor/collapser split; "
                             "pseudo-label confusion + prototype contamination + per-class "
                             "re-estimate impact for the zs and LP sources; and AUROC of "
                             "every label-free signal for separating oracle-recovered from "
                             "oracle-stuck points. Can run for hours; use "
                             "--corruptions to scope it.")
    parser.add_argument("--autopsy", action="store_true",
                        help="Run the per-condition hyperspace/decode autopsy (Phase 24) for "
                             "each corruption and print the comparison table: artifact profile, "
                             "margin/norm stats, cosine shift, ellipticity, LP headroom, "
                             "binarized mean quality. Skips the ladder.")
    args, _ = parser.parse_known_args()
    
    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Sensor normalization constants for the multi-view geometric transforms (read from
    # the ARCH config, not the Parser instance, so it is version-robust).
    mv_means = torch.tensor(ARCH["dataset"]["sensor"]["img_means"], dtype=torch.float32).view(5, 1, 1).to(device)
    mv_stds = torch.tensor(ARCH["dataset"]["sensor"]["img_stds"], dtype=torch.float32).view(5, 1, 1).to(device)
    
    # Seed the full pipeline (extraction workers, point subsampling) so feature
    # extraction and the pool/val split are reproducible across runs.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    method = args.method
    load_path = args.load_path
    
    if not os.path.exists(load_path):
        print(f"Error: {load_path} not found.")
        return
        
    gate_cfg = {"u_th": args.u_th, "u_coef": args.u_coef,
                "z_th": args.z_th, "z_coef": args.z_coef}
    
    print(f"\nLoading Model: {method} from {load_path}")
    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, load_path, path=load_path, method=method)
    model = trainer.model
    model.eval()
    
    clean_parser = Parser(root=args.kitti_dir, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                          labels=DATA["labels"], color_map=DATA["color_map"], learning_map=DATA["learning_map"],
                          learning_map_inv=DATA["learning_map_inv"], sensor=ARCH["dataset"]["sensor"],
                          max_points=ARCH["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)
                          
    clean_loader = clean_parser.get_train_set()
    
    clean_feats, clean_lbls = [], []
    NUM_BATCHES = args.num_batches
    
    print("-> Extracting Clean Latents (100 Frames)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(clean_loader, total=NUM_BATCHES)):
            if i >= NUM_BATCHES: break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            clean_feats.append(z_flat.cpu())
            clean_lbls.append(labels[mask].cpu())
            
    clean_feats = torch.cat(clean_feats, dim=0)
    clean_lbls = torch.cat(clean_lbls, dim=0)
    print(f"   [Total Clean Points Extracted: {len(clean_feats)}]")
    
    # Train Linear Probe on 128D (to use as our confidence oracle)
    print("-> Training Linear Probe Oracle (128D on 100k points)...")
    clf = LogisticRegression(max_iter=1000)
    train_size = min(100000, len(clean_feats))
    clf.fit(clean_feats[:train_size].numpy(), clean_lbls[:train_size].numpy())
    
    probe_clean_acc = clf.score(clean_feats[:train_size].numpy(), clean_lbls[:train_size].numpy())
    print(f"   [Base] Linear Probe Accuracy (Clean): {probe_clean_acc:.4f}\n")
    
    # Build robust 10kD HDC base prototypes over all clean points
    print("-> Building 10kD HDC Clean Base Prototypes...")
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    base_protos, proto_lbls = build_hdc_prototypes(clean_feats, clean_lbls, proj, device=device)
    
    # 128D clean class means (for the gate sweep's label-free cos-to-prototype signal)
    clean_means128 = {}
    for c in proto_lbls.tolist():
        m = clean_feats[clean_lbls == c]
        if len(m) > 0:
            clean_means128[c] = F.normalize(m.mean(dim=0), p=2, dim=0)

    # Source class prior pi_c (clean class frequencies), aligned to proto_lbls order,
    # for the decision-level prior-correction test (prediction-only).
    lbl_arr = clean_lbls.numpy()
    pi = {c: float((lbl_arr == c).mean()) for c in proto_lbls.tolist()}
    prior_vec = torch.tensor([pi[c] for c in proto_lbls.tolist()], dtype=torch.float32)
    
    # Per-dimension clean feature statistics (for the BN-style test-time alignment probe)
    torch.manual_seed(42)
    sub_idx = torch.randperm(len(clean_feats))[:500000]
    clean_stats = (clean_feats[sub_idx].mean(dim=0),
                   clean_feats[sub_idx].std(dim=0) + 1e-6)
    
    # Clean-data control: adapting clean -> clean, no poison exists. A good gate must
    # stay ~= naive EMA here; any large degradation is over-gating (a gate fault).
    clean_control = None
    if len(clean_feats) >= args.pool_size + 100000:
        print("\n-> Running Clean-Data Gate Control (adapt clean -> clean, no poison)...")
        clean_control = evaluate_oracle_gating(base_protos, proto_lbls,
                                               clean_feats[:args.pool_size + 100000],
                                               clean_lbls[:args.pool_size + 100000],
                                               clf, proj, device,
                                               pool_size=args.pool_size, gate_cfg=gate_cfg)
        print("   -> Clean Ladder (gate should ~= naive_ema):")
        for k, v in clean_control['gated'].items():
            if not k.endswith('_cfg') and not k.endswith('_miou'):
                print(f"      {k:<16}: {v:.4f}")
    
    # We no longer need the massive clean_feats tensor
    if not args.deep_label_analysis:
        del clean_feats
        del clean_lbls
    
    corruptions = CORRUPTIONS if not args.corruptions else [c.strip() for c in args.corruptions.split(',')]
    all_results = {}
    out_path = os.path.join(load_path, "oracle_gating_results.json")
    if os.path.exists(out_path):
        try:
            with open(out_path, 'r') as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} existing corruption results from {out_path} (results will be merged)")
        except Exception:
            print("Warning: could not parse existing results file; starting fresh.")
    
    for corruption in corruptions:
        print(f"\n{'='*60}")
        print(f"Evaluating Corruption: {corruption.upper()}")
        print(f"{'='*60}")
        
        fog_dir = os.path.join(args.kittic_dir, corruption, 'heavy')
        if not os.path.exists(fog_dir):
            fog_dir = os.path.join(args.kittic_dir, corruption, 'moderate')
            
        corrupt_parser = Parser(root=fog_dir, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                            labels=DATA["labels"], color_map=DATA["color_map"], learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"], sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)
        
        corrupt_loader = corrupt_parser.get_train_set()
        
        corrupt_feats, corrupt_lbls, corrupt_depths = [], [], []
        if args.iter1_pseudo_refine:
            corrupt_views = [[] for _ in VIEW_CONFIGS]
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(corrupt_loader, total=NUM_BATCHES, desc=f"   Ext. {corruption}")):
                if i >= NUM_BATCHES: break
                in_vol = batch[0].to(device)
                labels = batch[2].to(device).view(-1)
                mask = (batch[1].to(device) > 0).view(-1)
                
                out_tuple = model(in_vol)
                if len(out_tuple) == 3:
                    _, _, z8 = out_tuple
                else:
                    _, z8 = out_tuple
                z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
                
                if args.iter1_pseudo_refine:
                    torch.manual_seed(0)
                    n = z_flat.shape[0]
                    idx = torch.randperm(n, device=z_flat.device)[: min(n, 40000)]
                    corrupt_feats.append(z_flat[idx].cpu())
                    corrupt_lbls.append(labels[mask][idx].cpu())
                    corrupt_depths.append(in_vol[:, 0, :, :].reshape(-1)[mask][idx].cpu())
                    corrupt_views[0].append(z_flat[idx].cpu())
                    for k, (vname, vfn) in enumerate(build_mv_views(batch, mv_means, mv_stds, device)):
                        if k == 0:
                            continue
                        _, _, z8v = model(vfn)
                        z_flat_v = z8v.permute(0, 2, 3, 1).reshape(-1, z8v.shape[1])[mask]
                        corrupt_views[k].append(z_flat_v[idx].cpu())
                else:
                    corrupt_feats.append(z_flat.cpu())
                    corrupt_lbls.append(labels[mask].cpu())
                    corrupt_depths.append(in_vol[:, 0, :, :].reshape(-1)[mask].cpu())  # range channel, same mask order
                
        corrupt_feats = torch.cat(corrupt_feats, dim=0)
        corrupt_lbls = torch.cat(corrupt_lbls, dim=0)
        corrupt_depths = torch.cat(corrupt_depths, dim=0)
        
        if args.iter1_pseudo_refine:
            corrupt_views = [torch.cat(v, dim=0) for v in corrupt_views]
            res = {}
            print("      -> Running Iteration-1 Pseudo-Label Refinement (LP + Multi-View Consensus)...")
            i1 = iter1_pseudo_refine(base_protos, proto_lbls, corrupt_feats, corrupt_views,
                                     corrupt_lbls, clf, proj, device)
            res['iter1_pseudo_refine'] = i1
            m = i1['metrics']
            pa = i1['pseudo_acc']
            print("   -> Full-scene mIoU: zero-shot {:.4f} | zs-pseudo-reest {:.4f} | "
                  "LP-pseudo {:.4f} | MVAC-LP {:.4f} | MVAC-proto {:.4f} | oracle {:.4f}"
                  .format(m['zero_shot']['miou'], m['zs_pseudo_reestimate']['miou'],
                          m['LP_pseudo']['miou'], m['MVAC_LP']['miou'],
                          m['MVAC_proto']['miou'], m['oracle']['miou']))
            print("   -> Pool pseudo-label accuracy: "
                  + " | ".join(f"{k} {v:.4f}" for k, v in pa.items()))
            sc = i1['self_check']
            print(f"   -> Self-check: class0 frac pool {sc['pool_class0_frac']:.3f} / "
                  f"val {sc['val_class0_frac']:.3f} | LP acc pool {sc['LP_acc_pool']:.4f} / "
                  f"val {sc['LP_acc_val']:.4f}")
            all_results[corruption] = res
            continue
        
        probe_corrupt_acc = clf.score(corrupt_feats[:train_size].numpy(), corrupt_lbls[:train_size].numpy())
        print(f"   -> 128D Linear Probe Accuracy: {probe_corrupt_acc:.4f}")
        
        if args.autopsy:
            res = {}
            print("      -> Running Condition Autopsy...")
            au = condition_autopsy(base_protos, proto_lbls, clean_means128,
                                   corrupt_feats, corrupt_lbls, proj, device,
                                   clf=clf, clean_stats=clean_stats,
                                   corrupt_depths=corrupt_depths)
            res['autopsy'] = au
            all_results[corruption] = res
            continue
        if args.gate_sweep:
            res = {}
            print("      -> Running Artifact-Gate Sweep (in-memory)...")
            gs = gate_sweep(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj, device,
                            clf=clf, clean_means128=clean_means128)
            res['gate_sweep'] = gs
            print("   -> Label-Free Gate Pareto (best mIoU per retention band, no oracle):")
            for b in gs['pareto_label_free']:
                cfg = b['cfg']
                desc = (f"norm<{cfg[0]} marg>={cfg[1]} cos1>={cfg[2]} "
                        f"cos128>={cfg[3]} conf>={cfg[4]}")
                print(f"      {b['band']:<8}: ret {b['retention']*100:5.1f}% | acc {b['acc']:.4f} | "
                      f"mIoU {b['miou']:.4f} | {desc}")
            print("   -> Gate-Sweep Pareto (best mIoU per retention band, incl. oracle):")
            for b in gs['pareto']:
                cfg = b['cfg']
                if cfg[0] == 'loss':
                    desc = f"loss<={cfg[1]}"
                else:
                    desc = (f"norm<{cfg[0]} marg>={cfg[1]} cos1>={cfg[2]} "
                            f"cos128>={cfg[3]} conf>={cfg[4]}")
                print(f"      {b['band']:<8}: ret {b['retention']*100:5.1f}% | acc {b['acc']:.4f} | "
                      f"mIoU {b['miou']:.4f} | {desc}")
            lb = gs['loss_band']
            if lb:
                print("   -> Oracle-Loss Gate Bound (label-free achievable if loss were learned):")
                for band, v in lb.items():
                    print(f"      {band:<8}: ret {v['retention']*100:5.1f}% | acc {v['acc']:.4f} | "
                          f"mIoU {v['miou']:.4f} | loss<={v['max_loss']}")
            best = gs['best']
            print(f"   -> Best config: mIoU {best['miou']:.4f} | acc {best['acc']:.4f} | "
                  f"ret {best['retention']*100:.1f}% | cfg {best['cfg']}")
            pc = gs['per_class_iou']
            if pc:
                print("   -> Per-class IoU at best config: "
                      + ", ".join(f"{c}:{v:.2f}" for c, v in sorted(pc.items())))
            all_results[corruption] = res
            continue
        if args.rebalance:
            res = {}
            print("      -> Running Update-Side Prototype Rebalancing...")
            rb = prototype_rebalance(base_protos, proto_lbls, corrupt_feats, corrupt_lbls,
                                     proj, device, clf=clf, clean_means128=clean_means128)
            res['rebalance'] = rb
            zs = rb['zero_shot']
            print(f"   -> Full-scene baseline: acc {zs['acc']:.4f} | mIoU {zs['miou']:.4f}")
            bf = rb['best_label_free']
            cfg = bf['cfg']
            print(f"   -> Best label-free rebalance (selection>={bf['selection']*100:.1f}%): "
                  f"full-scene acc {bf['acc']:.4f} | mIoU {bf['miou']:.4f} | "
                  f"selection {bf['selection']*100:.1f}% | "
                  f"norm<{cfg[1]} m>={cfg[2]} c1>={cfg[3]} c128>={cfg[4]} conf>={cfg[5]}")
            bo = rb['best_oracle']
            print(f"   -> Oracle-loss bound: full-scene mIoU {bo['miou']:.4f} | "
                  f"selection {bo['selection']*100:.1f}% | loss<={bo['max_loss']}")
            all_results[corruption] = res
            continue
        if args.prior_sweep:
            res = {}
            print("      -> Running Source-Prior Correction Sweep (prediction-only)...")
            ps = prior_correction_sweep(base_protos, proto_lbls, prior_vec,
                                        corrupt_feats, corrupt_lbls, proj, device)
            res['prior_sweep'] = ps
            print("   -> Full-scene acc | mIoU per (tau, kappa) [score = kappa*cos + tau*log pi]:")
            for r in ps['rows']:
                marker = " (baseline)" if r['tau'] == 0.0 else ""
                print(f"      tau={r['tau']:>5} kappa={r['kappa']:>5}: acc {r['acc']:.4f} | "
                      f"mIoU {r['miou']:.4f}{marker}")
            all_results[corruption] = res
            continue
        if args.tta_oracle:
            res = {}
            print("      -> Running TTA Battery + Prototype-Oracle Bounds...")
            td = tta_oracle_decode(base_protos, proto_lbls, clean_stats,
                                   corrupt_feats, corrupt_lbls, clf, proj, device,
                                   gate_cfg=gate_cfg)
            res['tta_oracle'] = td
            print("   -> Full-scene acc | mIoU:")
            for name, v in td.items():
                frac = f" | frac {v['frac']*100:5.1f}%" if 'frac' in v else ""
                print(f"      {name:<26}: acc {v['acc']:.4f} | mIoU {v['miou']:.4f}{frac}")
            all_results[corruption] = res
            continue
        if args.oracle_pool_sweep:
            res = {}
            print("      -> Running Full-Label Oracle Pool-Size Sweep...")
            ops = oracle_pool_sweep(base_protos, proto_lbls, corrupt_feats, corrupt_lbls,
                                    proj, device)
            res['oracle_pool_sweep'] = ops
            zs = ops['zero_shot']
            print(f"   -> Full-scene acc | mIoU per full-label oracle pool size "
                  f"(zero-shot acc {zs['acc']:.4f} | mIoU {zs['miou']:.4f}):")
            for r in ops['rows']:
                print(f"      pool {r['pool_size']:>8}: acc {r['acc']:.4f} | mIoU {r['miou']:.4f}")
            all_results[corruption] = res
            continue
        if args.iter0_label_info:
            res = {}
            print("      -> Running Iteration-0 Label-Information Diagnostic...")
            i0 = iteration0_label_info(base_protos, proto_lbls, corrupt_feats, corrupt_lbls,
                                       proj, device)
            res['iter0_label_info'] = i0
            m = i0['metrics']
            print(f"   -> Full-scene: zero-shot acc {m['zero_shot']['acc']:.4f} mIoU {m['zero_shot']['miou']:.4f} | "
                  f"naive_ema acc {m['naive_ema']['acc']:.4f} mIoU {m['naive_ema']['miou']:.4f} | "
                  f"oracle acc {m['full_label_oracle']['acc']:.4f} mIoU {m['full_label_oracle']['miou']:.4f}")
            print(f"   -> Pool pseudo-label accuracy (10kD zero-shot vs true): "
                  f"{i0['pool_pseudo_label_acc']:.4f}")
            pci = i0['per_class_val_iou']
            print("   -> Per-class val IoU: zs | naive | oracle  (pool true/pred, prec, rec):")
            for c in proto_lbls.tolist():
                ci = i0['class_info'][c]
                z = pci['zero_shot'].get(c, float('nan'))
                n = pci['naive_ema'].get(c, float('nan'))
                o = pci['full_label_oracle'].get(c, float('nan'))
                print(f"      class {c:<2}: IoU {z:.3f} | {n:.3f} | {o:.3f} | "
                      f"pool {ci['true']}/{ci['pred']} prec {ci['prec']:.3f} rec {ci['rec']:.3f} | "
                      f"top contam {ci['top_contam']}")
            all_results[corruption] = res
            continue
        if args.iter0_update_diag:
            res = {}
            print("      -> Running Iteration-0 Update-Mechanism Diagnostic...")
            iu = iteration0_update_diag(base_protos, proto_lbls, corrupt_feats, corrupt_lbls,
                                        proj, device)
            res['iter0_update_diag'] = iu
            m = iu['metrics']
            print(f"   -> Full-scene mIoU: zero-shot {m['zero_shot']['miou']:.4f} | "
                  f"naive_pseudo {m['naive_pseudo']['miou']:.4f} | "
                  f"correct_subset_gate_bound {m['correct_subset_gate_bound']['miou']:.4f} | "
                  f"full_label_oracle {m['full_label_oracle']['miou']:.4f}")
            print("   -> Per-class: prec | n_correct/n_assigned | cos(naive,oracle) | "
                  "cos(correct,oracle):")
            for c in proto_lbls.tolist():
                ci = iu['class_info'][c]
                cn = f"{ci['cos_naive_oracle']:.3f}" if ci['cos_naive_oracle'] is not None else "  n/a"
                cc = f"{ci['cos_correct_oracle']:.3f}" if ci['cos_correct_oracle'] is not None else "  n/a"
                print(f"      class {c:<2}: prec {ci['prec']:.3f} | {ci['n_correct']}/{ci['n_assigned']} | "
                      f"{cn} | {cc}")
            all_results[corruption] = res
            continue
        if args.iter2_balanced_reestimate:
            res = {}
            print("      -> Running Iteration-2 Balanced Re-Estimate (Sinkhorn)...")
            i2 = iter2_balanced_reestimate(base_protos, proto_lbls, prior_vec,
                                           corrupt_feats, corrupt_lbls, proj, device)
            res['iter2_balanced_reestimate'] = i2
            m = i2['metrics']
            print("   -> Full-scene mIoU: zero-shot {:.4f} | zs-pseudo-reest {:.4f} | "
                  "balanced-hard {:.4f} | balanced-soft {:.4f} | oracle {:.4f}"
                  .format(m['zero_shot']['miou'], m['zs_pseudo_reestimate']['miou'],
                          m['balanced_hard']['miou'], m['balanced_soft']['miou'],
                          m['oracle']['miou']))
            pa = i2['pseudo_acc']
            print(f"   -> Pseudo acc: zs {pa['zs']:.4f} | balanced_hard {pa['balanced_hard']:.4f}")
            sk = i2['sinkhorn']
            print(f"   -> Sinkhorn (best tau {sk['best_tau']}, support-match to prior "
                  f"{sk['sup_match_best']:.3f} [0 = perfect]):")
            for tk, tv in sk['per_tau'].items():
                print(f"      {tk}: mIoU {tv['miou']:.4f} | sup_match {tv['sup_match']:.3f}")
            print("   -> Support (zs / balanced / prior):")
            for c in proto_lbls.tolist():
                print(f"      class {c:<2}: {i2['support']['zs'][str(c)]:>7} / "
                      f"{i2['support']['balanced'][str(c)]:>7} / {i2['support']['prior'][str(c)]:>7}")
            all_results[corruption] = res
            continue
        if args.react_test:
            res = {}
            print("      -> Running ReAct Norm-Clipping Test (before projection)...")
            rt = react_test(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, proj, device)
            res['react_test'] = rt
            print("   -> Frozen-prototype decode per clip threshold: acc | mIoU | bin_cos | frac_clipped:")
            for r in rt['rows']:
                t = f"{r['threshold']:.1f}" if r['threshold'] != float('inf') else "inf"
                print(f"      clip {t:>5}: acc {r['acc']:.4f} | mIoU {r['miou']:.4f} | "
                      f"bin_cos {r['bin_cos']:.4f} | frac_clipped {r['frac_clipped']:.3f}")
            all_results[corruption] = res
            continue
        if args.deep_label_analysis:
            res = {}
            print("      -> Running Deep Label-Information Analysis...")
            da = deep_label_analysis(base_protos, proto_lbls, clean_feats, clean_lbls,
                                     corrupt_feats, corrupt_lbls, clf, proj, device)
            res['deep_label_analysis'] = da
            print("   -> Pool pseudo acc: zs {:.4f} | LP {:.4f} | re-est mIoU: zs {:.4f} | "
                  "LP {:.4f} | oracle {:.4f}".format(
                      da['pseudo']['zs_acc_pool'], da['pseudo']['lp_acc_pool'],
                      da['pseudo']['zs_miou'], da['pseudo']['lp_miou'],
                      da['pseudo']['oracle_miou']))
            print("   -> Per-class geometry (cos_shift | norm cl/cr | tight cl/cr | "
                  "nearest-other cl/cr | LP acc cl/cr):")
            for c in proto_lbls.tolist():
                g = da['geometry'].get(str(c))
                if g is None:
                    continue
                print(f"      class {c:<2}: {g['cos_shift']:.3f} | {g['norm_clean']:.2f}/{g['norm_corrupt']:.2f} | "
                      f"{g['tight_clean']:.3f}/{g['tight_corrupt']:.3f} | "
                      f"{g['nearest_other_clean_dist'] if g['nearest_other_clean_dist'] is not None else -1:.3f}/"
                      f"{g['nearest_other_corrupt_dist'] if g['nearest_other_corrupt_dist'] is not None else -1:.3f} | "
                      f"{g['lp_clean_acc']:.3f}/{g['lp_corrupt_acc']:.3f}")
            print("   -> Recoverability AUROC (1 = signal separates recovered from stuck; "
                  "0.5 = no information):")
            for k, v in da['recover'].items():
                print(f"      {k:<12}: AUC {v['auc']:.3f} | mean recovered {v['mean_recovered']:.4f} | "
                      f"mean stuck {v['mean_stuck']:.4f}")
            all_results[corruption] = res
            continue
        res = evaluate_oracle_gating(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, clf, proj,
                                     device, pool_size=args.pool_size, gate_cfg=gate_cfg)
        res['probe_acc'] = probe_corrupt_acc
        all_results[corruption] = res
        
        print(f"   -> Perfect Oracle HDC Acc: {res['perfect_acc']:.4f} (Zero-Shot: {res['zero_shot']:.4f})")
        print("   -> Gated EMA Ladder (acc | mIoU):")
        for k, v in res['gated'].items():
            if k.endswith('_cfg') or k.endswith('_miou'):
                continue
            m = res['gated'].get(f'{k}_miou')
            if m is not None:
                print(f"      {k:<16}: {v:.4f} | {m:.4f}")
            else:
                print(f"      {k:<16}: {v:.4f}")
        zs_m = res['gated'].get('zero_shot_miou')
        if zs_m is not None:
            print(f"   -> Zero-Shot mIoU: {zs_m:.4f} | Oracle mIoU: {res['gated'].get('perfect_oracle_miou', 0):.4f} | "
                  f"Naive mIoU: {res['gated'].get('naive_ema_miou', 0):.4f}")
        pc = res.get('zero_shot_per_class_iou')
        if pc:
            print("   -> Zero-Shot Per-Class IoU: " + ", ".join(
                f"{CLASS_NAMES.get(c, str(c))}={v:.3f}" for c, v in sorted(pc.items())))
        if res['gated'].get('sdw_best_cfg'):
            print(f"      [sdw_best at u_th={res['gated']['sdw_best_cfg'][0]}, z_th={res['gated']['sdw_best_cfg'][1]}]")
        if res['gated'].get('geom_best_cfg'):
            print(f"      [geom_best at z_th={res['gated']['geom_best_cfg'][0]}]")
        qg = res.get('query_gate', {})
        if qg:
            print("   -> Query Gate (frozen prototypes, veto norm >= tau): acc | mIoU | retained")
            for k, v in qg.items():
                if v['acc'] is None:
                    print(f"      {k:<10}:   --   |   --   | {v['retained']*100:.1f}%")
                else:
                    print(f"      {k:<10}: {v['acc']:.4f} | {v['miou']:.4f} | {v['retained']*100:.1f}%")
        a = res.get('auroc', {})
        if a:
            print(f"   -> Signal AUROC (Helpful vs Harmful): "
                  f"conf {a.get('conf', 0):.3f} | norm {a.get('norm', 0):.3f} | "
                  f"joint_z {a.get('joint_z', 0):.3f} | lr {a.get('lr', 0):.3f}")
        ma = res.get('mode_auroc', {})
        if ma:
            print("   -> Gate-Mode AUROC (gate's own weight selectivity): "
                  + " | ".join(f"{k} {v:.3f}" for k, v in ma.items()))
        ws = res.get('weight_stats', {})
        if ws:
            print("   -> Gate Weight Stats (mean | %w~1 | %w~0):")
            for k, v in ws.items():
                print(f"      {k:<16}: {v['mean']:.3f} | {v['frac_one']*100:.0f}% | {v['frac_zero']*100:.0f}%")
        print(f"   -> Leave-One-Out (5k tests): {res['h_count']} Helpful, {res['hm_count']} Harmful")
        if res['hm_count'] > 0:
            print(f"      Helpful Conf: {res['h_conf']:.4f} | Harmful Conf: {res['hm_conf']:.4f}")
            print(f"      Helpful Norm: {res['h_norm']:.4f} | Harmful Norm: {res['hm_norm']:.4f}")
    
    if clean_control is not None:
        all_results['clean_control'] = clean_control
    
    print("\n\n" + "="*110)
    print(" GATED EMA LADDER (all corruptions) — acc | mIoU")
    print("="*110)
    header = (f"| {'Corruption':<16} | {'ZeroShot':<8} | {'ZS-mIoU':<8} | {'Naive':<7} | {'SDW*':<7} | "
              f"{'Oracle':<8} | {'Or-mIoU':<8} |")
    print(header)
    print("|" + "-"*17 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*9 + "|")
    for corruption, res in all_results.items():
        if corruption == 'clean_control':
            continue
        if 'gated' not in res:
            continue
        g = res['gated']
        zs_m = g.get('zero_shot_miou', 0.0)
        or_m = g.get('perfect_oracle_miou', 0.0)
        print(f"| {corruption:<16} | {g['zero_shot']:<8.4f} | {zs_m:<8.4f} | "
              f"{g['naive_ema']:<7.4f} | {g.get('sdw_best', 0):<7.4f} | "
              f"{res['perfect_acc']:<8.4f} | {or_m:<8.4f} |")
    print("="*110 + "\n")
    
    if args.autopsy:
        print("\n\n" + "="*140)
        print(" CONDITION AUTOPSY (frozen clean prototypes)")
        print("="*140)
        print(f"| {'Condition':<16} | {'Acc':<7} | {'mIoU':<7} | {'LP':<7} | {'LPmIoU':<8} | {'nMis':<7} | {'ArtFrac':<8} | "
              f"{'ArtSurv':<8} | {'marC/marM':<10} | {'nrmC/nrmM':<10} | {'<4norm':<7} | {'cosShift':<8} | "
              f"{'Ellip':<6} | {'BinCos':<7} | {'AlignAcc':<8} | {'AlignmIoU':<9} |")
        print("|" + "-"*17 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*7 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*10 + "|")
        for corruption, res in all_results.items():
            if 'autopsy' not in res:
                continue
            a = res['autopsy']
            surv = a['artifact_survivors']
            al = f"{a['align_acc']:.3f}/{a['align_miou']:.3f}" if a.get('align_acc') is not None else "n/a"
            lp_m = a.get('lp_miou', 0.0)
            print(f"| {corruption:<16} | {a['acc']:<7.3f} | {a['miou']:<7.3f} | {a['lp_acc']:<7.3f} | "
                  f"{lp_m:<8.3f} | {a['n_mis']:<7d} | {a['conf_artifact_frac']:<8.3f} | {surv[0]}/{surv[4]:<7d} | "
                  f"{a['margin_correct']:.2f}/{a['margin_mis']:.2f} | {a['norm_correct']:.1f}/{a['norm_mis']:.1f} | "
                  f"{a['near_origin']:<7.3f} | {a['cos_shift']:<8.3f} | {a['ellipticity']:<6.3f} | "
                  f"{a['binarized_cos']:<7.3f} | {al:<8} |")
        print("="*140 + "\n")
        print(" DEPTH/RANGE DIAGNOSTIC (far-thresh 25m; near_acc/far_acc of the proto decode)")
        print("="*140)
        for corruption, res in all_results.items():
            if 'autopsy' not in res:
                continue
            ds = res['autopsy'].get('depth_stats')
            if not ds:
                continue
            line = f"{corruption:<16} norm-depth corr {ds['norm_depth_corr']:+.3f} | "
            for c, row in sorted(ds.get('per_class', {}).items()):
                na = f"{row['near_acc']:.2f}" if row['near_acc'] is not None else "  -  "
                fa = f"{row['far_acc']:.2f}" if row['far_acc'] is not None else "  -  "
                line += f"c{c}(d{row['mean_depth']:.0f}m,far{row['far_frac']*100:.0f}%):{na}/{fa}  "
            print(line)
        print("="*140 + "\n")
    
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved Oracle Gating Results ({len(all_results)} corruptions) to {out_path}")
