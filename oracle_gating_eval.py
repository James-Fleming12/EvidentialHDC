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
from sklearn.metrics import roc_auc_score

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.HDC_utils import fuse_uncertainties, GATE_CFG

CORRUPTIONS = [
    'fog', 'snow', 'wet_ground', 'incomplete_echo', 
    'crosstalk', 'beam_missing', 'motion_blur', 'cross_sensor'
]

def get_hdc_projection(dim_in=128, dim_out=10000, device='cuda'):
    torch.manual_seed(42)
    proj = (torch.rand(dim_in, dim_out) > 0.5).float() * 2 - 1
    return proj.to(device)

def build_hdc_prototypes(feats_128, lbls, proj, num_classes=17, device='cuda', chunk_size=50000):
    protos = torch.zeros(num_classes, proj.shape[1], device=device)
    counts = torch.zeros(num_classes, device=device)
    
    for i in range(0, len(feats_128), chunk_size):
        chunk_f = feats_128[i:i+chunk_size].to(device)
        chunk_l = lbls[i:i+chunk_size].to(device)
        
        h_chunk = torch.sign(torch.matmul(chunk_f, proj))
        
        for c in range(num_classes):
            mask = chunk_l == c
            if mask.sum() > 0:
                protos[c] += h_chunk[mask].sum(dim=0)
                counts[c] += mask.sum()
                
    for c in range(num_classes):
        if counts[c] > 0:
            protos[c] /= counts[c]
            
    base_protos = F.normalize(protos, p=2, dim=1)
    proto_lbls = torch.arange(num_classes, device=device)
    
    # Filter out empty classes
    valid_mask = counts > 0
    return base_protos[valid_mask], proto_lbls[valid_mask]

def weighted_mean_update(base_protos, proto_lbls, pool_f_128, pool_pseudo, weights, proj,
                         device, mask=None, chunk_size=50000):
    """Chunked weighted class-mean prototype update (vectorized).

    Prototype_c = normalize( sum over pool points with pseudo-label c of w_i * sign(z_i @ proj) )

    Replaces the sequential EMA ladder: Phase 13 showed that with a small pool and
    constant alpha, prototypes get erased/re-estimated from ~1/alpha points, and
    that a 20k-point pool is far too small to refine 10kD prototypes whose base
    estimates come from millions of points. A large-pool weighted mean is the
    statistically honest adaptation operator, and chunked index_add keeps it fast.
    Classes with no pool support keep the base prototype.
    """
    num_proto = len(proto_lbls)
    D = proj.shape[1]
    S = torch.zeros(num_proto, D, device=device)
    W = torch.zeros(num_proto, device=device)
    for start in range(0, len(pool_f_128), chunk_size):
        end = min(start + chunk_size, len(pool_f_128))
        chunk = pool_f_128[start:end].to(device)
        pl = pool_pseudo[start:end]
        cw = weights[start:end].to(device)
        if mask is not None:
            cw = cw * mask[start:end].to(device)
        h = torch.sign(torch.matmul(chunk, proj))  # [B, 10000]
        valid = torch.isin(pl, proto_lbls)
        idx = torch.searchsorted(proto_lbls, pl)
        S.index_add_(0, idx[valid], (cw[valid].unsqueeze(1) * h[valid]).float())
        W.index_add_(0, idx[valid], cw[valid].float())
    empty = W <= 0
    S = F.normalize(S, p=2, dim=1)
    S[empty] = F.normalize(base_protos[empty], p=2, dim=1)
    return S

def eval_protos(protos, proto_lbls, val_feats, val_lbls):
    sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), protos.T)
    preds = proto_lbls[sims.argmax(dim=1)]
    return (preds == val_lbls).float().mean().item()

def compute_signal_aurocs(meta_list):
    """AUROC of each gate signal for separating Helpful (delta > 0) from Harmful (delta < 0) updates."""
    if len(meta_list) < 10:
        return {}
    confs = np.array([m['conf'] for m in meta_list], dtype=np.float64)
    norms = np.array([m['norm'] for m in meta_list], dtype=np.float64)
    deltas = np.array([m['delta'] for m in meta_list])
    y = (deltas > 0).astype(int)
    if len(np.unique(y)) < 2:
        return {}
    c_z = (confs - confs.mean()) / (confs.std() + 1e-8)
    n_z = (norms - norms.mean()) / (norms.std() + 1e-8)
    aucs = {
        'conf': roc_auc_score(y, confs),
        'norm': roc_auc_score(y, -n_z),          # higher norm -> harmful
        'joint_z': roc_auc_score(y, c_z - n_z),  # Phase 11 proposal
    }
    try:
        lr = LogisticRegression(max_iter=1000).fit(np.stack([confs, norms], axis=1), y)
        aucs['lr'] = roc_auc_score(y, lr.decision_function(np.stack([confs, norms], axis=1)))
    except Exception:
        aucs['lr'] = None
    return aucs

def evaluate_oracle_gating(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, clf, proj,
                           device='cuda', pool_size=1000000, gate_cfg=None):
    c_lbl = corrupt_lbls.to(device)
    
    # Get pseudo-labels and confidence for Fog points (in 128D)
    print("      -> Running Probe Inference (128D)...")
    corrupt_probs = clf.predict_proba(corrupt_feats.numpy())
    corrupt_pseudo_lbls = torch.tensor(corrupt_probs.argmax(axis=1)).to(device)
    corrupt_confidences = torch.tensor(corrupt_probs.max(axis=1)).to(device)
    
    # Extract sets to avoid OOM.
    # FIX (Phase 13): pool = first 20k points and val = last 100k points were ~98
    # frames apart (different scenes), so even the ground-truth oracle LOST to
    # zero-shot on every corruption. Both sets now come from one seeded uniform
    # permutation over all points, so adaptation and evaluation share the same
    # frame distribution. The whole pipeline is seeded in main() for reproducibility.
    val_size = 100000
    torch.manual_seed(42)
    perm = torch.randperm(len(corrupt_feats))
    pool_idx = perm[:pool_size]
    val_idx = perm[-val_size:]
    
    pool_f_128 = corrupt_feats[pool_idx].to(device)
    pool_lbls = c_lbl[pool_idx]
    pool_pseudo = corrupt_pseudo_lbls[pool_idx]
    pool_conf = corrupt_confidences[pool_idx]
    
    val_f_128 = corrupt_feats[val_idx].to(device)
    val_lbls = c_lbl[val_idx]
    
    print("      -> Projecting Validation Set to 10kD HDC...")
    val_feats = torch.sign(torch.matmul(val_f_128, proj))
    
    # Project only the leave-one-out subset (5k) to 10kD; the ladder projects its
    # large pool chunk-by-chunk inside weighted_mean_update to bound GPU memory.
    print("      -> Projecting Leave-One-Out Pool (5k) to 10kD HDC...")
    lou_size = min(5000, len(pool_f_128))
    pool_feats = torch.sign(torch.matmul(pool_f_128[:lou_size], proj))
    
    # Gate signals (z-scored over the pool) + all gate-mode weights up front
    pool_norm = torch.norm(pool_f_128, p=2, dim=1)
    n_z = (pool_norm - pool_norm.mean()) / (pool_norm.std() + 1e-8)
    c_z = (pool_conf - pool_conf.mean()) / (pool_conf.std() + 1e-8)
    u_epi = 1.0 - pool_conf.clamp(0.0, 1.0)  # epistemic proxy in (0,1], higher = worse
    
    w_modes = {}
    for mode in ['epistemic', 'geometric', 'soft_dual_weight', 'and_gate', 'ellipsoid_gate']:
        w_modes[mode] = fuse_uncertainties(u_epi, n_z, method=mode, cfg=gate_cfg)
    # Flipped joint (Phase 13 hypothesis): harmful = HIGH confidence AND HIGH norm
    w_modes['joint_flip'] = torch.exp(-torch.relu(c_z)) * torch.exp(-torch.relu(n_z))
    
    # Base accuracy on validation set
    sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), base_protos.T)
    base_preds = proto_lbls[sims.argmax(dim=1)]
    zero_shot_correct = (base_preds == val_lbls).sum().item()
    zero_shot_acc = zero_shot_correct / len(val_lbls)
    
    # Perfect Oracle test: weighted class-mean update restricted to true-label points
    print("      -> Running Perfect Oracle Test...")
    mask_perfect = (pool_pseudo == pool_lbls).float()
    w_one = torch.ones_like(pool_conf)
    adapted_protos = weighted_mean_update(base_protos, proto_lbls, pool_f_128,
                                          pool_pseudo, w_one, proj, device, mask=mask_perfect)
    perfect_acc = eval_protos(adapted_protos, proto_lbls, val_feats, val_lbls)
    
    # Leave-One-Update-Out (also collects per-update metadata for the AUROC diagnostics)
    print("      -> Running Leave-One-Update-Out Test...")
    helpful, neutral, harmful, all_meta = [], [], [], []
    alpha_single = 0.05 
    eval_pool_size = min(5000, len(pool_feats)) # 5000 updates tested
    
    for i in tqdm(range(eval_pool_size), desc="         Updates", leave=False):
        pl = pool_pseudo[i]
        idx = (proto_lbls == pl).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        
        new_protos = base_protos.clone()
        new_protos[idx] = new_protos[idx] * (1 - alpha_single) + pool_feats[i] * alpha_single
        new_protos[idx] = F.normalize(new_protos[idx], p=2, dim=0)
        
        sims = torch.matmul(F.normalize(val_feats, p=2, dim=1), new_protos.T)
        preds = proto_lbls[sims.argmax(dim=1)]
        new_correct = (preds == val_lbls).sum().item()
        
        delta = new_correct - zero_shot_correct
        
        meta = {
            'conf': pool_conf[i].item(),
            'norm': pool_norm[i].item(),
            'delta': delta,
        }
        for mode, w in w_modes.items():
            meta[f'w_{mode}'] = w[i].item()
        all_meta.append(meta)
        
        if delta > 0:
            helpful.append(meta)
        elif delta < 0:
            harmful.append(meta)
        else:
            neutral.append(meta)
    
    # Gate-mode selectivity: AUROC of each gate's own weight for Helpful vs Harmful updates.
    # AUC > 0.5 = the gate score is selective in this space; AUC < 0.5 = it admits poison.
    y_lou = np.array([m['delta'] > 0 for m in all_meta]).astype(int)
    mode_auroc = {}
    if len(np.unique(y_lou)) >= 2:
        for mode in w_modes:
            wv = np.array([m[f'w_{mode}'] for m in all_meta], dtype=np.float64)
            if wv.std() > 0:
                mode_auroc[mode] = roc_auc_score(y_lou, wv)
    
    # Gate weight statistics: expose degeneracy (Phase 13: on the VIB-capped space,
    # all shipped modes collapse into one binary norm gate because z-scores saturate)
    weight_stats = {}
    for mode, w in w_modes.items():
        wv = w.detach().cpu().numpy()
        weight_stats[mode] = {
            'mean': float(wv.mean()),
            'frac_one': float((wv >= 0.999).mean()),   # saturated admit
            'frac_zero': float((wv <= 1e-6).mean()),   # saturated veto
        }
    
    # Diagnostic 1: Gated EMA Ladder (weighted class-mean updates)
    print("      -> Running Gated EMA Ladder...")
    
    gated = {'zero_shot': zero_shot_acc, 'perfect_oracle': perfect_acc}
    
    w_uniform = torch.ones_like(pool_conf)
    gated['naive_ema'] = eval_protos(
        weighted_mean_update(base_protos, proto_lbls, pool_f_128, pool_pseudo, w_uniform, proj, device),
        proto_lbls, val_feats, val_lbls)
    
    w_top50 = torch.zeros_like(pool_conf)
    w_top50[torch.topk(pool_conf, pool_size // 2).indices] = 1.0
    gated['top50_conf'] = eval_protos(
        weighted_mean_update(base_protos, proto_lbls, pool_f_128, pool_pseudo, w_top50, proj, device),
        proto_lbls, val_feats, val_lbls)
    
    for mode in ['epistemic', 'geometric', 'soft_dual_weight', 'and_gate', 'ellipsoid_gate', 'joint_flip']:
        gated[mode] = eval_protos(
            weighted_mean_update(base_protos, proto_lbls, pool_f_128, pool_pseudo, w_modes[mode], proj, device),
            proto_lbls, val_feats, val_lbls)
    
    # Threshold envelope sweep: best-case soft_dual_weight / geometric over a small grid.
    # If even the best config cannot beat naive EMA, the gate family is dead in this space;
    # if it can, the shipped defaults were simply old-space calibration.
    print("      -> Sweeping Gate Thresholds (soft_dual_weight, geometric)...")
    best_sdw, best_sdw_cfg = -1.0, None
    for u_th in [0.25, 0.5, 0.75]:
        for z_th in [0.0, 0.5, 1.0]:
            w = fuse_uncertainties(u_epi, n_z, method='soft_dual_weight',
                                   cfg={"u_th": u_th, "u_coef": gate_cfg.get("u_coef", 1.5),
                                        "z_th": z_th, "z_coef": gate_cfg.get("z_coef", 1.0)})
            acc = eval_protos(weighted_mean_update(base_protos, proto_lbls, pool_f_128,
                                                   pool_pseudo, w, proj, device),
                              proto_lbls, val_feats, val_lbls)
            if acc > best_sdw:
                best_sdw, best_sdw_cfg = acc, [u_th, z_th]
    gated['sdw_best'] = best_sdw
    gated['sdw_best_cfg'] = best_sdw_cfg
    
    best_geom, best_geom_cfg = -1.0, None
    for z_th in [-0.5, 0.0, 0.5, 1.0]:
        w = fuse_uncertainties(u_epi, n_z, method='geometric',
                               cfg={"z_th": z_th, "z_coef": gate_cfg.get("z_coef", 1.0)})
        acc = eval_protos(weighted_mean_update(base_protos, proto_lbls, pool_f_128,
                                               pool_pseudo, w, proj, device),
                          proto_lbls, val_feats, val_lbls)
        if acc > best_geom:
            best_geom, best_geom_cfg = acc, [z_th]
    gated['geom_best'] = best_geom
    gated['geom_best_cfg'] = best_geom_cfg
    
    res = {
        'zero_shot': zero_shot_acc,
        'perfect_acc': perfect_acc,
        'gated': gated,
        'auroc': compute_signal_aurocs(all_meta),
        'mode_auroc': mode_auroc,
        'weight_stats': weight_stats,
        'h_conf': np.mean([m['conf'] for m in helpful]) if helpful else 0.0,
        'hm_conf': np.mean([m['conf'] for m in harmful]) if harmful else 0.0,
        'h_norm': np.mean([m['norm'] for m in helpful]) if helpful else 0.0,
        'hm_norm': np.mean([m['norm'] for m in harmful]) if harmful else 0.0,
        'h_count': len(helpful),
        'hm_count': len(harmful)
    }
    return res

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
    args, _ = parser.parse_known_args()
    
    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
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
            if not k.endswith('_cfg'):
                print(f"      {k:<16}: {v:.4f}")
    
    # We no longer need the massive clean_feats tensor
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
        
        corrupt_feats, corrupt_lbls = [], []
        
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
                corrupt_feats.append(z_flat.cpu())
                corrupt_lbls.append(labels[mask].cpu())
                
        corrupt_feats = torch.cat(corrupt_feats, dim=0)
        corrupt_lbls = torch.cat(corrupt_lbls, dim=0)
        
        probe_corrupt_acc = clf.score(corrupt_feats[:train_size].numpy(), corrupt_lbls[:train_size].numpy())
        print(f"   -> 128D Linear Probe Accuracy: {probe_corrupt_acc:.4f}")
        
        res = evaluate_oracle_gating(base_protos, proto_lbls, corrupt_feats, corrupt_lbls, clf, proj,
                                     device, pool_size=args.pool_size, gate_cfg=gate_cfg)
        res['probe_acc'] = probe_corrupt_acc
        all_results[corruption] = res
        
        print(f"   -> Perfect Oracle HDC Acc: {res['perfect_acc']:.4f} (Zero-Shot: {res['zero_shot']:.4f})")
        print("   -> Gated EMA Ladder:")
        for k, v in res['gated'].items():
            if not k.endswith('_cfg'):
                print(f"      {k:<16}: {v:.4f}")
        if res['gated'].get('sdw_best_cfg'):
            print(f"      [sdw_best at u_th={res['gated']['sdw_best_cfg'][0]}, z_th={res['gated']['sdw_best_cfg'][1]}]")
        if res['gated'].get('geom_best_cfg'):
            print(f"      [geom_best at z_th={res['gated']['geom_best_cfg'][0]}]")
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
    print(" GATED EMA LADDER (all corruptions)")
    print("="*110)
    header = (f"| {'Corruption':<16} | {'ZeroShot':<8} | {'Naive':<7} | {'Top50':<7} | {'Epi':<7} | "
              f"{'Geom':<7} | {'SDW':<7} | {'AND':<7} | {'Flip':<7} | {'SDW*':<7} | {'Geom*':<7} | {'Oracle':<8} |")
    print(header)
    print("|" + "-"*17 + "|" + "-"*9 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*9 + "|")
    for corruption, res in all_results.items():
        if corruption == 'clean_control':
            continue
        g = res['gated']
        print(f"| {corruption:<16} | {g['zero_shot']:<8.4f} | {g['naive_ema']:<7.4f} | {g['top50_conf']:<7.4f} | "
              f"{g.get('epistemic', 0):<7.4f} | {g.get('geometric', 0):<7.4f} | {g.get('soft_dual_weight', 0):<7.4f} | "
              f"{g.get('and_gate', 0):<7.4f} | {g.get('joint_flip', 0):<7.4f} | {g.get('sdw_best', 0):<7.4f} | "
              f"{g.get('geom_best', 0):<7.4f} | {res['perfect_acc']:<8.4f} |")
    print("="*110 + "\n")
    
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved Oracle Gating Results ({len(all_results)} corruptions) to {out_path}")

if __name__ == '__main__':
    main()
