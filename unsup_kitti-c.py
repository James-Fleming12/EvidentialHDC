import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import unsup_main
from dataset.kitti.parser import Parser
from modules import compare as _baselines
from modules.HDC_utils import set_uq_model
from unsup_main import (
    extract_metrics_from_conf_matrix,
    save_graphic,
    setup_logger,
    train_extractor,
    train_hdc,
)

NUM_CLASSES = 17
KITTI_DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
CORRUPTIONS = [
    'fog', 
    'wet_ground', 
    'snow', 
    'motion_blur', 
    'beam_missing', 
    'crosstalk', 
    'incomplete_echo', 
    'cross_sensor'
]
SEVERITY_MAP = {1: 'light', 2: 'moderate', 3: 'heavy', 4: 'extreme'}

CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS_KITTI_ALL = "config/labels/semantic-kitti-all.yaml"  # Standard 17 classes

ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
DATA = yaml.safe_load(open(CONFIG_LABELS_KITTI_ALL, 'r'))

def compute_auroc_torch(scores, labels):
    if len(scores) == 0: return float('nan')
    pos_mask = labels.bool()
    n1 = pos_mask.sum().item()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    unique_vals, inv, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    cum_counts = torch.cumsum(counts, dim=0)
    start_ranks = cum_counts - counts + 1
    avg_ranks = (start_ranks + cum_counts) / 2.0
    ranks = avg_ranks[inv].to(torch.float64)
    r1 = ranks[pos_mask].sum().item()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))

def compute_correlations_torch(x, y):
    if len(x) < 2: return "N/A"
    x = x.to(torch.float64); y = y.to(torch.float64)
    vx = x - x.mean(); vy = y - y.mean()
    cov = (vx * vy).mean()
    sx = torch.sqrt((vx**2).mean()); sy = torch.sqrt((vy**2).mean())
    pearson = float(cov / (sx * sy)) if sx != 0 and sy != 0 else 0.0
    _, inv_x, counts_x = torch.unique(x, return_inverse=True, return_counts=True)
    cum_x = torch.cumsum(counts_x, dim=0)
    ranks_x = ((cum_x - counts_x + 1 + cum_x) / 2.0)[inv_x].to(torch.float64)
    _, inv_y, counts_y = torch.unique(y, return_inverse=True, return_counts=True)
    cum_y = torch.cumsum(counts_y, dim=0)
    ranks_y = ((cum_y - counts_y + 1 + cum_y) / 2.0)[inv_y].to(torch.float64)
    vrx = ranks_x - ranks_x.mean(); vry = ranks_y - ranks_y.mean()
    cov_r = (vrx * vry).mean()
    srx = torch.sqrt((vrx**2).mean()); sry = torch.sqrt((vry**2).mean())
    spearman = float(cov_r / (srx * sry)) if srx != 0 and sry != 0 else 0.0
    return f"Pearson r={pearson:.6f}, Spearman rho={spearman:.6f} (over {len(x):,} pairs)"

def evaluate_and_adapt(model, target_dataloader, device, eval_only=False, update_method='frozen', **kwargs):
    logger = logging.getLogger("EvalAdapt")
    model_was_training = model.training
    
    miou_history = []
    head_miou_history = []
    mid_miou_history = []
    tail_miou_history = []
    acc_history = []
    iou_per_class_history = []
    num_classes = model.num_classes
    cumulative_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    firing_rates = []
    memory_errors = []

    for batch_idx, batch_data in enumerate(tqdm(target_dataloader, desc="Adapting", leave=False, miniters=50)):
        if kwargs.get('dry_run', False) and batch_idx >= 2:
            break
            
        proj_in = batch_data[0].to(device)
        proj_labels = batch_data[2].to(device).view(-1)
        
        if proj_in.shape[1] > 0:
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True):
                    latent_x = model.net(proj_in, only_feat=True)
                
                raw_enc, indices, _ = model.encode(proj_in)
                norm_enc = F.normalize(raw_enc, dim=1).to(device).to(model.classify.weight.dtype)
                
                if update_method == 'adapt_mem':
                    from modules.AdaptMemModel import AdaptiveMemoryBank
                    if not hasattr(model, 'mem_bank'):
                        # Capacity 20,000 allows for 9996 frozen Coreset Seed points + 10,004 dynamic online points
                        model.mem_bank = AdaptiveMemoryBank(hd_dim=10000, num_classes=num_classes, memory_capacity=20000).to(device)
                        
                        # Coreset Seed: Seed with the offline extracted coresets and lock them in reserved_slots!
                        if hasattr(model, 'coreset_seed_keys') and model.coreset_seed_keys is not None:
                            num_coreset_pts = model.coreset_seed_keys.size(0)
                            model.mem_bank.keys[:num_coreset_pts] = model.coreset_seed_keys
                            model.mem_bank.values[:num_coreset_pts] = model.coreset_seed_values
                            model.mem_bank.is_valid[:num_coreset_pts] = True
                            
                            # Lock these points from being overwritten
                            model.mem_bank.reserved_slots.fill_(num_coreset_pts)
                            model.mem_bank.ptr.fill_(num_coreset_pts)
                        else:
                            # Fallback just in case
                            proto_weights = torch.sign(model.classify.weight.clone().detach())
                            proto_weights[proto_weights == 0] = 1.0
                            for c in range(num_classes):
                                model.mem_bank.keys[c*10:(c+1)*10] = proto_weights[c]
                                model.mem_bank.values[c*10:(c+1)*10] = c
                                model.mem_bank.is_valid[c*10:(c+1)*10] = True
                            model.mem_bank.reserved_slots.fill_(num_classes * 10)
                            model.mem_bank.ptr.fill_(num_classes * 10)
                    
                    # --- Memory Bank Forward Pass ---
                    predictions, _ = model.mem_bank.query(norm_enc)
                    
                    if not eval_only:
                        # --- Iteration 7: Manifold Denoiser Gating ---
                        with torch.no_grad():
                            recon = model.denoiser(norm_enc)
                            recon_error = 1.0 - torch.cosine_similarity(norm_enc, recon, dim=1)
                        
                        # Use inverse error as purity (threshold is 0.8 in mem_bank.update, meaning error < 0.20)
                        mem_purity = 1.0 - recon_error
                        
                        rate, purity_err = model.mem_bank.update(norm_enc, predictions, mem_purity, true_labels=proj_labels[indices])
                        if rate is not None: firing_rates.append(rate)
                        if purity_err is not None and purity_err >= 0: memory_errors.append(purity_err)
                else:
                    # Legacy frozen inference
                    logits = model.classify(norm_enc)
                    predictions = torch.argmax(logits, dim=1)
                
                selected_labels = proj_labels[indices]
                mask = (selected_labels >= 0) & (selected_labels < num_classes)
                if mask.any():
                    hist = torch.bincount(
                        num_classes * selected_labels[mask] + predictions[mask], 
                        minlength=num_classes ** 2
                    ).reshape(num_classes, num_classes)
                    cumulative_confusion_matrix += hist
                    
            cumulative_miou, head_miou, mid_miou, tail_miou, cumulative_acc, cumulative_iou_per_class = extract_metrics_from_conf_matrix(cumulative_confusion_matrix)
            miou_history.append(cumulative_miou)
            head_miou_history.append(head_miou)
            mid_miou_history.append(mid_miou)
            tail_miou_history.append(tail_miou)
            acc_history.append(cumulative_acc)
            iou_per_class_history.append(cumulative_iou_per_class)
            
    if not eval_only:
        model.train(model_was_training)
        
    avg_firing = sum(firing_rates) / max(1, len(firing_rates))
    avg_mem_err = sum(memory_errors) / max(1, len(memory_errors))
        
    return {
        "mIoU": miou_history, 
        "Head_mIoU": head_miou_history,
        "Mid_mIoU": mid_miou_history,
        "Tail_mIoU": tail_miou_history,
        "Accuracy": acc_history, 
        "IoU_per_class": iou_per_class_history, 
        "FiringRate": avg_firing, 
        "MemoryError": avg_mem_err,
        "UpdateMagnitude": 0.0,
        "ConfusionMatrix": cumulative_confusion_matrix.cpu().numpy().tolist()
    }
def pretrain_pipeline(ARCH, DATA, data_dir, pretrained_path, return_trainer=False, skip_extractor=False, resume_path=None, hdc_epochs=15, extractor_epochs=60):
    log_base = os.path.dirname(pretrained_path)
    os.makedirs(log_base, exist_ok=True)
    
    unsup_main.LOG_DIR = log_base
    unsup_main.MODEL_DIR = log_base
    unsup_main.HDC_SAVE_PATH = os.path.join(log_base, "hdc.pth")
    unsup_main.HDC_SUB_PATH = pretrained_path

    if not skip_extractor:
        ARCH["train"]["batch_size"] = 24
        print(f"Pretraining feature extractor on {data_dir}...")
        trainer = train_extractor(ARCH, DATA, epochs=extractor_epochs, data_dir=data_dir, return_trainer=True, resume_path=resume_path)
    else:
        print("Skipping feature extractor pretraining...")
        trainer = None
    
    ARCH["train"]["batch_size"] = 6
    print(f"Pretraining HDC density model on {data_dir} for {hdc_epochs} epochs...")
    model, _ = train_hdc(ARCH, DATA, epochs=hdc_epochs, data_dir=data_dir, return_extractor=True)

    if return_trainer:
        return model, trainer
    return model

def save_degradation_plot(save_path, title, data_dict, metric="mIoU", baseline_val=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    severities = [1, 2, 3, 4, 5]
    colors = plt.cm.tab10.colors
    
    for i, (corr, sev_dict) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        initial_vals = [sev_dict.get(s, (0, 0))[0] for s in severities]
        final_vals = [sev_dict.get(s, (0, 0))[1] for s in severities]
        
        plt.plot(severities, initial_vals, marker='x', linestyle=':', color=color, alpha=0.6, label=f'{corr} (Initial)')
        plt.plot(severities, final_vals, marker='o', linestyle='-', color=color, label=f'{corr} (Final)')
        
    if baseline_val is not None:
        plt.axhline(y=baseline_val, color='r', linestyle='--', label=f'Clean Baseline ({baseline_val:.4f})')
    
    plt.title(f"{title} - {metric} Degradation")
    plt.xlabel("Severity")
    plt.ylabel(metric)
    plt.xticks(severities)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_hdc_model(path, num_classes=NUM_CLASSES, mv_tta='none'):
    # print(f"Loading pretrained HDC model from {path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
    modeldir = os.path.dirname(path)

    if mv_tta != 'none':
        from modules.HDC_utils import set_mv_tta_model
        model = set_mv_tta_model(ARCH, modeldir, 'rp', 0, 0, num_classes, device, mv_tta=mv_tta)
    else:
        model = set_uq_model(ARCH, modeldir, 'rp', 0, 0, num_classes, device)
    
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.to(device)
    return model

def main():
    import random
    parser = argparse.ArgumentParser(description="Test Unsupervised Updates on KITTI-C")
    parser.add_argument('--pretrain', action='store_true', help='Run pretraining on SemanticKITTI before evaluating')
    parser.add_argument('--chunked', action='store_true', help='Use chunked protocol: continuous adaptation across disjoint 1/7th splits instead of full independent sequences.')
    parser.add_argument('--reset_per_corruption', action='store_true', help='Reset the model to the clean pretrained weights before adapting on each corruption (requires --chunked).')
    parser.add_argument('--continual', action='store_true', help='Continual learning mode: continuous adaptation across sequences without resetting.')
    parser.add_argument('--pretrained_path', type=str, default='logs/kitti_pretrain/hdc_sub.pth', help='Path to load pretrained model')
    parser.add_argument('--log_dir', type=str, default='logs/kitti_c_test', help='Directory to save logs and graphics')
    parser.add_argument('--method', type=str, default='frozen', help='Method to test.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for noise floor tests')
    parser.add_argument('--dry_run', action='store_true', help='Run only 2 batches per condition to quickly verify no crashes will occur.')
    parser.add_argument('--continue_pretrain', action='store_true', help='Resume pretraining from the existing pretrained_path')
    parser.add_argument('--continue', dest='continue_epochs', type=int, default=0, help='Continue feature extractor training for this many epochs, reinitialize HDC, and perform adaptation')
    parser.add_argument('--skip_extractor', action='store_true', help='Skip feature extractor pretraining and only retrain the HDC model')
    parser.add_argument('--extractor_epochs', type=int, default=60, help='Number of epochs to train the feature extractor')
    parser.add_argument('--hdc_epochs', type=int, default=15, help='Number of epochs to train the HDC density model')
    parser.add_argument('--severity', type=int, default=3, help='Severity level for corruptions')
    parser.add_argument('--kitti_dir', type=str, default='/mnt/alpha/jmfleming/KITTI', help='Path to SemanticKITTI dataset for pretraining')
    parser.add_argument('--kittic_dir', type=str, default='/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C', help='Path to real SemanticKITTI-C dataset')
    parser.add_argument('--corruptions', type=str, default='fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor', help='Comma separated list of corruptions to test. Defaults to all 8 corruptions.')

    parser.add_argument('--tau', type=float, default=None, help='Logit adjustment tau for inference prior. If not set, BM is used. τ=0 is normalized baseline. τ<0 is majority amplifier. τ>0 is balanced softmax.')
    parser.add_argument('--kappa', type=float, default=15.0, help='Logit scale kappa used with tau. Controls evidence weighting vs prior.')
    parser.add_argument('--normalize_weights', action='store_true', help='Force weights to be normalized after every update (disables Bayesian Momentum accumulator).')
    parser.add_argument('--gate_mode', type=str, default='epistemic', help='Uncertainty gating strategy: epistemic, geometric, and_gate, or_gate, oracle, rescue_gate, ellipsoid_gate, soft_dual_weight, view_var_gate')
    parser.add_argument('--no_diagnostics', action='store_false', dest='diagnostics', help='Disable heavy diagnostic tracking.')
    parser.add_argument('--dump_features', action='store_true', default=False, help='Dump per-point feature tensors for Test D5/D6 offline probe.')
    parser.set_defaults(diagnostics=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.continue_epochs > 0:
        args.pretrain = True
        args.continue_pretrain = True
        args.extractor_epochs = args.continue_epochs

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, 'kitti_c.log'))

    try:
        ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
        DATA = yaml.safe_load(open(CONFIG_LABELS_KITTI_ALL, 'r'))
    except Exception as e:
        logger.error(f"Error loading configs: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.pretrain:
        logger.info(f"Starting Pretraining on SemanticKITTI at {args.kitti_dir}...")
        resume_dir = os.path.dirname(args.pretrained_path) if args.continue_pretrain else None
            
        model, trainer = pretrain_pipeline(
            ARCH, DATA, data_dir=args.kitti_dir, 
            pretrained_path=args.pretrained_path, return_trainer=True, 
            skip_extractor=args.skip_extractor, resume_path=resume_dir, 
            hdc_epochs=args.hdc_epochs, extractor_epochs=args.extractor_epochs
        )
        
        if trainer is not None:
            opt_path = os.path.join(os.path.dirname(args.pretrained_path), 'feature_optimizer.pth')
            torch.save(trainer.optimizer.state_dict(), opt_path)
            logger.info(f"Successfully pretrained model on SemanticKITTI. Optimizer state saved to {opt_path}")
            
    sev = args.severity
    methods_to_run = [m.strip() for m in args.method.split(',')]
    
    global_results_path = os.path.join(args.log_dir, 'global_results.json')
    global_results = None
    if os.path.exists(global_results_path):
        try:
            with open(global_results_path, 'r') as f:
                global_results = json.load(f)
        except json.JSONDecodeError:
            global_results = None
            
    full_method_names = []
    for m in methods_to_run:
        if m == 'frozen':
            if args.tau is not None and args.tau != 0.0:
                full_method_names.append(f"frozen_tau_{args.tau}")
            else:
                full_method_names.append('frozen')
        elif m in ['conformalhdc', 'hyperdum', 'd3ctta']:
            full_method_names.append(m)
        else:
            m_name = f"{m}"
            full_method_names.append(m_name)
                    
    if global_results is not None:
        # Ensure the dicts for the current methods exist in case they were never run
        for m in full_method_names:
            if m not in global_results.get('mIoU', {}):
                global_results.setdefault('mIoU', {})[m] = {c: {} for c in CORRUPTIONS}
            if m not in global_results.get('Accuracy', {}):
                global_results.setdefault('Accuracy', {})[m] = {c: {} for c in CORRUPTIONS}
    else:
        global_results = {
            'mIoU': {m: {c: {} for c in CORRUPTIONS} for m in full_method_names},
            'Accuracy': {m: {c: {} for c in CORRUPTIONS} for m in full_method_names},
        }
    
    shared_init_metrics = {}
    
    # Load dataset once and partition it to find chunks
    # Note on Protocol: Divides the valid set into 7 disjoint chunks (1 per corruption).
    # This evaluates each corruption on 1/7 of the validation set (e.g., ~581 frames) instead 
    # of the full set. We are preserving this behavior to identically match the 3-chunk protocol. 
    # Per-domain metrics will be noisier on 400 frames, so do not directly compare these 
    # chunked metrics to full-set benchmarks.
    logger.info("Initializing baseline dataset to calculate chunk sizes...")
    parser_obj = Parser(root=KITTI_DATA_DIR,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=None,
                    labels=DATA["labels"],
                    color_map=DATA.get("color_map", {}),
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=1,
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=False)
    
    target_dataset = parser_obj.validloader.dataset
    total_len = len(target_dataset)
    chunk_size = total_len // len(CORRUPTIONS)
    
    indices = list(range(total_len))
    chunks = []
    for i in range(len(CORRUPTIONS)):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < len(CORRUPTIONS) - 1 else total_len
        chunks.append(indices[start_idx:end_idx])

    base_model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES)
    base_model.populate_source_statistics(args.kitti_dir, ARCH, DATA, device, dry_run=args.dry_run)
    
    source_stats_cache = {
        'class_latent_means': base_model.class_latent_means,
        'source_density_mean': getattr(base_model, 'source_density_mean', None),
        'source_density_std': getattr(base_model, 'source_density_std', None),
        'source_mu_cos': getattr(base_model, 'source_mu_cos', None),
        'source_sigma_cos': getattr(base_model, 'source_sigma_cos', None),
        'drift_mu_0': getattr(base_model, 'drift_mu_0', None),
        'source_class_freq': getattr(base_model, 'source_class_freq', None),
        'source_bank': getattr(base_model, 'source_bank', None),
        'coreset_seed_keys': getattr(base_model, 'coreset_seed_keys', None),
        'coreset_seed_values': getattr(base_model, 'coreset_seed_values', None),
        'denoiser_state_dict': getattr(base_model, 'denoiser_state_dict', None)
    }

    clean_state_dict = torch.load(args.pretrained_path, map_location=device)
    
    logger.info("Pre-loading corruption datasets...")
    corruption_datasets = {}
    for ctype in CORRUPTIONS:
        sev_str = SEVERITY_MAP.get(sev, 'moderate')
        corruption_root = os.path.join(args.kittic_dir, ctype, sev_str)
        seq_dir = os.path.join(corruption_root, "sequences")
        if not os.path.exists(seq_dir):
            logger.info(f"Directory structure doesn't match standard KITTI. Creating 'sequences/08' symlink in {corruption_root}...")
            os.makedirs(seq_dir, exist_ok=True)
            os.symlink("..", os.path.join(seq_dir, "08"))
        try:
            parser_obj = Parser(root=corruption_root,
                                train_sequences=DATA["split"]["valid"],
                                valid_sequences=DATA["split"]["valid"],
                                test_sequences=None,
                                labels=DATA["labels"],
                                color_map=DATA.get("color_map", {}),
                                learning_map=DATA["learning_map"],
                                learning_map_inv=DATA["learning_map_inv"],
                                sensor=ARCH["dataset"]["sensor"],
                                max_points=ARCH["dataset"]["max_points"],
                                batch_size=1,
                                workers=ARCH["train"]["workers"],
                                gt=True,
                                shuffle_train=False)
            corruption_datasets[ctype] = parser_obj.validloader.dataset
        except Exception as e:
            logger.error(f"Failed to load KITTI-C corruption dataset at {corruption_root}: {e}")

    # Initialize the model exactly ONCE to be shared
    model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES)
    if source_stats_cache is not None:
        model.class_latent_means = source_stats_cache['class_latent_means'].to(device) if source_stats_cache['class_latent_means'] is not None else None
        model.source_density_mean = source_stats_cache['source_density_mean'].to(device) if source_stats_cache.get('source_density_mean') is not None else None
        model.source_density_std = source_stats_cache['source_density_std'].to(device) if source_stats_cache['source_density_std'] is not None else None
        model.source_mu_cos = source_stats_cache['source_mu_cos'].to(device) if source_stats_cache['source_mu_cos'] is not None else None
        model.source_sigma_cos = source_stats_cache['source_sigma_cos'].to(device) if source_stats_cache['source_sigma_cos'] is not None else None
        model.drift_mu_0 = source_stats_cache['drift_mu_0'].to(device) if source_stats_cache['drift_mu_0'] is not None else None
        model.source_class_freq = source_stats_cache['source_class_freq'].to(device) if source_stats_cache['source_class_freq'] is not None else None
        model.source_bank = source_stats_cache['source_bank'].to(device) if source_stats_cache.get('source_bank') is not None else None
        
        # Load the pre-trained Manifold Denoiser
        from modules.AdaptMemModel import HDCDenoiser
        model.denoiser = HDCDenoiser(hd_dim=10000, hidden_dim=256).to(device)
        if source_stats_cache.get('denoiser_state_dict') is not None:
            model.denoiser.load_state_dict({k: v.to(device) for k, v in source_stats_cache['denoiser_state_dict'].items()})
        model.denoiser.eval()
        
        # Iteration 7: Set the denoiser threshold (replaces old 0.8 cohesion threshold)
        # Average Fog Correct error is ~0.77. Average Fog Halluc error is ~0.93.
        # We set error threshold to 0.85 (which means mem_purity > 0.15)
        model.mem_bank.purity_threshold = 0.15

    for current_method, full_method_name in zip(methods_to_run, full_method_names):
        logger.info("=========================================")
        logger.info(f"Starting Evaluation for Method: {full_method_name}")
        logger.info("=========================================")
        
        active_corruptions = CORRUPTIONS
        if args.corruptions:
            active_corruptions = [c.strip() for c in args.corruptions.split(',')]

        results_miou = {c: {} for c in active_corruptions}
        results_acc = {c: {} for c in active_corruptions}

        # Reset model at the start of each new method loop
        model.load_state_dict(clean_state_dict, strict=False)
        attrs_to_del = ['mem_bank', 'drift_mu_c', 'class_freq_ema', 'class_update_counts', 'class_M', 'running_density_std', 'running_density_mean', '_contingency_table', '_mv_contingency_table', '_decay_logs', '_class_n_points', '_class_n_fired', '_class_true_errors_rejected', '_class_correct_rejected', '_firing_log', '_veto_stats', '_update_magnitude_log', 'initial_classify_weights', '_feature_dump_list', '_baseline_adapter', '_baseline_adapter_name']
        for attr in attrs_to_del:
            if hasattr(model, attr):
                delattr(model, attr)
            
        eval_model = model

        for i, ctype in enumerate(active_corruptions):
            if args.reset_per_corruption and args.chunked and not args.continual:
                logger.info("Resetting model to clean pretrained weights for this corruption.")
                model.load_state_dict(clean_state_dict, strict=False)
                attrs_to_del = ['mem_bank', 'drift_mu_c', 'class_freq_ema', 'class_update_counts', 'class_M', 'running_density_std', 'running_density_mean', '_contingency_table', '_mv_contingency_table', '_decay_logs', '_class_n_points', '_class_n_fired', '_class_true_errors_rejected', '_class_correct_rejected', '_firing_log', '_veto_stats', '_update_magnitude_log', 'initial_classify_weights', '_feature_dump_list', '_baseline_adapter', '_baseline_adapter_name']
                for attr in attrs_to_del:
                    if hasattr(model, attr):
                        delattr(model, attr)
                
            logger.info(f"Testing {ctype} severity {sev} (Chunk {i+1}/{len(active_corruptions)})")
            
            if ctype not in corruption_datasets:
                continue
                
            full_corruption_dataset = corruption_datasets[ctype]
            
            # Prevent silent misalignment bugs by ensuring corrupted frame count matches baseline clean chunk length
            assert len(full_corruption_dataset) == total_len, (
                f"Length mismatch: Clean baseline length is {total_len}, "
                f"but {ctype}-{sev_str} length is {len(full_corruption_dataset)}. "
                f"Chunks will misalign."
            )
            
            chunk_dataset = None
            if not args.chunked:
                # Standard protocol: full sequence, independent adaptation
                chunk_dataset = full_corruption_dataset
                if not args.continual:
                    # Reset model before each corruption
                    model.load_state_dict(clean_state_dict, strict=False)
                    attrs_to_del = ['mem_bank', 'drift_mu_c', 'class_freq_ema', 'class_update_counts', 'class_M', 'running_density_std', 'running_density_mean', '_contingency_table', '_mv_contingency_table', '_decay_logs', '_class_n_points', '_class_n_fired', '_class_true_errors_rejected', '_class_correct_rejected', '_firing_log', '_veto_stats', '_update_magnitude_log', 'initial_classify_weights', '_feature_dump_list', '_baseline_adapter', '_baseline_adapter_name']
                    for attr in attrs_to_del:
                        if hasattr(model, attr):
                            delattr(model, attr)
            else:
                # Chunked protocol: continuous adaptation across disjoint splits
                chunk_dataset = torch.utils.data.Subset(full_corruption_dataset, chunks[i])
            
            assert chunk_dataset is not None, "chunk_dataset was not assigned!"
            target_dataloader = DataLoader(chunk_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
            
            try:
                # Always run 3-pass protocol to get True Initial and True Final
                init_key = (ctype, sev, current_method, args.chunked)
                if not args.continual and (not args.chunked or args.reset_per_corruption) and init_key in shared_init_metrics:
                    logger.debug("  -> Pass 1: Reusing cached True Initial metrics (Frozen)")
                    init_metrics = shared_init_metrics[init_key]
                else:
                    logger.debug("  -> Pass 1: Computing True Initial metrics (Frozen)")
                    init_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=True, update_method=current_method, dry_run=args.dry_run)
                    if not args.continual and (not args.chunked or args.reset_per_corruption):
                        shared_init_metrics[init_key] = init_metrics
                
                # Pass 2: Adapt (only if method is not frozen)
                if current_method != 'frozen':
                    logger.debug("  -> Pass 2: Adapting model weights")
                    if current_method != 'adapt_mem':
                        eval_model.train()
                    else:
                        eval_model.eval() # adapt_mem relies on frozen features!
                    adapt_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=False, update_method=current_method, dry_run=args.dry_run)
                else:
                    adapt_metrics = init_metrics
                    
                # Pass 3: True Final (Frozen on chunk using adapted weights)
                if current_method != 'adapt_mem':
                    logger.debug("  -> Pass 3: Computing True Final metrics (Frozen)")
                    eval_model.eval()
                    final_metrics = evaluate_and_adapt(eval_model, target_dataloader, device, eval_only=True, update_method=current_method, dry_run=args.dry_run)
                else:
                    logger.debug("  -> Pass 3: Skipping True Final metrics for adapt_mem (sliding window buffer is fundamentally online-only)")
                    final_metrics = adapt_metrics
                
                metrics = adapt_metrics  # Just for the trajectory json
                if len(init_metrics["mIoU"]) > 0:
                    initial_miou = init_metrics["mIoU"][-1]
                    final_miou = final_metrics["mIoU"][-1]
                    online_miou = adapt_metrics["mIoU"][-1]
                    initial_acc = init_metrics["Accuracy"][-1]
                    final_acc = final_metrics["Accuracy"][-1]
                    
                    if 'ConfusionMatrix' in init_metrics:
                        cm_init = np.array(init_metrics['ConfusionMatrix'])
                        tp_init = np.diag(cm_init)
                        fp_init = cm_init.sum(axis=0) - tp_init
                        fn_init = cm_init.sum(axis=1) - tp_init
                        
                        if 'ConfusionMatrix' in final_metrics:
                            cm_fin = np.array(final_metrics['ConfusionMatrix'])
                            tp_fin = np.diag(cm_fin)
                            fp_fin = cm_fin.sum(axis=0) - tp_fin
                            fn_fin = cm_fin.sum(axis=1) - tp_fin
                        else:
                            tp_fin, fp_fin, fn_fin = tp_init, fp_init, fn_init
                            
                        tail_classes = [2, 3, 6, 7, 10]
                        logger.info(f"  -> Initial Tail TP:  {tp_init[tail_classes].tolist()} | Final Tail TP:  {tp_fin[tail_classes].tolist()} (Delta: {(tp_fin - tp_init)[tail_classes].tolist()})")
                        logger.info(f"  -> Initial Tail FP:  {fp_init[tail_classes].tolist()} | Final Tail FP:  {fp_fin[tail_classes].tolist()} (Delta: {(fp_fin - fp_init)[tail_classes].tolist()})")
                        logger.info(f"  -> Initial Tail FN:  {fn_init[tail_classes].tolist()} | Final Tail FN:  {fn_fin[tail_classes].tolist()} (Delta: {(fn_fin - fn_init)[tail_classes].tolist()})")
                else:
                    initial_miou = final_miou = online_miou = initial_acc = final_acc = 0.0
                    
                firing_rate_str = ""
                if "FiringRate" in adapt_metrics:
                    firing_rate_str = f", FiringRate={adapt_metrics['FiringRate']*100:.2f}%"
                    if "MemoryError" in adapt_metrics and adapt_metrics['MemoryError'] >= 0:
                        firing_rate_str += f", MemError={adapt_metrics['MemoryError']*100:.2f}%"
                    if "UpdateMagnitude" in adapt_metrics:
                        firing_rate_str += f", UpdateMag={adapt_metrics['UpdateMagnitude']:.4f}"
            except Exception as e:
                import traceback
                logger.error(f"FATAL ERROR during {ctype} sev {sev} ({current_method}): {e}")
                logger.error(traceback.format_exc())
                logger.info("Skipping to next cell to protect the overnight run...")
                continue
            
            if len(metrics["mIoU"]) > 0:
                results_miou[ctype][sev] = (initial_miou, final_miou)
                results_acc[ctype][sev] = (initial_acc, final_acc)
                
                global_results['mIoU'][full_method_name][ctype][sev] = (initial_miou, final_miou)
                global_results['Accuracy'][full_method_name][ctype][sev] = (initial_acc, final_acc)
                
                initial_head = init_metrics["Head_mIoU"][-1]
                final_head = final_metrics["Head_mIoU"][-1]
                initial_mid = init_metrics["Mid_mIoU"][-1]
                final_mid = final_metrics["Mid_mIoU"][-1]
                initial_tail = init_metrics["Tail_mIoU"][-1]
                final_tail = final_metrics["Tail_mIoU"][-1]
                
                protocol_str = "continual" if args.continual else ("chunked" if args.chunked else "full")
                n_frames_str = len(target_dataloader)
                logger.info(f"Result for {ctype}-{sev} [protocol={protocol_str}, n_frames={n_frames_str}]: Initial mIoU={initial_miou:.4f} -> Final (Online)={online_miou:.4f} -> Final (Frozen)={final_miou:.4f} (Head: {initial_head:.4f} -> {final_head:.4f}, Mid: {initial_mid:.4f} -> {final_mid:.4f}, Tail: {initial_tail:.4f} -> {final_tail:.4f}), Acc={initial_acc:.4f} -> {final_acc:.4f}{firing_rate_str}")
                
                if hasattr(model, 'target_prior'):
                    tail_classes = [2, 3, 6, 7, 10]
                    tail_rate = model.target_prior[tail_classes].sum().item()
                    src_tail_rate = model.source_class_freq[tail_classes].sum().item() if hasattr(model, 'source_class_freq') else 0.0
                    logger.info(f"  [D1] Rare-class prediction rate: {tail_rate:.5f} (Source: {src_tail_rate:.5f}) -> Ratio: {tail_rate/max(1e-5, src_tail_rate):.2f}")
                
                if hasattr(model, '_d4_gains') and len(model._d4_gains) > 0:
                    g_min, g_max = min(model._d4_gains), max(model._d4_gains)
                    th_min, th_max = min(model._d4_ths), max(model._d4_ths)
                    logger.info(f"  [D4] Gain saturation check: Gain range [{g_min:.3f}, {g_max:.3f}], Eff-Threshold range [{th_min:.5f}, {th_max:.5f}]")
                suffix = f"_{full_method_name}"
                
                traj_json_path = os.path.join(args.log_dir, f'traj_{ctype}_{sev}{suffix}.json')
                with open(traj_json_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                    
                save_graphic(os.path.join(args.log_dir, f'traj_{ctype}_{sev}{suffix}.png'), f'{ctype} Sev {sev}', metrics)
                
                with open(os.path.join(args.log_dir, f'results{suffix}.json'), 'w') as f:
                    json.dump({'mIoU': results_miou, 'Accuracy': results_acc}, f, indent=4)
                    
                with open(os.path.join(args.log_dir, 'global_results.json'), 'w') as f:
                    json.dump(global_results, f, indent=4)
                    
                if args.dump_features and hasattr(model, '_feature_dump_list') and len(model._feature_dump_list) > 0:
                    dump_path = os.path.join(args.log_dir, f'features_dump_{ctype}_{sev}{suffix}.pt')
                    logger.info(f"Saving feature dump ({len(model._feature_dump_list)} frames) to {dump_path}...")
                    torch.save(model._feature_dump_list, dump_path)
                    model._feature_dump_list = []
            else:
                logger.info(f"No valid frames evaluated for {ctype}-{sev}")

        total_evals = sum(len(sev_dict) for sev_dict in results_miou.values())
        if total_evals == 0 and not args.dry_run:
            logger.error(f"CRITICAL ERROR: No evaluation outcomes recorded in results_miou for {full_method_name}. Check for silent exceptions during evaluation.")
            raise RuntimeError(f"Evaluation failed to record any metrics for {full_method_name}.")

        suffix = f"_{full_method_name}"
        save_degradation_plot(os.path.join(args.log_dir, f'degradation_miou{suffix}.png'), 'KITTI-C', results_miou, metric='mIoU', baseline_val=None)
        save_degradation_plot(os.path.join(args.log_dir, f'degradation_acc{suffix}.png'), 'KITTI-C', results_acc, metric='Accuracy', baseline_val=None)

if __name__ == "__main__":
    main()
