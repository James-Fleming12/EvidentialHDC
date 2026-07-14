import argparse
import logging
import os
import json
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from common.laserscan import SemLaserScan, LaserScan
from dataset.kitti.parser import Parser
import unsup_main
from unsup_main import train_extractor, train_hdc, extract_metrics_from_conf_matrix, setup_logger, save_graphic
from modules.HDC_utils import UQModel
from modules.HDC_utils import set_knn_model

NUM_CLASSES = 7
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
# Note on Severity: D3CTTA evaluates on "moderate" severity. 
# Depending on Robo3D version, this maps to severity 2 (light/moderate/heavy) or 3 (1-5 scale).
# When comparing to D3CTTA, ensure you run with the severity integer that maps to 'moderate'.
SEVERITY_MAP = {1: 'light', 2: 'moderate', 3: 'heavy', 4: 'extreme'}

CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS_KITTI = "config/labels/semantic-kitti.yaml"  # The 7-class mapped version from D3CTTA
CONFIG_LABELS_KITTI_ALL = "config/labels/semantic-kitti-all.yaml"  # Standard 17 classes

def evaluate_and_adapt(model, target_dataloader, device, eval_only=False, update_method='density', dry_run=False, custom_update_fn=None):
    miou_history = []
    acc_history = []
    iou_per_class_history = []
    num_classes = model.num_classes
    cumulative_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for batch_idx, batch_data in enumerate(tqdm(target_dataloader, desc="Adapting", leave=False)):
        if dry_run and batch_idx >= 2:
            break
        
        proj_in = batch_data[0].to(device)
        proj_labels = batch_data[2].to(device).view(-1)
        if batch_idx == 0:
            print(f"DEBUG: len(batch_data) = {len(batch_data)}")
            if len(batch_data) > 10:
                print(f"DEBUG: batch_data[10].shape = {batch_data[10].shape}")
        proj_xyz = batch_data[10].to(device) if len(batch_data) > 10 else None
        
        if proj_in.shape[1] > 0:
            model.eval()
            with torch.no_grad():
                logits, sims, indices, h = model(proj_in)
                predictions = torch.argmax(logits, dim=1)
                selected_labels = proj_labels[indices]
                
                mask = (selected_labels >= 0) & (selected_labels < num_classes)
                if mask.any():
                    hist = torch.bincount(
                        num_classes * selected_labels[mask] + predictions[mask], 
                        minlength=num_classes ** 2
                    ).reshape(num_classes, num_classes)
                    cumulative_confusion_matrix += hist
                
            cumulative_miou, cumulative_acc, cumulative_iou_per_class = extract_metrics_from_conf_matrix(cumulative_confusion_matrix)
            miou_history.append(cumulative_miou)
            acc_history.append(cumulative_acc)
            iou_per_class_history.append(cumulative_iou_per_class)
            
            # Adapt: Inference Update
            if not eval_only:
                model.eval()
                if update_method == 'epistemic':
                    # TODO: Implement Pillar 1 (Epistemic UQ)
                    pass
                elif update_method == 'consistency':
                    # TODO: Implement Pillar 1 (Spatial/Temporal Consistency)
                    pass
                elif update_method == 'hybrid':
                    # TODO: Implement Pillar 1 (Fusion)
                    pass
                elif update_method == 'hybrid_balanced':
                    # TODO: Implement Pillar 3 (Subcluster Ledger)
                    pass
                elif update_method == 'multi_view':
                    # TODO: Implement Pillar 2 (Multi-view TTA)
                    pass
                elif update_method == 'oracle':
                    # TODO: Implement Oracle gating
                    pass
    
    avg_firing_rate = 0.0
    if hasattr(model, '_firing_log') and len(model._firing_log) > 0:
        avg_firing_rate = sum(model._firing_log) / len(model._firing_log)
        model._firing_log = []
        
    avg_update_magnitude = 0.0
    if hasattr(model, '_update_magnitude_log') and len(model._update_magnitude_log) > 0:
        avg_update_magnitude = sum(model._update_magnitude_log) / len(model._update_magnitude_log)
        model._update_magnitude_log = []
        
    return {"mIoU": miou_history, "Accuracy": acc_history, "IoU_per_class": iou_per_class_history, "FiringRate": avg_firing_rate, "UpdateMagnitude": avg_update_magnitude}


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
        print(f"Skipping feature extractor pretraining...")
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


def load_hdc_model(path, num_classes=NUM_CLASSES):
    print(f"Loading pretrained HDC model from {path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
    modeldir = os.path.dirname(path)

    model = set_knn_model(ARCH, modeldir, 'rp', 0, 0, num_classes, device)
    
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.to(device)
    return model

def populate_knn_bank(model, data_dir, arch_cfg, data_cfg, device):
    bank_path = os.path.join(os.path.dirname(data_dir), "knn_bank.pt")
    if os.path.exists(bank_path):
        print(f"Loading pre-populated k-NN bank from {bank_path}...")
        bank_data = torch.load(bank_path, map_location=device)
        model.bank = bank_data
        print("Calibrating robust thresholds from source data...")
        model.calibrate_thresholds(coverage=0.50)
        return

    print(f"Populating k-NN bank from {data_dir}...")
    parser = Parser(root=data_dir,
                    train_sequences=data_cfg["split"]["train"],
                    valid_sequences=data_cfg["split"]["valid"],
                    test_sequences=None,
                    labels=data_cfg["labels"],
                    color_map=data_cfg.get("color_map", {}),
                    learning_map=data_cfg["learning_map"],
                    learning_map_inv=data_cfg["learning_map_inv"],
                    sensor=arch_cfg["dataset"]["sensor"],
                    max_points=arch_cfg["dataset"]["max_points"],
                    batch_size=1,
                    workers=arch_cfg["train"]["workers"],
                    gt=True,
                    shuffle_train=True) 
    
    dataloader = DataLoader(parser.trainloader.dataset, batch_size=1, shuffle=True, num_workers=4)
    model.eval()
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Populating Bank"):
            proj_in = batch_data[0].to(device)
            proj_labels = batch_data[2].to(device).view(-1)
            
            if proj_in.shape[1] > 0:
                enc, _, _ = model.encode(proj_in)
                original_x = proj_in.permute(0, 2, 3, 1).contiguous().reshape(-1, proj_in.shape[1])
                valid_mask = torch.any(original_x != 0, dim=1)
                
                if not torch.any(valid_mask):
                    continue
                    
                enc = enc[valid_mask]
                labels = proj_labels[valid_mask]
                
                model.update_bank(enc, labels)
                
                if all(model.bank[c].shape[0] >= model.bank_size for c in range(model.num_classes)):
                    break
                    
        print("Calibrating robust thresholds from source data...")
        model.calibrate_thresholds(coverage=0.50)
        
    print(f"Saving populated bank to {bank_path}...")
    torch.save(model.bank, bank_path)

def main():
    parser = argparse.ArgumentParser(description="Test Unsupervised Updates on KITTI-C")
    parser.add_argument('--pretrain', action='store_true', help='Run pretraining on SemanticKITTI before evaluating')
    parser.add_argument('--chunked', action='store_true', help='Use D3CTTA chunked protocol: continuous adaptation across disjoint 1/7th splits instead of full independent sequences.')
    parser.add_argument('--reset_per_corruption', action='store_true', help='Reset the model to the clean pretrained weights before adapting on each corruption (even when using chunks).')
    parser.add_argument('--skip_extractor', action='store_true', help='Skip feature extractor pretraining and only retrain the HDC model')
    parser.add_argument('--pretrained_path', type=str, default='logs/kitti_pretrain/hdc_sub.pth', help='Path to load pretrained model')
    parser.add_argument('--log_dir', type=str, default='logs/kitti_c_test', help='Directory to save logs and graphics')
    parser.add_argument('--method', type=str, choices=['frozen', 'epistemic', 'consistency', 'hybrid', 'hybrid_balanced', 'multi_view', 'oracle', 'all'], default='frozen', help='Method to test.')
    parser.add_argument('--dry_run', action='store_true', help='Run only 2 batches per condition to quickly verify no crashes will occur.')
    parser.add_argument('--continue_pretrain', action='store_true', help='Resume pretraining from the existing pretrained_path')
    parser.add_argument('--continue', dest='continue_epochs', type=int, default=0, help='Continue feature extractor training for this many epochs, reinitialize HDC, and perform adaptation')
    parser.add_argument('--extractor_epochs', type=int, default=60, help='Number of epochs to train the feature extractor')
    parser.add_argument('--hdc_epochs', type=int, default=15, help='Number of epochs to train the HDC density model')
    parser.add_argument('--severity', type=int, default=3, help='Severity level for corruptions')
    parser.add_argument('--kitti_dir', type=str, default='/mnt/alpha/jmfleming/KITTI', help='Path to SemanticKITTI dataset for pretraining')
    parser.add_argument('--kittic_dir', type=str, default='/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C', help='Path to real SemanticKITTI-C dataset')
    parser.add_argument('--corruptions', type=str, default=None, help='Comma separated list of corruptions to test. Defaults to all 8.')
    args = parser.parse_args()

    if args.continue_epochs > 0:
        args.pretrain = True
        args.continue_pretrain = True
        args.extractor_epochs = args.continue_epochs

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, 'kitti_c.log'))

    global NUM_CLASSES
    NUM_CLASSES = 17
        
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
    methods_to_run = ['prototype', 'knn'] if args.method == 'all' else [args.method]
    
    global_results = {
        'mIoU': {m: {c: {} for c in CORRUPTIONS} for m in methods_to_run},
        'Accuracy': {m: {c: {} for c in CORRUPTIONS} for m in methods_to_run},
    }
    
    shared_init_metrics = {}
    
    # Load dataset once and partition it to find chunks
    # Note on Protocol: D3CTTA divides the valid set into 7 disjoint chunks (1 per corruption).
    # This evaluates each corruption on 1/7 of the validation set (e.g., ~581 frames) instead 
    # of the full set. We are preserving this behavior to identically match their protocol. 
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

    for current_method in methods_to_run:
        logger.info(f"=========================================")
        logger.info(f"Starting Evaluation for Method: {current_method}")
        logger.info(f"=========================================")
        
        active_corruptions = CORRUPTIONS
        if args.corruptions:
            active_corruptions = [c.strip() for c in args.corruptions.split(',')]

        results_miou = {c: {} for c in active_corruptions}
        results_acc = {c: {} for c in active_corruptions}

        model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES)
        if current_method == 'knn':
            populate_knn_bank(model, args.kitti_dir, ARCH, DATA, device)
            # Make a clean backup of the bank to restore on reset
            clean_bank = {k: v.clone() for k, v in model.bank.items()}

        for i, ctype in enumerate(active_corruptions):
            if args.reset_per_corruption and args.chunked:
                logger.info("Resetting model to clean pretrained weights for this corruption.")
                model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES)
                if current_method == 'knn':
                    model.bank = {k: v.clone() for k, v in clean_bank.items()}
                
            logger.info(f"Testing {ctype} severity {sev} (Chunk {i+1}/{len(active_corruptions)})")
            
            # Map severity integer to Robo3D folder name
            sev_str = SEVERITY_MAP.get(sev, 'moderate')
            
            # NOTE (PLAN): The Parser natively expects a "sequences" folder inside root. 
            # (e.g., SemanticKITTI-C/fog/moderate/sequences/08/velodyne)
            # If the download layout is just SemanticKITTI-C/fog/moderate/velodyne, this will fail.
            # Plan: We will either symlink the paths or create a custom KITTI-C Parser subclass
            # that alters the root string logic once the exact directory layout is confirmed.
            corruption_root = os.path.join(args.kittic_dir, ctype, sev_str)
            seq_dir = os.path.join(corruption_root, "sequences")
            if not os.path.exists(seq_dir):
                logger.info(f"Directory structure doesn't match standard KITTI. Creating 'sequences/08' symlink in {corruption_root}...")
                os.makedirs(seq_dir, exist_ok=True)
                # Create sequences/08 that points to the parent directory (corruption_root)
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
                full_corruption_dataset = parser_obj.validloader.dataset
            except Exception as e:
                logger.error(f"Failed to load KITTI-C corruption dataset at {corruption_root}: {e}")
                continue
            
            # Prevent silent misalignment bugs by ensuring corrupted frame count matches baseline clean chunk length
            assert len(full_corruption_dataset) == total_len, (
                f"Length mismatch: Clean baseline length is {total_len}, "
                f"but {ctype}-{sev_str} length is {len(full_corruption_dataset)}. "
                f"Chunks will misalign."
            )
            
            if not args.chunked:
                # Standard protocol: full sequence, independent adaptation
                chunk_dataset = full_corruption_dataset
                # Reset model before each corruption
                model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES)
                if current_method == 'knn':
                    model.bank = {k: v.clone() for k, v in clean_bank.items()}
            else:
                # D3CTTA protocol: chunks, continuous adaptation
                chunk_dataset = torch.utils.data.Subset(full_corruption_dataset, chunks[i])
            
            target_dataloader = DataLoader(chunk_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
            
            try:
                if not args.chunked or args.reset_per_corruption:
                    # Pass 1: True Initial (Frozen on chunk)
                    if (ctype, sev) not in shared_init_metrics:
                        logger.info("  -> Pass 1: Computing True Initial metrics (Frozen)")
                        init_metrics = evaluate_and_adapt(model, target_dataloader, device, eval_only=True, dry_run=args.dry_run)
                        shared_init_metrics[(ctype, sev)] = init_metrics
                    else:
                        logger.info("  -> Pass 1: Reusing cached True Initial metrics (Frozen)")
                        init_metrics = shared_init_metrics[(ctype, sev)]
                    
                    # Pass 2: Adapt (only if method is not frozen)
                    if current_method != 'frozen':
                        logger.info("  -> Pass 2: Adapting model weights")
                        adapt_metrics = evaluate_and_adapt(model, target_dataloader, device, eval_only=False, update_method=current_method, dry_run=args.dry_run)
                    else:
                        adapt_metrics = init_metrics
                        
                    # Pass 3: True Final (Frozen on chunk using adapted weights)
                    logger.info("  -> Pass 3: Computing True Final metrics (Frozen)")
                    final_metrics = evaluate_and_adapt(model, target_dataloader, device, eval_only=True, dry_run=args.dry_run)
                    
                    # We only care about the absolute end of the frozen evaluations for the sequence
                    metrics = adapt_metrics  # Just for the trajectory json
                    if len(init_metrics["mIoU"]) > 0:
                        initial_miou = init_metrics["mIoU"][-1]
                        final_miou = final_metrics["mIoU"][-1]
                        initial_acc = init_metrics["Accuracy"][-1]
                        final_acc = final_metrics["Accuracy"][-1]
                    else:
                        initial_miou = final_miou = initial_acc = final_acc = 0.0
                        
                    firing_rate_str = ""
                    if "FiringRate" in adapt_metrics:
                        firing_rate_str = f", FiringRate={adapt_metrics['FiringRate']*100:.2f}%"
                        if "UpdateMagnitude" in adapt_metrics:
                            firing_rate_str += f", UpdateMag={adapt_metrics['UpdateMagnitude']:.4f}"
                else:
                    # Original single-pass continuous evaluation
                    metrics = evaluate_and_adapt(model, target_dataloader, device, eval_only=(current_method == 'frozen'), update_method=current_method, dry_run=args.dry_run)
                    if len(metrics["mIoU"]) > 0:
                        initial_miou = metrics["mIoU"][0]
                        final_miou = metrics["mIoU"][-1]
                        initial_acc = metrics["Accuracy"][0]
                        final_acc = metrics["Accuracy"][-1]
                    else:
                        initial_miou = final_miou = initial_acc = final_acc = 0.0
                        
                    firing_rate_str = ""
                    if "FiringRate" in metrics:
                        firing_rate_str = f", FiringRate={metrics['FiringRate']*100:.2f}%"
                        if "UpdateMagnitude" in metrics:
                            firing_rate_str += f", UpdateMag={metrics['UpdateMagnitude']:.4f}"
            except Exception as e:
                logger.error(f"FATAL ERROR during {ctype} sev {sev} ({current_method}): {e}")
                logger.info("Skipping to next cell to protect the overnight run...")
                continue
            
            if len(metrics["mIoU"]) > 0:
                results_miou[ctype][sev] = (initial_miou, final_miou)
                results_acc[ctype][sev] = (initial_acc, final_acc)
                
                global_results['mIoU'][current_method][ctype][sev] = (initial_miou, final_miou)
                global_results['Accuracy'][current_method][ctype][sev] = (initial_acc, final_acc)
                
                logger.info(f"Result for {ctype}-{sev}: Initial mIoU={initial_miou:.4f} -> Final={final_miou:.4f}, Initial Acc={initial_acc:.4f} -> Final={final_acc:.4f}{firing_rate_str}")
                suffix = f"_{current_method}"
                
                traj_json_path = os.path.join(args.log_dir, f'traj_{ctype}_{sev}{suffix}.json')
                with open(traj_json_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                    
                save_graphic(os.path.join(args.log_dir, f'traj_{ctype}_{sev}{suffix}.png'), f'{ctype} Sev {sev}', metrics)
                
                with open(os.path.join(args.log_dir, f'results{suffix}.json'), 'w') as f:
                    json.dump({'mIoU': results_miou, 'Accuracy': results_acc}, f, indent=4)
                    
                with open(os.path.join(args.log_dir, 'global_results.json'), 'w') as f:
                    json.dump(global_results, f, indent=4)
            else:
                logger.info(f"No valid frames evaluated for {ctype}-{sev}")

        suffix = f"_{current_method}"
        save_degradation_plot(os.path.join(args.log_dir, f'degradation_miou{suffix}.png'), 'KITTI-C', results_miou, metric='mIoU', baseline_val=None)
        save_degradation_plot(os.path.join(args.log_dir, f'degradation_acc{suffix}.png'), 'KITTI-C', results_acc, metric='Accuracy', baseline_val=None)

if __name__ == "__main__":
    main()
