import argparse
import logging
import os
import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from common.laserscan import SemLaserScan, LaserScan
from dataset.kitti.parser import Parser
import unsup_main
from unsup_main import train_extractor, train_hdc, extract_metrics_from_conf_matrix, setup_logger, save_graphic, load_hdc_model

def corrupt_beam(points, severity):
    distances = np.linalg.norm(points[:, :3], axis=1)
    pitch = np.arcsin(points[:, 2] / (distances + 1e-6))
    bins = np.linspace(np.min(pitch), np.max(pitch), 65)
    ring_ids = np.digitize(pitch, bins)
    drop_fraction = 0.05 * severity 
    unique_rings = np.unique(ring_ids)
    num_drop = int(len(unique_rings) * drop_fraction)
    dropped_rings = np.random.choice(unique_rings, num_drop, replace=False)
    mask = ~np.isin(ring_ids, dropped_rings)
    return points[mask], mask, 0

def corrupt_crosstalk(points, severity):
    num_points = len(points)
    noise_fraction = 0.02 * severity 
    num_noise = int(num_points * noise_fraction)
    min_bounds = np.min(points[:, :3], axis=0)
    max_bounds = np.max(points[:, :3], axis=0)
    noise_xyz = np.random.uniform(min_bounds, max_bounds, size=(num_noise, 3))
    noise_intensity = np.random.uniform(0, 0.1, size=(num_noise, 1)) 
    noise_points = np.hstack((noise_xyz, noise_intensity))
    return np.vstack((points, noise_points)), np.ones(len(points), dtype=bool), num_noise

def corrupt_fog(points, severity):
    distances = np.linalg.norm(points[:, :3], axis=1)
    beta = 0.005 * severity 
    survival_prob = np.exp(-beta * distances)
    random_draw = np.random.uniform(0, 1, size=len(points))
    mask = random_draw < survival_prob
    return points[mask], mask, 0

def corrupt_echo(points, severity):
    intensity_threshold = np.percentile(points[:, 3], 90)
    high_ref_mask = points[:, 3] > intensity_threshold
    echo_points = points[high_ref_mask].copy()
    shift_multiplier = 1.0 + (0.1 * severity) 
    echo_points[:, :3] = echo_points[:, :3] * shift_multiplier
    echo_points[:, 3] = echo_points[:, 3] * 0.5 
    return np.vstack((points, echo_points)), np.ones(len(points), dtype=bool), len(echo_points)

def corrupt_motion(points, severity):
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    timeline = (azimuth - np.min(azimuth)) / (np.max(azimuth) - np.min(azimuth) + 1e-6)
    max_translation = 0.2 * severity 
    blur_shift = np.outer(timeline, np.array([max_translation, 0, 0])) 
    points[:, :3] += blur_shift
    return points, np.ones(len(points), dtype=bool), 0

def corrupt_snow(points, severity):
    num_flakes = 1000 * severity
    flake_xyz = np.random.uniform(-10, 10, size=(num_flakes, 3)) 
    flake_intensity = np.random.uniform(0.5, 1.0, size=(num_flakes, 1)) 
    snowflakes = np.hstack((flake_xyz, flake_intensity))
    ground_mask = points[:, 2] < -1.0 
    drop_prob = 0.1 * severity
    survive_ground = np.random.uniform(0, 1, size=np.sum(ground_mask)) > drop_prob
    final_points_mask = np.ones(len(points), dtype=bool)
    final_points_mask[ground_mask] = survive_ground
    filtered_points = points[final_points_mask]
    return np.vstack((filtered_points, snowflakes)), final_points_mask, num_flakes

def corrupt_cross_sensor(points, severity):
    distances = np.linalg.norm(points[:, :3], axis=1)
    pitch = np.arcsin(points[:, 2] / (distances + 1e-6))
    
    # 64 bins simulate the typical 64-beam sensor
    bins = np.linspace(np.min(pitch), np.max(pitch), 65) 
    ring_ids = np.digitize(pitch, bins)
    
    # Severity 1, 2, 3 maps to keeping 1/2, 1/4, or 1/8 of the beams
    step = 2 ** severity 
    
    # Keep only beams where the ring ID aligns with the step size
    mask = (ring_ids % step == 0)
    
    return points[mask], mask, 0

def apply_corruption(points, corruption_type, severity):
    if corruption_type == 'beam':
        return corrupt_beam(points, severity)
    elif corruption_type == 'cross_sensor':
        return corrupt_cross_sensor(points, severity)
    elif corruption_type == 'crosstalk':
        return corrupt_crosstalk(points, severity)
    elif corruption_type == 'fog':
        return corrupt_fog(points, severity)
    elif corruption_type == 'echo':
        return corrupt_echo(points, severity)
    elif corruption_type == 'motion':
        return corrupt_motion(points, severity)
    elif corruption_type == 'snow':
        return corrupt_snow(points, severity)
    return points, np.ones(len(points), dtype=bool), 0

class LiDARCorruptionWrapper(Dataset):
    def __init__(self, base_dataset, corruption_type=None, severity=1):
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        original_open = SemLaserScan.open_scan
        original_laser_open = LaserScan.open_scan
        
        wrapper_self = self
        
        original_label_open = SemLaserScan.open_label

        def patched_open_scan(scan_self, filename):
            scan = np.fromfile(filename, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            
            if wrapper_self.corruption_type:
                scan, mask, added_count = apply_corruption(scan, wrapper_self.corruption_type, wrapper_self.severity)
                scan_self.corruption_mask = mask
                scan_self.corruption_added = added_count
            else:
                scan_self.corruption_mask = None
                scan_self.corruption_added = 0
                
            points = scan[:, 0:3]
            remissions = scan[:, 3]
            
            if scan_self.drop_points is not False:
                scan_self.points_to_drop = np.random.randint(0, len(points)-1, int(len(points)*scan_self.drop_points))
                points = np.delete(points, scan_self.points_to_drop, axis=0)
                remissions = np.delete(remissions, scan_self.points_to_drop)

            scan_self.set_points(points, remissions)

        def patched_open_label(scan_self, filename):
            if not any(filename.endswith(ext) for ext in scan_self.EXTENSIONS_LABEL):
                raise RuntimeError("Filename extension is not valid label file.")
            label = np.fromfile(filename, dtype=np.int32)
            label = label.reshape((-1))
            
            if scan_self.drop_points is not False:
                label = np.delete(label, scan_self.points_to_drop)
                
            if getattr(scan_self, 'corruption_mask', None) is not None:
                label = label[scan_self.corruption_mask]
                
            if getattr(scan_self, 'corruption_added', 0) > 0:
                fake_label = np.zeros(scan_self.corruption_added, dtype=label.dtype)
                label = np.concatenate([label, fake_label])
                
            scan_self.set_label(label)

        SemLaserScan.open_scan = patched_open_scan
        LaserScan.open_scan = patched_open_scan
        SemLaserScan.open_label = patched_open_label
        
        try:
            data = self.base_dataset[idx]
        finally:
            SemLaserScan.open_scan = original_open
            LaserScan.open_scan = original_laser_open
            SemLaserScan.open_label = original_label_open
            
        return data

def evaluate_and_adapt(model, target_dataloader, device, eval_only=False):
    """Helper method executing the forward/eval/adapt cycle."""
    miou_history = []
    acc_history = []
    num_classes = model.num_classes
    cumulative_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for batch_data in tqdm(target_dataloader, desc="Adapting", leave=False):
        proj_in = batch_data[0].to(device)
        proj_labels = batch_data[2].to(device).view(-1)
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
                
            cumulative_miou, cumulative_acc, _ = extract_metrics_from_conf_matrix(cumulative_confusion_matrix)
            miou_history.append(cumulative_miou)
            acc_history.append(cumulative_acc)
            # Adapt: Inference Update
            if not eval_only:
                model.train()
                if hasattr(model, 'G_d'):  # Duck typing for D3CTTA
                    model.inference_update(
                        h=h,
                        predictions=predictions,
                        xyz=proj_xyz
                    )
                else:
                    model.inference_update(
                        proj_in,
                        learning_rate=0.001,
                        distance_sensitivity=3.0,
                        thresholds=[0.45, 0.80]
                    )
            
    return {"mIoU": miou_history, "Accuracy": acc_history}

def pretrain_pipeline(ARCH, DATA, data_dir, pretrained_path, return_trainer=False, skip_extractor=False, resume_path=None, hdc_epochs=10):
    import unsup_main
    log_base = os.path.dirname(pretrained_path)
    os.makedirs(log_base, exist_ok=True)
    
    unsup_main.LOG_DIR = log_base
    unsup_main.MODEL_DIR = log_base
    unsup_main.HDC_SAVE_PATH = os.path.join(log_base, "hdc.pth")
    unsup_main.HDC_SUB_PATH = pretrained_path

    if not skip_extractor:
        ARCH["train"]["batch_size"] = 24
        print(f"Pretraining feature extractor on {data_dir}...")
        trainer = train_extractor(ARCH, DATA, data_dir=data_dir, return_trainer=True, resume_path=resume_path)
    else:
        print(f"Skipping feature extractor pretraining...")
        trainer = None
    
    ARCH["train"]["batch_size"] = 6
    print(f"Pretraining HDC density model on {data_dir} for {hdc_epochs} epochs...")
    model, _ = train_hdc(ARCH, DATA, epochs=hdc_epochs, data_dir=data_dir, return_extractor=True)
    
    print("Initializing subclusters...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = Parser(root=data_dir,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=None,
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=ARCH["train"]["batch_size"],
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=True)
    
    dataloader = parser.get_train_set()
    model.init_subclusters(dataloader)
    
    torch.save(model.state_dict(), pretrained_path)
    print(f"Subcluster Initialized Model saved to {pretrained_path}")
    
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

def load_d3ctta_model(path):
    print(f"Loading pretrained feature extractor for D3CTTA from {path}...")
    from modules.network.ResNet import ResNet_34
    from modules.D3CTTA import D3CTTA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    se_path = os.path.join(os.path.dirname(path), "SENet_valid_best")
    if os.path.exists(se_path):
        tmp_dict = torch.load(se_path, map_location='cpu')
        NUM_CLASSES = tmp_dict['state_dict']['semantic_output.bias'].shape[0]
    else:
        NUM_CLASSES = 17 # Fallback for NuScenes
    feature_extractor = ResNet_34(NUM_CLASSES, aux=False, use_adaptor=True)
    
    se_path = os.path.join(os.path.dirname(path), "SENet_valid_best")
    if os.path.exists(se_path):
        w_dict = torch.load(se_path, map_location=device)
        feature_extractor.load_state_dict(w_dict['state_dict'], strict=False)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    model = D3CTTA(feature_extractor, num_classes=NUM_CLASSES)
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Test Unsupervised Updates on NuScenes-C")
    parser.add_argument('--pretrain', action='store_true', help='Pretrain the model on standard NuScenes')
    parser.add_argument('--skip_extractor', action='store_true', help='Skip feature extractor pretraining and only retrain the HDC model')
    parser.add_argument('--pretrained_path', type=str, default='logs/nusc_pretrain/hdc_sub.pth', help='Path to load pretrained model')
    parser.add_argument('--log_dir', type=str, default='logs/nusc_c_test', help='Directory to save logs and graphics')
    parser.add_argument('--compare', action='store_true', help='Use D3CTTA with pretrained feature extractor instead of HDC')
    parser.add_argument('--continue_pretrain', action='store_true', help='Resume pretraining from the existing pretrained_path')
    parser.add_argument('--hdc_epochs', type=int, default=30, help='Number of epochs to train the HDC density model')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, 'nusc_c.log'))

    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p-gen.yml", 'r'))
        DATA = yaml.safe_load(open("config/labels/nuscenes_new.yaml", 'r'))
    except Exception as e:
        logger.error(f"Error loading configs: {e}")
        return

    data_dir = "/mnt/alpha/jmfleming/nuscenes_kitti"

    if args.pretrain:
        logger.info(f"Starting Pretraining on standard NuScenes at {data_dir}...")
        resume_dir = os.path.dirname(args.pretrained_path) if args.continue_pretrain else None
        model, trainer = pretrain_pipeline(ARCH, DATA, data_dir=data_dir, pretrained_path=args.pretrained_path, return_trainer=True, skip_extractor=args.skip_extractor, resume_path=resume_dir, hdc_epochs=args.hdc_epochs)
        
        if trainer is not None:
            opt_path = os.path.join(os.path.dirname(args.pretrained_path), 'feature_optimizer.pth')
            torch.save(trainer.optimizer.state_dict(), opt_path)
            logger.info(f"Successfully pretrained model on NuScenes. Optimizer state saved to {opt_path}")
        
        if args.compare:
            logger.info("Compare flag enabled. Loading D3CTTA model...")
            model = load_d3ctta_model(args.pretrained_path)
    else:
        if args.compare:
            logger.info(f"Loading pretrained feature extractor for D3CTTA from {args.pretrained_path}...")
            model = load_d3ctta_model(args.pretrained_path)
        else:
            logger.info(f"Loading pretrained model from {args.pretrained_path}")
            model = load_hdc_model(args.pretrained_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    corruptions = ['beam', 'cross_sensor', 'crosstalk', 'fog', 'echo', 'motion', 'snow']
    severities = [1, 2, 3, 4, 5]

    results_miou = {c: {} for c in corruptions}
    results_acc = {c: {} for c in corruptions}

    logger.info("Evaluating on clean baseline (Sunny/Original) dataset...")
    baseline_parser = Parser(root=data_dir,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=None,
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=1,
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=False)
    baseline_loader = DataLoader(baseline_parser.validloader.dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
    
    baseline_metrics = evaluate_and_adapt(model, baseline_loader, device, eval_only=True)
    if len(baseline_metrics["mIoU"]) > 0:
        logger.info(f"Clean Baseline: mIoU={baseline_metrics['mIoU'][-1]:.4f}, Acc={baseline_metrics['Accuracy'][-1]:.4f}")
    
    for ctype in corruptions:
        for sev in severities:
            logger.info(f"Testing {ctype} severity {sev}")
            
            parser_obj = Parser(root=data_dir,
                            train_sequences=DATA["split"]["train"],
                            valid_sequences=DATA["split"]["valid"],
                            test_sequences=None,
                            labels=DATA["labels"],
                            color_map=DATA["color_map"],
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"],
                            batch_size=1,
                            workers=ARCH["train"]["workers"],
                            gt=True,
                            shuffle_train=False)
            
            target_dataset = parser_obj.validloader.dataset
            corrupted_dataset = LiDARCorruptionWrapper(target_dataset, corruption_type=ctype, severity=sev)
            target_dataloader = DataLoader(corrupted_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
            
            if args.compare:
                model = load_d3ctta_model(args.pretrained_path)
            else:
                model = load_hdc_model(args.pretrained_path)
            
            metrics = evaluate_and_adapt(model, target_dataloader, device)
            
            if len(metrics["mIoU"]) > 0:
                initial_miou = metrics["mIoU"][0]
                final_miou = metrics["mIoU"][-1]
                initial_acc = metrics["Accuracy"][0]
                final_acc = metrics["Accuracy"][-1]
                
                results_miou[ctype][sev] = (initial_miou, final_miou)
                results_acc[ctype][sev] = (initial_acc, final_acc)
                
                logger.info(f"Result for {ctype}-{sev}: Initial mIoU={initial_miou:.4f} -> Final={final_miou:.4f}, Initial Acc={initial_acc:.4f} -> Final={final_acc:.4f}")
                suffix = "_d3ctta" if args.compare else ""
                
                traj_json_path = os.path.join(args.log_dir, f'traj_{ctype}_{sev}{suffix}.json')
                with open(traj_json_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                    
                save_graphic(os.path.join(args.log_dir, f'traj_{ctype}_{sev}{suffix}.png'), f'{ctype} Sev {sev}', metrics)
            else:
                logger.info(f"No valid frames evaluated for {ctype}-{sev}")

    suffix = "_d3ctta" if args.compare else ""
    
    baseline_miou = baseline_metrics['mIoU'][-1] if len(baseline_metrics.get('mIoU', [])) > 0 else None
    baseline_acc = baseline_metrics['Accuracy'][-1] if len(baseline_metrics.get('Accuracy', [])) > 0 else None
    
    save_degradation_plot(os.path.join(args.log_dir, f'degradation_miou{suffix}.png'), 'NuScenes-C', results_miou, metric='mIoU', baseline_val=baseline_miou)
    save_degradation_plot(os.path.join(args.log_dir, f'degradation_acc{suffix}.png'), 'NuScenes-C', results_acc, metric='Accuracy', baseline_val=baseline_acc)
    
    with open(os.path.join(args.log_dir, f'results{suffix}.json'), 'w') as f:
        json.dump({'mIoU': results_miou, 'Accuracy': results_acc, 'Baseline_mIoU': baseline_miou, 'Baseline_Acc': baseline_acc}, f, indent=4)

if __name__ == "__main__":
    main()
