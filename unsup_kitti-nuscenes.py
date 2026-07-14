import argparse
import logging
import os
import json
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
from torch.utils.data import Dataset, DataLoader

from dataset.kitti.parser import Parser
import unsup_main
from unsup_main import train_extractor, train_hdc, extract_metrics_from_conf_matrix, setup_logger, save_graphic
from modules.HDC_utils import UQModel
from modules.aug_model import AugModel

NUM_CLASSES = 17  # 16 shared cross-dataset classes + 1 ignored class (0)
CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS_KITTI = "config/labels/semantic-kitti-all.yaml"  # 16-class mapping
CONFIG_LABELS_NUSCENES = "config/labels/nuscenes_new.yaml"  # 16-class mapping

def evaluate_and_adapt(model, target_dataloader, device, eval_only=False, update_method='density', dry_run=False):
    miou_history = []
    acc_history = []
    iou_per_class_history = []
    num_classes = model.num_classes
    cumulative_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for batch_idx, batch_data in enumerate(tqdm(target_dataloader, desc="Adapting on NuScenes", leave=False)):
        if dry_run and batch_idx >= 2:
            break
        
        proj_in = batch_data[0].to(device)
        proj_labels = batch_data[2].to(device).view(-1)
        if batch_idx == 0:
            print(f"DEBUG (NuScenes Parse): len(batch_data) = {len(batch_data)}")
            if len(batch_data) > 10:
                print(f"DEBUG (NuScenes Parse): batch_data[10].shape (proj_xyz) = {batch_data[10].shape}")
            else:
                print("WARNING: len(batch_data) <= 10. proj_xyz is MISSING! Density/ExpA will degrade.")
                
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
    return {"mIoU": miou_history, "Accuracy": acc_history, "IoU_per_class": iou_per_class_history}


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


def load_hdc_model(path, num_classes=NUM_CLASSES):
    print(f"Loading pretrained HDC model from {path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
    modeldir = os.path.dirname(path)

    model = AugModel(ARCH, modeldir, 'rp', 0, 0, num_classes, device, subcluster_type='continuous')
    
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print("\n" + "="*80)
            print("CRITICAL WARNING (Stale Checkpoint Detected!):")
            print(f"You are trying to load a checkpoint from {path}.")
            print("This script expects a model trained on the 16-class cross-dataset taxonomy (NUM_CLASSES=17).")
            print("The checkpoint you provided was likely trained on a different taxonomy (e.g., the 7-class D3CTTA mapping).")
            print("Please delete the stale checkpoint, or run with --pretrain to generate a new one aligned with NuScenes.")
            print("="*80 + "\n")
        raise e
        
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Test Unsupervised Updates for KITTI to NuScenes Cross-Dataset Adaptation")
    parser.add_argument('--pretrain', action='store_true', help='Pretrain the model on SemanticKITTI dataset')
    parser.add_argument('--eval_kitti', action='store_true', help='Evaluate frozen model on KITTI validation set as a baseline check')
    parser.add_argument('--skip_extractor', action='store_true', help='Skip feature extractor pretraining and only retrain the HDC model')
    parser.add_argument('--pretrained_path', type=str, default='logs/kitti_pretrain/hdc_sub.pth', help='Path to load pretrained model')
    parser.add_argument('--log_dir', type=str, default='logs/kitti_nuscenes_test', help='Directory to save logs and graphics')
    parser.add_argument('--method', type=str, choices=['frozen', 'epistemic', 'consistency', 'hybrid', 'hybrid_balanced', 'multi_view', 'oracle', 'all'], default='frozen', help='Method to test.')
    parser.add_argument('--dry_run', action='store_true', help='Run only 2 batches to quickly verify no crashes will occur.')
    parser.add_argument('--continue_pretrain', action='store_true', help='Resume pretraining from the existing pretrained_path')
    parser.add_argument('--continue', dest='continue_epochs', type=int, default=0, help='Continue feature extractor training for this many epochs, reinitialize HDC, and perform adaptation')
    parser.add_argument('--extractor_epochs', type=int, default=60, help='Number of epochs to train the feature extractor')
    parser.add_argument('--hdc_epochs', type=int, default=15, help='Number of epochs to train the HDC density model')
    parser.add_argument('--kitti_dir', type=str, default='/mnt/alpha/jmfleming/KITTI', help='Path to SemanticKITTI dataset for pretraining')
    parser.add_argument('--nusc_dir', type=str, default='/mnt/alpha/jmfleming/nuscenes_kitti', help='Path to real NuScenes dataset (in KITTI format)')
    args = parser.parse_args()

    if args.continue_epochs > 0:
        args.pretrain = True
        args.continue_pretrain = True
        args.extractor_epochs = args.continue_epochs

    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.log_dir, 'kitti_nuscenes.log'))

    try:
        ARCH = yaml.safe_load(open(CONFIG_ARCH, 'r'))
        DATA_KITTI = yaml.safe_load(open(CONFIG_LABELS_KITTI, 'r'))
        DATA_NUSC = yaml.safe_load(open(CONFIG_LABELS_NUSCENES, 'r'))
    except Exception as e:
        logger.error(f"Error loading configs: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.pretrain:
        logger.info(f"Starting Pretraining on KITTI at {args.kitti_dir}...")
        resume_dir = os.path.dirname(args.pretrained_path) if args.continue_pretrain else None
        
        model, trainer = pretrain_pipeline(
            ARCH, DATA_KITTI, data_dir=args.kitti_dir, 
            pretrained_path=args.pretrained_path, return_trainer=True, 
            skip_extractor=args.skip_extractor, resume_path=resume_dir, 
            hdc_epochs=args.hdc_epochs, extractor_epochs=args.extractor_epochs
        )
        
        if trainer is not None:
            opt_path = os.path.join(os.path.dirname(args.pretrained_path), 'feature_optimizer.pth')
            torch.save(trainer.optimizer.state_dict(), opt_path)
            logger.info(f"Successfully pretrained model on KITTI. Optimizer state saved to {opt_path}")
            
    # The user can run 'frozen' explicitly if they want the baseline anchor
    methods_to_run = ['density', 'exp_a_anchor_off'] if args.method == 'all' else [args.method]
    
    global_results = {
        'mIoU': {m: {} for m in methods_to_run},
        'Accuracy': {m: {} for m in methods_to_run},
    }
    
    # Reload model for evaluation
    model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES)

    if args.eval_kitti:
        logger.info("="*41)
        logger.info("Evaluating Source Baseline on KITTI Validation Set...")
        logger.info("="*41)
        try:
            kitti_parser = Parser(root=args.kitti_dir,
                                  train_sequences=DATA_KITTI["split"]["valid"],
                                  valid_sequences=DATA_KITTI["split"]["valid"],
                                  test_sequences=None,
                                  labels=DATA_KITTI["labels"],
                                  color_map=DATA_KITTI.get("color_map", {}),
                                  learning_map=DATA_KITTI["learning_map"],
                                  learning_map_inv=DATA_KITTI["learning_map_inv"],
                                  sensor=ARCH["dataset"]["sensor"],
                                  max_points=ARCH["dataset"]["max_points"],
                                  batch_size=1,
                                  workers=ARCH["train"]["workers"],
                                  gt=True,
                                  shuffle_train=False)
            kitti_dataloader = DataLoader(kitti_parser.validloader.dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
            kitti_metrics = evaluate_and_adapt(model, kitti_dataloader, device, eval_only=True, update_method='frozen_kitti', dry_run=args.dry_run)
            
            kitti_miou = kitti_metrics["mIoU"][-1] if len(kitti_metrics["mIoU"]) > 0 else 0
            kitti_acc = kitti_metrics["Accuracy"][-1] if len(kitti_metrics["Accuracy"]) > 0 else 0
            logger.info(f"Source Baseline (KITTI) Final Results: mIoU={kitti_miou:.4f}, Accuracy={kitti_acc:.4f}")
            global_results['KITTI_Baseline'] = {'mIoU': kitti_miou, 'Accuracy': kitti_acc}
        except Exception as e:
            logger.error(f"Failed to evaluate KITTI baseline: {e}")
            print(f"\n[!] CRASH DETECTED in KITTI Eval. Traceback:")
            traceback.print_exc()

    for current_method in methods_to_run:
        logger.info(f"=========================================")
        logger.info(f"Starting Cross-Dataset Evaluation for Method: {current_method}")
        logger.info(f"Target Domain: NuScenes ({args.nusc_dir})")
        logger.info(f"=========================================")
        
        # Ensure we always use the fresh loaded model
        model = load_hdc_model(args.pretrained_path, num_classes=NUM_CLASSES)

        # We must override the sensor params so the spherical projection doesn't cut off or squash the cars!
        # NuScenes only has 32 beams (vs KITTI's 64). If we project 32 beams into a 64-pixel high image,
        # 75% of the image will be empty space (-1). We must reduce the projection height to 32 to maintain density.
        nusc_sensor = ARCH["dataset"]["sensor"].copy()
        nusc_sensor["fov_up"] = 10.0
        nusc_sensor["fov_down"] = -30.0
        nusc_sensor["img_prop"] = nusc_sensor["img_prop"].copy()
        nusc_sensor["img_prop"]["height"] = 32
        # NuScenes only has ~1000 points per revolution. Projecting into W=2048 means 50% horizontal sparsity!
        # Shrinking the width to 1024 will pack the points tightly and restore horizontal continuity for the CNN.
        nusc_sensor["img_prop"]["width"] = 1024

        logger.info(f"Initializing NuScenes Target Dataset...")
        try:
            parser_obj = Parser(root=args.nusc_dir,
                                train_sequences=DATA_NUSC["split"]["valid"],
                                valid_sequences=DATA_NUSC["split"]["valid"],
                                test_sequences=None,
                                labels=DATA_NUSC["labels"],
                                color_map=DATA_NUSC.get("color_map", {}),
                                learning_map=DATA_NUSC["learning_map"],
                                learning_map_inv=DATA_NUSC["learning_map_inv"],
                                sensor=nusc_sensor,
                                max_points=ARCH["dataset"]["max_points"],
                                batch_size=1,
                                workers=ARCH["train"]["workers"],
                                gt=True,
                                shuffle_train=False)
            
            target_dataset = parser_obj.validloader.dataset
            target_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
        except Exception as e:
            logger.error(f"Failed to load NuScenes dataset at {args.nusc_dir}: {e}")
            continue
            
        try:
            metrics = evaluate_and_adapt(model, target_dataloader, device, eval_only=(current_method == 'frozen'), update_method=current_method, dry_run=args.dry_run)
        except Exception as e:
            logger.error(f"FATAL ERROR during NuScenes adaptation ({current_method}): {e}")
            print(f"\n[!] CRASH DETECTED in {current_method}. Traceback:")
            traceback.print_exc()
            continue
        
        if len(metrics["mIoU"]) > 0:
            initial_miou = metrics["mIoU"][0]
            final_miou = metrics["mIoU"][-1]
            initial_acc = metrics["Accuracy"][0]
            final_acc = metrics["Accuracy"][-1]
            
            global_results['mIoU'][current_method] = (initial_miou, final_miou)
            global_results['Accuracy'][current_method] = (initial_acc, final_acc)
            
            logger.info(f"Result on NuScenes: Initial mIoU={initial_miou:.4f} -> Final={final_miou:.4f}, Initial Acc={initial_acc:.4f} -> Final={final_acc:.4f}")
            suffix = f"_{current_method}"
            
            traj_json_path = os.path.join(args.log_dir, f'traj_nuscenes{suffix}.json')
            with open(traj_json_path, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            save_graphic(os.path.join(args.log_dir, f'traj_nuscenes{suffix}.png'), f'NuScenes Cross-Dataset ({current_method})', metrics)
            
            with open(os.path.join(args.log_dir, 'global_results.json'), 'w') as f:
                json.dump(global_results, f, indent=4)
        else:
            logger.info(f"No valid frames evaluated for NuScenes")

if __name__ == "__main__":
    main()
