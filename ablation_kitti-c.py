import os
import json
import logging
import argparse
import importlib
import torch
import numpy as np
from torch.utils.data import DataLoader

# Dynamically import unsup_kitti-c module (handling hyphen in filename)
ukc = importlib.import_module("unsup_kitti-c")

# Define systematic ablation suite for our unified architecture
ABLATIONS = {
    "frozen": {
        "name": "Frozen Baseline (No Adaptation)",
        "update_method": "frozen",
        "gate_mode": "epistemic",
        "ic_method": "none",
        "tau": -1.0,
        "normalize_weights": False,
        "mv_tta": "none",
        "dynamic_geom": False,
        "kappa": 15.0
    },
    "full_method": {
        "name": "Full Unified Method (Soft Dual-Weighting + BM-IC4 + Temporal Consistency)",
        "update_method": "evidential_hdc_tta",
        "gate_mode": "soft_dual_weight",
        "ic_method": "ic4",
        "tau": -1.0,
        "normalize_weights": False,
        "mv_tta": "none",
        "dynamic_geom": True,
        "kappa": 15.0
    },
    "no_dual_gating": {
        "name": "Ablation: Without Dual Gating (Epistemic Dirichlet Gating Only)",
        "update_method": "evidential_hdc_tta",
        "gate_mode": "epistemic",
        "ic_method": "ic4",
        "tau": -1.0,
        "normalize_weights": False,
        "mv_tta": "none",
        "dynamic_geom": True,
        "kappa": 15.0
    },
    "no_temporal_consistency": {
        "name": "Ablation: Without Temporal Consistency (Normalized Weights / No BM Inertia)",
        "update_method": "evidential_hdc_tta",
        "gate_mode": "soft_dual_weight",
        "ic_method": "ic4",
        "tau": -1.0,
        "normalize_weights": True,  # Disables BM prototype accumulator inertia
        "mv_tta": "none",
        "dynamic_geom": True,
        "kappa": 15.0
    },
    "no_inter_class_balance": {
        "name": "Ablation: Without Inter-Class Balance (No Tau-Prior Boundary Shift)",
        "update_method": "evidential_hdc_tta",
        "gate_mode": "soft_dual_weight",
        "ic_method": "ic4",
        "tau": 0.0,  # 0.0 removes prior frequency adjustment
        "normalize_weights": False,
        "mv_tta": "none",
        "dynamic_geom": True,
        "kappa": 15.0
    },
    "no_intra_class_balance": {
        "name": "Ablation: Without Intra-Class Balance (No IC4 Active Learning Gradient Scaling)",
        "update_method": "evidential_hdc_tta",
        "gate_mode": "soft_dual_weight",
        "ic_method": "none",  # 'none' disables IC4 active learning multiplier
        "tau": -1.0,
        "normalize_weights": False,
        "mv_tta": "none",
        "dynamic_geom": True,
        "kappa": 15.0
    },
    "no_gating": {
        "name": "Ablation: Without Uncertainty Gating (Uniform Weighting)",
        "update_method": "evidential_hdc_tta",
        "gate_mode": "uniform",
        "ic_method": "ic4",
        "tau": -1.0,
        "normalize_weights": False,
        "mv_tta": "none",
        "dynamic_geom": True,
        "kappa": 15.0
    }
}

def main():
    parser = argparse.ArgumentParser(description="Evidential HDC Ablation Study Runner (Section 7.3)")
    parser.add_argument("--ablations", type=str, default="default",
                        help="Comma-separated list of ablations to run, or 'default'/'all'. Available: " + ", ".join(ABLATIONS.keys()))
    parser.add_argument("--pretrained_path", type=str, default="logs/kitti_pretrain/hdc_sub.pth", help="Path to pretrained HDC model")
    parser.add_argument("--log_dir", type=str, default="logs/ablation_kitti_c", help="Directory to save ablation logs and plots")
    parser.add_argument("--corruptions", type=str, default="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor",
                        help="Comma-separated list of corruptions to evaluate")
    parser.add_argument("--severity", type=int, default=3, help="Severity level (1 to 5)")
    parser.add_argument("--chunked", action="store_true", help="Use chunked protocol across disjoint splits")
    parser.add_argument("--reset_per_corruption", action="store_true", help="Reset model to clean weights before each corruption")
    parser.add_argument("--continual", action="store_true", help="Continual learning mode (no resets between sequences)")
    parser.add_argument("--dry_run", action="store_true", help="Run only 2 batches per condition for quick verification")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI", help="Path to SemanticKITTI dataset")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C", help="Path to SemanticKITTI-C dataset")
    parser.add_argument("--no_diagnostics", action="store_false", dest="diagnostics", help="Disable heavy diagnostic logging")
    parser.add_argument("--dump_features", action="store_true", default=False, help="Dump offline probe features")
    
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    logger = ukc.setup_logger("AblationSuite", os.path.join(args.log_dir, "ablation_suite.log"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    active_corruptions = [c.strip() for c in args.corruptions.split(",")]
    
    if args.ablations in ["default", "all"]:
        active_ablations = ["frozen", "full_method", "no_dual_gating", "no_temporal_consistency", "no_inter_class_balance", "no_intra_class_balance"]
        if args.ablations == "all":
            active_ablations.append("no_gating")
    else:
        active_ablations = [a.strip() for a in args.ablations.split(",")]
        for a in active_ablations:
            if a not in ABLATIONS:
                raise ValueError(f"Unknown ablation '{a}'. Available: {list(ABLATIONS.keys())}")
                
    logger.info("==========================================================")
    logger.info("Starting Evidential HDC Ablation Suite (Section 7.3)")
    logger.info(f"Pretrained Model: {args.pretrained_path}")
    logger.info(f"Log Directory:    {args.log_dir}")
    logger.info(f"Corruptions:      {active_corruptions} (Severity: {args.severity})")
    logger.info(f"Active Ablations: {active_ablations}")
    logger.info("==========================================================")

    # Load initial dataset split for chunking calculation
    parser_obj = ukc.Parser(root=ukc.KITTI_DATA_DIR,
                            train_sequences=ukc.DATA["split"]["train"],
                            valid_sequences=ukc.DATA["split"]["valid"],
                            test_sequences=None,
                            labels=ukc.DATA["labels"],
                            color_map=ukc.DATA.get("color_map", {}),
                            learning_map=ukc.DATA["learning_map"],
                            learning_map_inv=ukc.DATA["learning_map_inv"],
                            sensor=ukc.ARCH["dataset"]["sensor"],
                            max_points=ukc.ARCH["dataset"]["max_points"],
                            batch_size=1,
                            workers=ukc.ARCH["train"]["workers"],
                            gt=True,
                            shuffle_train=False)
    
    target_dataset = parser_obj.validloader.dataset
    total_len = len(target_dataset)
    chunk_size = total_len // len(ukc.CORRUPTIONS)
    
    indices = list(range(total_len))
    chunks = []
    for i in range(len(ukc.CORRUPTIONS)):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < len(ukc.CORRUPTIONS) - 1 else total_len
        chunks.append(indices[start_idx:end_idx])

    # Load base model and populate source statistics
    logger.info("Loading base pretrained model and source statistics...")
    base_model = ukc.load_hdc_model(args.pretrained_path, num_classes=ukc.NUM_CLASSES, mv_tta="none")
    ukc.populate_source_statistics(base_model, args.kitti_dir, ukc.ARCH, ukc.DATA, device, dry_run=args.dry_run)
    
    source_stats_cache = {
        "class_latent_means": base_model.class_latent_means,
        "source_density_mean": getattr(base_model, "source_density_mean", None),
        "source_density_std": getattr(base_model, "source_density_std", None),
        "source_mu_cos": getattr(base_model, "source_mu_cos", None),
        "source_sigma_cos": getattr(base_model, "source_sigma_cos", None),
        "drift_mu_0": getattr(base_model, "drift_mu_0", None),
        "source_class_freq": getattr(base_model, "source_class_freq", None),
        "source_bank": getattr(base_model, "source_bank", None)
    }
    clean_state_dict = torch.load(args.pretrained_path, map_location=device)
    
    # Pre-load corruption datasets
    logger.info("Pre-loading corruption datasets...")
    corruption_datasets = {}
    sev_str = ukc.SEVERITY_MAP.get(args.severity, "moderate")
    for ctype in active_corruptions:
        corruption_root = os.path.join(args.kittic_dir, ctype, sev_str)
        seq_dir = os.path.join(corruption_root, "sequences")
        if not os.path.exists(seq_dir):
            os.makedirs(seq_dir, exist_ok=True)
            if not os.path.exists(os.path.join(seq_dir, "08")):
                os.symlink("..", os.path.join(seq_dir, "08"))
        try:
            parser_c = ukc.Parser(root=corruption_root,
                                  train_sequences=ukc.DATA["split"]["valid"],
                                  valid_sequences=ukc.DATA["split"]["valid"],
                                  test_sequences=None,
                                  labels=ukc.DATA["labels"],
                                  color_map=ukc.DATA.get("color_map", {}),
                                  learning_map=ukc.DATA["learning_map"],
                                  learning_map_inv=ukc.DATA["learning_map_inv"],
                                  sensor=ukc.ARCH["dataset"]["sensor"],
                                  max_points=ukc.ARCH["dataset"]["max_points"],
                                  batch_size=1,
                                  workers=ukc.ARCH["train"]["workers"],
                                  gt=True,
                                  shuffle_train=False)
            corruption_datasets[ctype] = parser_c.validloader.dataset
        except Exception as e:
            logger.error(f"Failed loading dataset for {ctype}: {e}")
            
    global_results = {}
    
    # Loop over active ablation tests
    for ab_key in active_ablations:
        cfg = ABLATIONS[ab_key]
        logger.info("\n" + "=" * 60)
        logger.info(f">>> ABLATION TEST: [{ab_key}] - {cfg['name']}")
        logger.info("=" * 60)
        
        # Reset model to clean state and restore source statistics
        model = ukc.load_hdc_model(args.pretrained_path, num_classes=ukc.NUM_CLASSES, mv_tta=cfg["mv_tta"])
        model.load_state_dict(clean_state_dict, strict=False)
        for k, v in source_stats_cache.items():
            if v is not None:
                setattr(model, k, v if not isinstance(v, torch.Tensor) else v.clone())
                
        # Clean up runtime tracking attributes
        attrs_to_del = ["drift_mu_c", "class_freq_ema", "class_update_counts", "class_M",
                        "running_density_std", "running_density_mean", "_contingency_table",
                        "_mv_contingency_table", "_decay_logs", "_class_n_points", "_class_n_fired",
                        "_class_true_errors_rejected", "_class_correct_rejected", "_firing_log",
                        "_veto_stats", "_update_magnitude_log", "initial_classify_weights"]
        for attr in attrs_to_del:
            if hasattr(model, attr):
                delattr(model, attr)
                
        results_miou = {c: {} for c in active_corruptions}
        results_acc = {c: {} for c in active_corruptions}
        
        for i, ctype in enumerate(active_corruptions):
            if args.reset_per_corruption and args.chunked and not args.continual:
                logger.info("Resetting model to clean pretrained weights for this corruption.")
                model.load_state_dict(clean_state_dict, strict=False)
                for attr in attrs_to_del:
                    if hasattr(model, attr):
                        delattr(model, attr)
                        
            logger.info(f"Evaluating {ctype} severity {args.severity} (Ablation: {ab_key})")
            if ctype not in corruption_datasets:
                continue
            full_dataset = corruption_datasets[ctype]
            assert len(full_dataset) == total_len, f"Length mismatch on {ctype}"
            
            if not args.chunked:
                chunk_dataset = full_dataset
                if not args.continual:
                    model.load_state_dict(clean_state_dict, strict=False)
                    for attr in attrs_to_del:
                        if hasattr(model, attr):
                            delattr(model, attr)
            else:
                chunk_dataset = torch.utils.data.Subset(full_dataset, chunks[i])
                
            dataloader = DataLoader(chunk_dataset, batch_size=1, shuffle=False, num_workers=ukc.ARCH["train"]["workers"])
            
            try:
                is_frozen = (cfg["update_method"] == "frozen")
                if not is_frozen:
                    logger.debug(f"  -> Adapting model online ({cfg['update_method']})")
                    model.train()
                else:
                    logger.debug("  -> Computing frozen baseline evaluation")
                    model.eval()

                metrics = ukc.evaluate_and_adapt(
                    model, dataloader, device,
                    eval_only=is_frozen,
                    update_method=cfg["update_method"],
                    dry_run=args.dry_run,
                    ic_method=cfg["ic_method"],
                    tau=cfg["tau"],
                    kappa=cfg["kappa"],
                    normalize_weights=cfg["normalize_weights"],
                    mv_tta=cfg["mv_tta"],
                    gate_mode=cfg["gate_mode"],
                    dynamic_geom=cfg["dynamic_geom"],
                    dump_features=args.dump_features,
                    diagnostics=args.diagnostics
                )

                if len(metrics["mIoU"]) > 0:
                    initial_miou = metrics["mIoU"][0]
                    final_miou = metrics["mIoU"][-1]
                    initial_acc = metrics["Accuracy"][0]
                    final_acc = metrics["Accuracy"][-1]
                    
                    # Store as 2-tuples (initial, final) required by save_degradation_plot
                    results_miou[ctype][args.severity] = (initial_miou, final_miou)
                    results_acc[ctype][args.severity] = (initial_acc, final_acc)
                    logger.info(f"  --> [{ab_key}] {ctype}-{args.severity} Result: Initial mIoU={initial_miou:.4f} -> Final={final_miou:.4f} | Acc={initial_acc:.4f} -> {final_acc:.4f}")
                    
                    # Save per-corruption trajectory JSON and graphic
                    suffix = f"_{ab_key}"
                    traj_json_path = os.path.join(args.log_dir, f"traj_{ctype}_{args.severity}{suffix}.json")
                    with open(traj_json_path, "w") as f:
                        json.dump(metrics, f, indent=4)
                    try:
                        ukc.save_graphic(os.path.join(args.log_dir, f"traj_{ctype}_{args.severity}{suffix}.png"), f"{ctype} Sev {args.severity} ({ab_key})", metrics)
                    except Exception as g_err:
                        logger.warning(f"Could not generate trajectory graphic for {ctype}-{args.severity}: {g_err}")
                else:
                    results_miou[ctype][args.severity] = (0.0, 0.0)
                    results_acc[ctype][args.severity] = (0.0, 0.0)

            except Exception as e:
                logger.error(f"CRITICAL ERROR evaluating {ctype}-{args.severity} under ablation {ab_key}: {e}", exc_info=True)
                results_miou[ctype][args.severity] = (0.0, 0.0)
                results_acc[ctype][args.severity] = (0.0, 0.0)

        total_evals = sum(len(sev_dict) for sev_dict in results_miou.values())
        if total_evals == 0 and not args.dry_run:
            logger.error(f"CRITICAL ERROR: No evaluation outcomes recorded for ablation '{ab_key}'. Check for silent exceptions during evaluation.")
            raise RuntimeError(f"Evaluation failed to record any metrics for ablation '{ab_key}'.")

        global_results[ab_key] = {
            "name": cfg["name"],
            "mIoU": results_miou,
            "Accuracy": results_acc
        }
        
        # Save per-ablation plots and JSON
        suffix = f"_{ab_key}"
        with open(os.path.join(args.log_dir, f"results{suffix}.json"), "w") as f:
            json.dump({"name": cfg["name"], "mIoU": results_miou, "Accuracy": results_acc}, f, indent=4)
        try:
            ukc.save_degradation_plot(os.path.join(args.log_dir, f"degradation_miou{suffix}.png"), f"KITTI-C ({ab_key})", results_miou, metric="mIoU")
            ukc.save_degradation_plot(os.path.join(args.log_dir, f"degradation_acc{suffix}.png"), f"KITTI-C ({ab_key})", results_acc, metric="Accuracy")
        except Exception as plot_err:
            logger.warning(f"Could not generate degradation plot for {ab_key}: {plot_err}")

    # Save comprehensive global ablation results
    global_json_path = os.path.join(args.log_dir, "global_ablation_results.json")
    with open(global_json_path, "w") as f:
        json.dump(global_results, f, indent=4)
    logger.info(f"\n✅ All ablation tests completed successfully! Comprehensive summary saved to: {global_json_path}")

if __name__ == "__main__":
    main()
