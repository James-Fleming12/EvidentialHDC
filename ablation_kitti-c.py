"""
ablation_kitti-c.py -- corrected ablation suite.

See docs/method_details.md Section 8 for the full list of functional changes.
The short version of what was wrong with the previous runner:

  * It ran ONE pass and read
        initial = metrics["mIoU"][0]     # cumulative mIoU after ONE frame
        final   = metrics["mIoU"][-1]    # cumulative mIoU after ~581 frames
    which are different quantities. That is why the FROZEN row reported a -7.53
    "drop" while performing zero updates, and why every ablation shared the
    identical "initial" of 41.20. Every Delta in Ablation Tables 1-3 was void.
  * It discarded Head/Mid/Tail mIoU, which evaluate_and_adapt already returns --
    so it could not show that Accuracy rose +1.85 while mIoU fell -0.24.
  * One seed, so contributions of +-0.01..0.3 had no noise floor.
  * Leave-one-out only, so component interactions were invisible.

This version: 3-pass protocol, Head/Mid/Tail recorded, add-one-in ladder,
multi-seed, gate-preset sweep, and a live-path check that warns if the corrected
gating in HDC_utils is never actually invoked.
"""

import os

# ---------------------------------------------------------------------------
# THREAD LIMITING -- must happen BEFORE torch/numpy are imported.
#
# Without this, PyTorch sets intra-op threads to the physical core count AND
# every DataLoader worker process inherits the same default, so total runnable
# threads is roughly (1 + num_workers) x cores. On a 64-core box with 12 workers
# that is ~830 threads, i.e. 13x oversubscription -- enough to drive the load
# average into the hundreds and make the machine unresponsive, which takes the
# tmux SERVER down with it. Override with ABLATION_THREADS if you know better.
# ---------------------------------------------------------------------------
_T = os.environ.get("ABLATION_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _T)

import json
import inspect
import logging
import argparse
import importlib
import random

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader

torch.set_num_threads(int(_T))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already initialised; harmless

ukc = importlib.import_module("unsup_kitti-c")
from modules import HDC_utils

# ==========================================================================
# Ablation matrix
# ==========================================================================
def _cfg(name, family, tau=-1.0, gate_mode="epistemic", ic_method="none",
         normalize_weights=False, dynamic_geom=False, mv_tta="none",
         update_method="evidential_hdc_tta", gain_control=False, kappa=15.0,
         preset="soft", gate_cfg=None, prior_mode="source", consistent_tau_weights=False, veto_tau_mismatch=False,
         lr_schedule="constant", adapt_frames=None, base_lr=0.01,
         rotation_cap=None, loosen_beta=0.0, prior_est=False, prior_switch=False, prior_ramp=False, prior_inverse=False, adaptive_budget=False, fire_th=None, boost_tail_prior=False):
    return dict(name=name, family=family, update_method=update_method,
                gate_mode=gate_mode, ic_method=ic_method, tau=tau, kappa=kappa,
                normalize_weights=normalize_weights, mv_tta=mv_tta,
                dynamic_geom=dynamic_geom, gain_control=gain_control,
                preset=preset, gate_cfg=gate_cfg or {},
                prior_mode=prior_mode, consistent_tau_weights=consistent_tau_weights,
                veto_tau_mismatch=veto_tau_mismatch,
                lr_schedule=lr_schedule, adapt_frames=adapt_frames,
                base_lr=base_lr, rotation_cap=rotation_cap, loosen_beta=loosen_beta, prior_est=prior_est, prior_switch=prior_switch, prior_ramp=prior_ramp, prior_inverse=prior_inverse, adaptive_budget=adaptive_budget, fire_th=fire_th, boost_tail_prior=boost_tail_prior)

ABLATIONS = {
    # ---- reference ----------------------------------------------------
    "frozen": _cfg("Frozen (no adaptation)", "ref", update_method="frozen",
                   gate_mode="epistemic"),
    "prior_oracle": _cfg("Prior oracle (FROZEN, pi = TRUE chunk prior)", "ref",
                         update_method="frozen", prior_mode="chunk"),
    "oracle": _cfg("Oracle gate (GT-gated CEILING, not a method)", "ref",
                   gate_mode="oracle"),

    # ---- gate preset sweep --------------------------------------------
    # The old ablation compared soft_dual_weight (u_th=0.5, coef=1.5) against
    # epistemic (u_th=0.1, coef=2.0): different REGIMES, not just offsets.
    # ---- gain control --------------------------------------------------
    "gain_full": _cfg("Full + domain-gap gain control", "gain", gain_control=True),
    "gain_epi": _cfg("Epistemic only + gain control", "gain",
                     gate_mode="epistemic", gain_control=True),

    # ---- PRIOR REMOVAL (family 'prior') ------------------------------------
    # Every adaptation arm so far carries tau=-1 on the PSEUDO-LABELS. These
    # isolate whether the prior on the labels helps or hurts the gradient.
    "adapt_tau0":      _cfg("Adapt, tau=0 pseudo-labels (prior OFF for TTA)",
                            "prior", tau=0.0),
    "adapt_tau0_d0b":  _cfg("Adapt, tau=0 + consistent gate",
                            "prior", tau=0.0, consistent_tau_weights=True),
    "adapt_tau_half":  _cfg("Adapt, tau=-0.5 (half prior on labels)",
                            "prior", tau=-0.5, consistent_tau_weights=True),

    # ---- SCHEDULE RECOVERY (family 'recover') ------------------------------
    # Reproduce the prelim ~1/t annealing on purpose, on the D0b-fixed base.
    "rec_invt":        _cfg("D0b + inv_t LR schedule", "recover",
                            consistent_tau_weights=True, lr_schedule="inv_t"),
    "rec_cosine":      _cfg("D0b + cosine LR schedule", "recover",
                            consistent_tau_weights=True, lr_schedule="cosine"),
    "rec_stop100":     _cfg("D0b + adapt 100f of 581 then freeze", "recover",
                            consistent_tau_weights=True, adapt_frames=100),
    "rec_stop250":     _cfg("D0b + adapt 250f of 581 then freeze", "recover",
                            consistent_tau_weights=True, adapt_frames=250),
    "rec_lr_hi":       _cfg("D0b + constant LR 0.05", "recover",
                            consistent_tau_weights=True, base_lr=0.05),
    "rec_lr_lo":       _cfg("D0b + constant LR 0.002", "recover",
                            consistent_tau_weights=True, base_lr=0.002),
    "g2_frozen":       _cfg("G2 signal collection (frozen)", "g2",
                            consistent_tau_weights=True, base_lr=0.0),
                            
    # ---- 6th Iteration Methods ----
    "m_a_cap":         _cfg("M-A: Per-class rotation cap 20deg", "methods",
                            consistent_tau_weights=True, rotation_cap=20.0),
    "m_ab_gain":       _cfg("M-B: Cap + Gain Control", "methods",
                            consistent_tau_weights=True, rotation_cap=20.0, gain_control=True),
    "m_abc_loosen":    _cfg("M-C: Cap + Gain + Loosen Gate", "methods",
                            consistent_tau_weights=True, rotation_cap=20.0, gain_control=True, loosen_beta=1.0),
    "m_abcd_prior":    _cfg("M-D: Cap + Gain + Loosen + Prior", "methods",
                            consistent_tau_weights=True, rotation_cap=20.0, gain_control=True, loosen_beta=1.0, prior_est=True),
    "m_d_prior_only":  _cfg("M-D: Prior Only (frozen)", "methods",
                            update_method="frozen", gate_mode="epistemic", prior_est=True),
    "m_d_prior_switch":_cfg("M-D: Prior Switch (frozen)", "methods",
                            update_method="frozen", gate_mode="epistemic", prior_est=True, prior_switch=True),
    "m_d_prior_ramp":  _cfg("M-D: Prior Ramp (frozen)", "methods",
                            update_method="frozen", gate_mode="epistemic", prior_est=True, prior_ramp=True),
    "m_d_prior_inverse":_cfg("M-D: Prior Inverse Switch (frozen)", "methods",
                            update_method="frozen", gate_mode="epistemic", prior_est=True, prior_inverse=True),
    "m_d_prior_boosted":_cfg("M-D: Prior Boosted Switch (frozen)", "methods",
                            update_method="frozen", gate_mode="epistemic", prior_est=True, prior_switch=True, boost_tail_prior=True),
    "m_a_adaptive_cap":_cfg("M-A: Adaptive Budget 20deg", "methods",
                            consistent_tau_weights=True, rotation_cap=20.0, adaptive_budget=True),
    "standard_t3a":_cfg("D-STANDARD: T3A Prototype Update", "baselines",
                            update_method="standard_t3a", prior_est=False, prior_switch=False),
    "conformalhdc":_cfg("ConformalHDC (128D latent)", "baselines",
                            update_method="conformalhdc", prior_est=False, prior_switch=False),
    "conformalhdc_10k":_cfg("ConformalHDC (10,000D HDC)", "baselines",
                            update_method="conformalhdc_10k", prior_est=False, prior_switch=False),
                            
    # ---- Legacy Validation (T1 & T-LOOSE & T-DRIFT) ----
    "legacy_frozen_t0":_cfg("Legacy: Frozen (tau=0)", "legacy", update_method="frozen", tau=0.0),
    "legacy_frozen_t1":_cfg("Legacy: Frozen (tau=-1)", "legacy", update_method="frozen", tau=-1.0),
    "legacy_loose_t0": _cfg("Legacy: Loose Adaptation (tau=0)", "legacy", 
                            update_method="evidential_hdc_tta", gate_mode="legacy_soft",
                            preset="soft", ic_method="ic4", tau=0.0, mv_tta="none", dynamic_geom=True, fire_th=0.0),
    "legacy_loose_t1": _cfg("Legacy: Loose Adaptation (tau=-1)", "legacy", 
                            update_method="evidential_hdc_tta", gate_mode="legacy_soft",
                            preset="soft", ic_method="ic4", tau=-1.0, mv_tta="none", dynamic_geom=True, fire_th=0.0),
                            
    # ---- D3 Drift-Knee Sweep ----
    "d3_cap_2":        _cfg("D3: Rotation cap 2deg", "d3_sweep",
                            consistent_tau_weights=True, rotation_cap=2.0),
    "d3_cap_5":        _cfg("D3: Rotation cap 5deg", "d3_sweep",
                            consistent_tau_weights=True, rotation_cap=5.0),
    "d3_cap_10":       _cfg("D3: Rotation cap 10deg", "d3_sweep",
                            consistent_tau_weights=True, rotation_cap=10.0),
    "d3_cap_20":       _cfg("D3: Rotation cap 20deg", "d3_sweep",
                            consistent_tau_weights=True, rotation_cap=20.0),
}

SETS = {
    "gain": ["frozen", "gain_full", "gain_epi"],
    "ceiling": ["frozen", "prior_oracle", "oracle"],
    "tau_prior": ["frozen", "adapt_tau0", "adapt_tau0_d0b", "adapt_tau_half"],
    "recover": ["frozen", "rec_invt", "rec_cosine", "rec_stop100", "rec_stop250", "rec_lr_hi", "rec_lr_lo"],
    "g2": ["g2_frozen"],
    "methods": ["frozen", "m_a_cap", "m_ab_gain", "m_abc_loosen", "m_abcd_prior"],
    "prior": ["frozen", "m_d_prior_only", "prior_oracle"],
    "d3_sweep": ["frozen", "d3_cap_2", "d3_cap_5", "d3_cap_10", "d3_cap_20"],
    "legacy_val": ["legacy_frozen_t0", "legacy_frozen_t1", "legacy_loose_t0", "legacy_loose_t1"],
    "overnight": ["frozen", "m_d_prior_only", "m_d_prior_switch", "m_d_prior_ramp", "m_d_prior_inverse", "m_a_adaptive_cap", "m_abc_loosen"],
}
SETS["all"] = list(ABLATIONS.keys())

# ==========================================================================
# Compatibility shims (tolerate either version of unsup_kitti-c.py)
# ==========================================================================
def _load_configs():
    ARCH = getattr(ukc, "ARCH", None) or yaml.safe_load(open(ukc.CONFIG_ARCH))
    DATA = getattr(ukc, "DATA", None) or yaml.safe_load(open(ukc.CONFIG_LABELS_KITTI_ALL))
    return ARCH, DATA

def _build_model(path, num_classes, mv_tta):
    sig = inspect.signature(ukc.load_hdc_model)
    kw = {"num_classes": num_classes}
    if "mv_tta" in sig.parameters:
        kw["mv_tta"] = mv_tta
    return ukc.load_hdc_model(path, **kw)

def _call_eval(model, dataloader, device, **kwargs):
    sig = inspect.signature(ukc.evaluate_and_adapt)
    ok = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return ukc.evaluate_and_adapt(model, dataloader, device, **ok)

RESET_ATTRS = [
    "drift_mu_c", "class_freq_ema", "class_update_counts", "class_M",
    "running_density_std", "running_density_mean", "_contingency_table",
    "_mv_contingency_table", "_decay_logs", "_class_n_points", "_class_n_fired",
    "_class_true_errors_rejected", "_class_correct_rejected", "_firing_log",
    "_veto_stats", "_update_magnitude_log", "initial_classify_weights",
    "_feature_dump_list", "target_prior", "class_conf", "_eval_class_conf_sum",
    "_eval_class_counts", "_d0a_mismatch_count", "_d0a_mismatch_weight_sum",
    "_d0a_all_weight_sum", "_d0a_total_points", "_d4_gains", "_d4_ths", "initial_class_conf"
]

def _reset_model(model, clean_state_dict):
    model.load_state_dict(clean_state_dict, strict=False)
    for a in RESET_ATTRS:
        if hasattr(model, a):
            try:
                delattr(model, a)
            except AttributeError:
                setattr(model, a, None)
    model.gain_controller = None
    model.gate_cfg = None

def _restore_stats(model, cache, device):
    for k, v in cache.items():
        if v is None:
            continue
        setattr(model, k, v.clone().to(device) if isinstance(v, torch.Tensor) else v)

def _tail(m, key, default=0.0):
    seq = m.get(key) or []
    return seq[-1] if len(seq) else default

def main():
    p = argparse.ArgumentParser("Evidential HDC ablation suite (corrected)")
    p.add_argument("--ablations", default="loo", help="comma list of keys, or a named set: " + ", ".join(SETS))
    p.add_argument("--pretrained_path", default="logs/kitti_pretrain/hdc_sub.pth")
    p.add_argument("--log_dir", default="logs/ablation_v2")
    p.add_argument("--corruptions", default="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor")
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--seeds", default="42")
    p.add_argument("--chunked", action="store_true")
    p.add_argument("--reset_per_corruption", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--kitti_dir", default="/mnt/alpha/jmfleming/KITTI")
    p.add_argument("--kittic_dir", default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    p.add_argument("--no_diagnostics", action="store_false", dest="diagnostics")
    p.add_argument("--fire_th", type=float, default=0.01, help="minimum fused weight to contribute. The old code used >0, which never vetoes since exp() is strictly positive.")
    p.add_argument("--gap_lo", type=float, default=0.35)
    p.add_argument("--gap_hi", type=float, default=0.75)
    p.add_argument("--calibrate_gap", action="store_true", help="measure mean epistemic uncertainty on CLEAN source and exit")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers. Overrides ARCH['train']['workers'], which is tuned for TRAINING and is far too high for batch_size=1 inference. Total CPU load ~ (1+num_workers) x ABLATION_THREADS.")
    p.add_argument("--stats_cache", default="logs/source_stats_cache.pt", help="cache source statistics here; every stage otherwise recomputes populate_source_statistics (550 frames) from scratch")
    p.add_argument("--force_stats", action="store_true", help="ignore the stats cache")
    p.add_argument("--skip_done", action="store_true", help="resume: skip (seed, ablation, corruption) triples already present in log_dir/records.json")
    p.set_defaults(diagnostics=True)
    a = p.parse_args()

    os.makedirs(a.log_dir, exist_ok=True)
    logger = ukc.setup_logger(os.path.join(a.log_dir, "ablation.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ARCH, DATA = _load_configs()
    # One mutation covers every Parser and DataLoader built below.
    _orig_workers = ARCH["train"]["workers"]
    ARCH["train"]["workers"] = a.num_workers
    logger.info(f"threads/proc={_T}  dataloader workers={a.num_workers} "
                f"(ARCH default was {_orig_workers})")
    corruptions = [c.strip() for c in a.corruptions.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    keys = SETS[a.ablations] if a.ablations in SETS else [k.strip() for k in a.ablations.split(",")]
    for k in keys:
        if k not in ABLATIONS:
            raise ValueError(f"unknown ablation '{k}'; available: {list(ABLATIONS)}")

    logger.info("=" * 78)
    logger.info(f"ablations: {keys}")
    logger.info(f"seeds: {seeds}")
    logger.info(f"corruptions: {corruptions} @ sev {a.severity}")
    logger.info(f"protocol: {'chunked' if a.chunked else 'full'} (reset_per_corruption={a.reset_per_corruption})  fire_th={a.fire_th}")
    logger.info("=" * 78)

    # chunk layout
    pobj = ukc.Parser(root=ukc.KITTI_DATA_DIR, train_sequences=DATA["split"]["train"],
                      valid_sequences=DATA["split"]["valid"], test_sequences=None,
                      labels=DATA["labels"], color_map=DATA.get("color_map", {}),
                      learning_map=DATA["learning_map"],
                      learning_map_inv=DATA["learning_map_inv"],
                      sensor=ARCH["dataset"]["sensor"],
                      max_points=ARCH["dataset"]["max_points"], batch_size=1,
                      workers=ARCH["train"]["workers"], gt=True, shuffle_train=False)
    total_len = len(pobj.validloader.dataset)
    nch = len(ukc.CORRUPTIONS)
    cs = total_len // nch
    chunks = [list(range(i * cs, (i + 1) * cs if i < nch - 1 else total_len)) for i in range(nch)]

    # Seed before source stats (using the first seed from the list)
    init_seed = seeds[0] if seeds else 42
    torch.manual_seed(init_seed); torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed); random.seed(init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # source stats (cached: each stage is a separate process and would otherwise
    # redo 550 frames of forward passes before any ablation starts)
    STAT_KEYS = ["class_latent_means", "source_density_mean", "source_density_std", "source_mu_cos", "source_sigma_cos", "drift_mu_0", "source_class_freq", "source_bank"]
    stats = None
    if a.stats_cache and os.path.exists(a.stats_cache) and not a.force_stats and not a.dry_run:
        try:
            stats = torch.load(a.stats_cache, map_location="cpu")
            logger.info(f"loaded source statistics from cache: {a.stats_cache}")
        except Exception as e:
            logger.warning(f"stats cache unreadable ({e}); recomputing")
            stats = None
    if stats is None:
        logger.info("populating source statistics ...")
        base = _build_model(a.pretrained_path, ukc.NUM_CLASSES, "none")
        ukc.populate_source_statistics(base, a.kitti_dir, ARCH, DATA, device, dry_run=a.dry_run)
        stats = {k: getattr(base, k, None) for k in STAT_KEYS}
        stats = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                 for k, v in stats.items()}
        if a.stats_cache and not a.dry_run:
            os.makedirs(os.path.dirname(a.stats_cache) or ".", exist_ok=True)
            torch.save(stats, a.stats_cache)
            logger.info(f"saved source statistics -> {a.stats_cache}")
        del base
        torch.cuda.empty_cache()
    missing = [k for k in ["source_density_mean", "source_density_std",
                           "source_mu_cos", "source_class_freq"] if stats.get(k) is None]
    if missing:
        raise RuntimeError(f"source statistics missing: {missing}. A stale cache here silently restores the uncentred-kernel bug (exp(-128)).")
    clean_sd = torch.load(a.pretrained_path, map_location=device)

    # ---- gap calibration on clean source ----
    if a.calibrate_gap:
        logger.info("calibrating gap_lo on CLEAN source validation data ...")
        m = _build_model(a.pretrained_path, ukc.NUM_CLASSES, "none")
        _reset_model(m, clean_sd); _restore_stats(m, stats, device)
        gc = HDC_utils.GainController(gap_lo=0.0, gap_hi=1.0)
        m.gain_controller = gc
        m.train()
        dl = DataLoader(pobj.validloader.dataset, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
        _call_eval(m, dl, device, eval_only=False, update_method="evidential_hdc_tta", dry_run=a.dry_run, ic_method="none", tau=-1.0, kappa=15.0, normalize_weights=True, mv_tta="none", gate_mode="uniform", dynamic_geom=False, diagnostics=False, fire_th=a.fire_th)
        logger.info(f"CLEAN SOURCE {gc.summary()}")
        if gc.gap is not None:
            logger.info(f"  ==> set --gap_lo {gc.gap:.4f}  --gap_hi {2*gc.gap:.4f}")
        else:
            logger.warning("  gain controller never invoked: evaluate_and_adapt is not routing through DualGateModel.online_update.")
        return

    # datasets
    logger.info("pre-loading corruption datasets ...")
    sev_str = ukc.SEVERITY_MAP.get(a.severity, "moderate")
    dsets = {}
    for ct in corruptions:
        root = os.path.join(a.kittic_dir, ct, sev_str)
        sq = os.path.join(root, "sequences")
        if not os.path.exists(sq):
            os.makedirs(sq, exist_ok=True)
            if not os.path.exists(os.path.join(sq, "08")):
                os.symlink("..", os.path.join(sq, "08"))
        try:
            pc = ukc.Parser(root=root, train_sequences=DATA["split"]["valid"],
                            valid_sequences=DATA["split"]["valid"], test_sequences=None,
                            labels=DATA["labels"], color_map=DATA.get("color_map", {}),
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"], batch_size=1,
                            workers=ARCH["train"]["workers"], gt=True, shuffle_train=False)
            dsets[ct] = pc.validloader.dataset
        except Exception as e:
            logger.error(f"failed to load {ct}: {e}")

    records, frozen_cache = [], {}
    warned_path = False

    # ---- resume ----
    done = set()
    rec_path = os.path.join(a.log_dir, "records.json")
    if a.skip_done and os.path.exists(rec_path):
        try:
            records = json.load(open(rec_path))
            current_proto = "chunked" if a.chunked else "full"
            for r in records:
                if r.get("protocol") and r.get("protocol") != current_proto:
                    raise ValueError(f"Mixed protocols! Log contains {r.get('protocol')} but current is {current_proto}")
            done = {(r["seed"], r["ablation"], r["corruption"]) for r in records}
            logger.info(f"resuming: {len(records)} records already present, {len(done)} (seed, ablation, corruption) triples will be skipped")
        except Exception as e:
            logger.warning(f"could not read {rec_path} for resume ({e}); starting fresh")
            records, done = [], set()

    for seed in seeds:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        np.random.seed(seed); random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for key in keys:
            cfg = ABLATIONS[key]
            logger.info("\n" + "=" * 78)
            logger.info(f">>> seed={seed}  {key}  ({cfg['name']})")
            logger.info(f"    tau={cfg['tau']} gate={cfg['gate_mode']} preset={cfg['preset']} "
                        f"ic={cfg['ic_method']} norm_w={cfg['normalize_weights']} "
                        f"dyn={cfg['dynamic_geom']} mv={cfg['mv_tta']} gain={cfg['gain_control']}")
            logger.info("=" * 78)

            model = _build_model(a.pretrained_path, ukc.NUM_CLASSES, cfg["mv_tta"])
            _reset_model(model, clean_sd); _restore_stats(model, stats, device)

            for i, ct in enumerate(corruptions):
                if ct not in dsets:
                    continue
                if (seed, key, ct) in done:
                    logger.info(f"  skip (already done): seed={seed} {key} {ct}")
                    continue
                if a.reset_per_corruption:
                    _reset_model(model, clean_sd); _restore_stats(model, stats, device)

                gate_cfg = dict(HDC_utils.PRESETS[cfg["preset"]])
                gate_cfg.update(cfg["gate_cfg"])
                model.gate_cfg = gate_cfg
                model.gain_controller = (
                    HDC_utils.GainController(gap_lo=a.gap_lo, gap_hi=a.gap_hi)
                    if cfg["gain_control"] else None)

                ds = dsets[ct]
                chunk_idx = ukc.CORRUPTIONS.index(ct)
                chunk = torch.utils.data.Subset(ds, chunks[chunk_idx]) if a.chunked else ds
                dl = DataLoader(chunk, batch_size=1, shuffle=False, num_workers=ARCH["train"]["workers"])
                dl.corruption = ct

                common = dict(dry_run=a.dry_run, ic_method=cfg["ic_method"], tau=cfg["tau"],
                              kappa=cfg["kappa"], normalize_weights=cfg["normalize_weights"],
                              mv_tta=cfg["mv_tta"], gate_mode=cfg["gate_mode"],
                              dynamic_geom=cfg["dynamic_geom"], diagnostics=a.diagnostics,
                              fire_th=cfg.get("fire_th") if cfg.get("fire_th") is not None else a.fire_th,
                              consistent_tau_weights=cfg["consistent_tau_weights"],
                              veto_tau_mismatch=cfg["veto_tau_mismatch"],
                              lr_schedule=cfg["lr_schedule"], base_lr=cfg["base_lr"],
                              rotation_cap=cfg.get("rotation_cap", None),
                              loosen_beta=cfg.get("loosen_beta", 0.0),
                              prior_est=cfg.get("prior_est", False),
                              prior_switch=cfg.get("prior_switch", False),
                              prior_ramp=cfg.get("prior_ramp", False),
                              prior_inverse=cfg.get("prior_inverse", False),
                              adaptive_budget=cfg.get("adaptive_budget", False),
                              boost_tail_prior=cfg.get("boost_tail_prior", False),
                              update_method=cfg["update_method"])

                # --- PRIOR ORACLE: replace pi_source with the chunk's TRUE prior ---
                # The frozen confusion matrix's ROW SUMS are the GT class counts,
                # so this costs no extra data pass.
                if cfg.get("prior_mode") == "chunk":
                    base_fk = (seed, ct, cfg["tau"], cfg["kappa"], cfg["mv_tta"], "source", False, False, False, False)
                    if base_fk not in frozen_cache:
                        raise RuntimeError(
                            "prior_oracle needs the plain 'frozen' ablation to run first "
                            "so the chunk GT prior can be read from its confusion matrix. "
                            "Put 'frozen' before 'prior_oracle' in the ablation list.")
                    cm = np.array(frozen_cache[base_fk]["ConfusionMatrix"], dtype=np.float64)
                    gt_counts = cm.sum(axis=1)
                    pi_chunk = gt_counts / max(1.0, gt_counts.sum())
                    model.source_class_freq = torch.clamp(
                        torch.tensor(pi_chunk, dtype=torch.float32, device=device), min=1e-5)
                    pi_src = stats["source_class_freq"].to(device).float()
                    l1 = float((model.source_class_freq - pi_src).abs().sum())
                    logger.info(f"    [prior_oracle] L1(pi_chunk, pi_source) = {l1:.4f}  "
                                f"(0 => no prior drift => nothing for online estimation to recover)")
                else:
                    model.source_class_freq = stats["source_class_freq"].to(device).float()

                try:
                    # Pass 1: frozen -> TRUE initial. Cached across configs that
                    # share (seed, corruption, tau, kappa, mv_tta); gate_mode and
                    # ic_method do not affect frozen evaluation.
                    fk = (seed, ct, cfg["tau"], cfg["kappa"], cfg["mv_tta"],
                          cfg.get("prior_mode", "source"),
                          cfg.get("prior_est", False),
                          cfg.get("prior_switch", False),
                          cfg.get("prior_ramp", False),
                          cfg.get("prior_inverse", False),
                          cfg.get("boost_tail_prior", False),
                          cfg["update_method"])
                    if fk in frozen_cache:
                        init_m, init_conf = frozen_cache[fk]
                        if init_conf is not None:
                            model.initial_class_conf = init_conf.clone().to(device)
                    else:
                        model.eval()
                        init_m = _call_eval(model, dl, device, eval_only=True, **common)
                        frozen_cache[fk] = (init_m, getattr(model, 'initial_class_conf', None))
                        
                        # Fix Intra-Ablation State Leak: eval_only=True mutates target_prior. 
                        # We MUST reset before adaptation begins!
                        _reset_model(model, clean_sd)
                        _restore_stats(model, stats, device)

                    if cfg["update_method"] == "frozen":
                        adapt_m = final_m = init_m
                    else:
                        HDC_utils.reset_counters()
                        model.train()
                        adapt_m = _call_eval(model, dl, device, eval_only=False, **common)
                        c = HDC_utils.counters()
                        if not warned_path and c["fuse"] == 0 and c["update"] == 0:
                            warned_path = True
                            logger.warning(
                                "  !! HDC_utils gating was NEVER invoked during adaptation.\n"
                                "     evaluate_and_adapt is using its own inline gating, so the "
                                "corrected fusion, the fire_th veto and gain control are ALL "
                                "INACTIVE. Route unsup_kitti-c.py's gate block through "
                                "HDC_utils.fuse_uncertainties before trusting these rows.")
                        model.eval()
                        final_m = _call_eval(model, dl, device, eval_only=True, **common)

                    rec = dict(
                        seed=seed, ablation=key, name=cfg["name"], family=cfg["family"],
                        corruption=ct, severity=a.severity,
                        protocol="chunked" if a.chunked else "full", n_frames=len(dl),
                        gate_mode=cfg["gate_mode"], preset=cfg["preset"], tau=cfg["tau"],
                        ic_method=cfg["ic_method"], normalize_weights=cfg["normalize_weights"],
                        gain_control=cfg["gain_control"], mv_tta=cfg["mv_tta"],
                        rotation_cap=cfg["rotation_cap"], loosen_beta=cfg["loosen_beta"], prior_est=cfg["prior_est"],
                        fire_th=a.fire_th,
                        init_miou=_tail(init_m, "mIoU"), final_miou=_tail(final_m, "mIoU"),
                        online_miou=_tail(adapt_m, "mIoU"),
                        init_head=_tail(init_m, "Head_mIoU"), final_head=_tail(final_m, "Head_mIoU"),
                        init_mid=_tail(init_m, "Mid_mIoU"), final_mid=_tail(final_m, "Mid_mIoU"),
                        init_tail=_tail(init_m, "Tail_mIoU"), final_tail=_tail(final_m, "Tail_mIoU"),
                        init_acc=_tail(init_m, "Accuracy"), final_acc=_tail(final_m, "Accuracy"),
                        firing_rate=adapt_m.get("FiringRate", 0.0),
                        update_mag=adapt_m.get("UpdateMagnitude", 0.0),
                        hdc_fuse_calls=HDC_utils.counters()["fuse"],
                        hdc_update_calls=HDC_utils.counters()["update"],
                    )
                    if cfg["gain_control"] and model.gain_controller is not None:
                        rec["gain_summary"] = model.gain_controller.summary()
                        logger.info(f"    {model.gain_controller.summary()}")
                    records.append(rec)

                    logger.info(
                        f"  {ct}-{a.severity} [{rec['protocol']}, n={rec['n_frames']}] "
                        f"mIoU {rec['init_miou']:.4f} -> {rec['final_miou']:.4f} "
                        f"(online {rec['online_miou']:.4f}) | "
                        f"H {rec['init_head']:.4f}->{rec['final_head']:.4f} "
                        f"M {rec['init_mid']:.4f}->{rec['final_mid']:.4f} "
                        f"T {rec['init_tail']:.4f}->{rec['final_tail']:.4f} | "
                        f"Acc {rec['init_acc']:.4f}->{rec['final_acc']:.4f} | "
                        f"fire {rec['firing_rate']*100:.2f}% mag {rec['update_mag']:.5f}")

                except Exception as e:
                    logger.error(f"FAILED {key}/{ct}: {e}", exc_info=True)

                del dl
                torch.cuda.empty_cache()

                with open(rec_path, "w") as f:
                    json.dump(records, f, indent=2)

    if not records:
        raise RuntimeError("no records produced -- check the log for exceptions.")
    out = os.path.join(a.log_dir, "records.json")
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"\nwrote {len(records)} records -> {out}")
    logger.info(f"now run:  python analyze_ablations.py --records {out} --pct")

if __name__ == "__main__":
    main()