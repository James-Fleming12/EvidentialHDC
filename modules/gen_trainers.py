import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np

from modules.trainer import Trainer
from common.avgmeter import AverageMeter
from modules.DGLSS import dglss_sifc_loss, dglss_scc_loss, get_dglss_view

# Phase 24.2 casualty list: the classes whose corrupted features collapse and absorb
# into neighbors (Road, Building, Other-ground, Traffic-sign, Bicycle). Phase 25 probe:
# up-weight these anchors in the SupCon loss so the contrastive signal concentrates on
# the classes that need corrupted-view separability the most.
FRAGILE_CLASSES = {2, 7, 13, 14, 15}
FRAGILE_SUPCON_W = 3.0
# Phase 25.7: same-class (clean, extreme-aug) pairs repel above this cosine margin
HARDNEG_MARGIN = 0.5
# DGLSS / DGLSS++ consistency weights (lambda_1 SIFC, lambda_2 SCC in the papers)
DGLSS_TAU = 0.7
# DGLSS / DGLSS++ / standard-implementation arms. These are VIB-FREE by construction:
# they route through their own branch (plain bottleneck, no reparameterization, no KL)
# so the comparison with the paper implementations is not contaminated by VIB.
DGLSS_METHODS = {'supcon_vib_dglss', 'supcon_vib_dglsspp', 'supcon_vib_dglss_enc',
                 'supcon_vib_dglsspp_cor', 'supcon_vib_dglsspp_supcon',
                 'supcon_vib_dglsspp_bal', 'supcon_vib_dglsspp_vib',
                 'supcon_vib_dglsspp_corsupcon',
                 'supcon_vib_dglsspp_corsupcon_nogmsifc',
                 'supcon_vib_dglsspp_corsupcon_nolscc',
                 'supcon_vib_dglsspp_corsupcon_nocons',
                 'supcon_vib_dglsspp_corsupcon_w03',
                 'supcon_vib_dglsspp_corsupcon_w05',
                 'supcon_vib_dglsspp_corsupcon_blend03',
                 'supcon_vib_dglsspp_corsupcon_blend05',
                 'supcon_vib_dglsspp_corsupcon_cond',
                 'supcon_vib_dglsspp_corsupcon_ch64',
                 'supcon_vib_dglsspp_corsupcon_ch96',
                 'supcon_vib_dglsspp_corsupcon_coclust',
                 'supcon_vib_dglsspp_corsupcon_coclust_w005',
                 'supcon_vib_dglsspp_corsupcon_nnpull',
                 'supcon_vib_dglsspp_corsupcon_nocons_nnpull'}

# SupCon anchoring-direction variants (tested at micro): each changes ONE knob of the
# clean-anchoring, with two points per sweep so the direction is testable against
# mis-tuning. key -> kwargs to supcon_loss / SupCon weight multiplier.
SUPCON_VARIANTS = {
    'supcon_vib_dglsspp_corsupcon': {},
    'supcon_vib_dglsspp_corsupcon_w03': {'weight': 0.03},
    'supcon_vib_dglsspp_corsupcon_w05': {'weight': 0.05},
    'supcon_vib_dglsspp_corsupcon_blend03': {'weight': 0.1, 'blend_alpha': 0.3},
    'supcon_vib_dglsspp_corsupcon_blend05': {'weight': 0.1, 'blend_alpha': 0.5},
    'supcon_vib_dglsspp_corsupcon_cond': {'weight': 0.1, 'cond': True},
    'supcon_vib_dglsspp_corsupcon_ch64': {'weight': 0.1, 'channels': 64},
    'supcon_vib_dglsspp_corsupcon_ch96': {'weight': 0.1, 'channels': 96},
    # corrupted-only clustering: the clean-anchor SupCon PLUS a pull of the corrupted
    # points toward their CORRUPTED class centroids (alpha=1.0), which maximizes
    # intra-corrupted packing while leaving the shifted direction intact (the two
    # drivers of the label ceiling from Iteration 12). Two weights for robustness.
    'supcon_vib_dglsspp_corsupcon_coclust': {'weight': 0.1, 'coclust_w': 0.1},
    'supcon_vib_dglsspp_corsupcon_coclust_w005': {'weight': 0.1, 'coclust_w': 0.05},
    # neighborhood-purity regularizer (Iteration 13.2): clean-anchor SupCon PLUS a
    # pull of each corrupted point toward its nearest SAME-CLASS neighbor (raises the
    # 1-NN purity that drives both the ceiling and the AL readiness). Tested on the
    # full method and on the simpler nocons base (which the muddle check found
    # AL-cleanest).
    'supcon_vib_dglsspp_corsupcon_nnpull': {'weight': 0.1, 'nnpull_w': 0.1},
    'supcon_vib_dglsspp_corsupcon_nocons_nnpull': {'weight': 0.1, 'nnpull_w': 0.1},
    # AL-oriented objectives (Iteration-11 line): train the feature geometry that the
    # AL/TTA bottleneck measures. Each adds ONE loss to the robust corsupcon base:
    #   _ball       : intra-class ball tightening (EMA class centers, cosine) --
    #                 shrinks the fat-blob radius (intra-cos 0.62-0.70) that drives
    #                 the mean-estimation sample complexity, the R1-prototype
    #                 viability, and the T-error -> W-error amplification.
    #   _spec       : covariance condition-number penalty -- flattens the spectrum
    #                 that the inverse covariance amplifies (the 4-6x ridge-relevant
    #                 error, the fractional update needing beta < 1).
    #   _ball_spec  : both, at half weights (the two levers together).
    'supcon_vib_dglsspp_corsupcon_ball': {'weight': 0.1, 'ball_w': 0.1},
    'supcon_vib_dglsspp_corsupcon_spec': {'weight': 0.1, 'spec_w': 0.1},
    'supcon_vib_dglsspp_corsupcon_ball_spec': {'weight': 0.1, 'ball_w': 0.05,
                                               'spec_w': 0.05},
}

# Decoupling variants (Iteration-15 shortlist): split the bottleneck into an invariant
# head (conv_2, carries GMSIFC+LSCC+SupCon = the TTA/assignment machinery) and a
# corruption head (conv_corr, CE + LSCC only, NO clean anchor = keeps the shifted,
# recoverable class structure the labeled ceiling needs). The decoder/HDC read the
# concatenation [inv, corr], so the oracle has access to the retained direction.
#   _twobranch  : independent corr head (mode='ind'), total dim = inv_dim + corr_dim
#   _residual   : corr = inv + delta (mode='res'), weakly L2-regularized
#   _corrfree   : DROP LSCC on the corr slice (the Iteration-16 finding: LSCC is a
#                 clean-view alignment term that re-anchors corr; this variant leaves
#                 CE on the full concat as corr's only clean pull -> genuinely free)
#   _dircons    : residual + displacement-direction consistency L_dir = 1 - cos(dz,
#                 sg(delta_c)) with a per-class EMA displacement direction (idea #3:
#                 same-class corrupted points move coherently; direction, not magnitude)
#   _dircons_w02 : dircons at dir_w=0.2 (Iteration-19 follow-up: reach the classes
#                 that did not shift at 0.1, e.g. car, to push the crosstalk ceiling
#                 past DGLSS++'s 0.214)  [micro-gated Iteration 19.5: car corr_dir
#                 stays 0.95 -> the pull strength is not the lever]
#   _dircons_frag: dircons applied ONLY to the casualty classes (2/7/13/14/15), so
#                 the healthy classes' residual stays uncoupled (Iteration-19 finding:
#                 the all-class coupling costs the healthy conditions ~0.01-0.02)
#   _dircons_w02_res01: dir_w 0.2 AND L_res 0.05 -> 0.01 (Iteration-19.5 lever a:
#                 relax the residual penalty so car's displacement can actually
#                 develop instead of being shrunk toward zero)
#   _dircons_frag_w02: fragile-only dircons at dir_w 0.2 (Iteration-19.5 levers a+b
#                 combined: stronger direction, concentrated on the casualty classes)
#   _distill     : teacher-preserved ceiling branch (feedback direction): the corr
#                 branch (independent, no residual coupling) is distilled toward the
#                 FROZEN plain-DGLSS++ medium's corrupted-view features via cosine
#                 geometry, instead of a self-referential EMA displacement direction.
#                 TTA stays on the inv branch (GMSIFC+LSCC+SupCon, untouched); the
#                 HDC decoder reads the concat. teacher_w scales L_distill.
# Each is micro-gated on the feature-space mechanism (corr dir_retention < 1, inv
# feat_cos high, concatenated oracle up) before any medium commitment.
# method key -> (inv_dim, corr_dim, corr_mode, res_w, lscc_corr, dir_w, dir_fragile, teacher_w)
DECOUPLE_VARIANTS = {
    'supcon_vib_dglsspp_corsupcon_twobranch_128_64':        (128, 64, 'ind', 0.0, True, 0.0, False, 0.0),
    'supcon_vib_dglsspp_corsupcon_twobranch_128_128':       (128, 128, 'ind', 0.0, True, 0.0, False, 0.0),
    'supcon_vib_dglsspp_corsupcon_residual_128_128':        (128, 128, 'res', 0.05, True, 0.0, False, 0.0),
    'supcon_vib_dglsspp_corsupcon_twobranch_128_64_corrfree': (128, 64, 'ind', 0.0, False, 0.0, False, 0.0),
    'supcon_vib_dglsspp_corsupcon_residual_128_128_dircons':  (128, 128, 'res', 0.05, True, 0.1, False, 0.0),
    'supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_w02': (128, 128, 'res', 0.05, True, 0.2, False, 0.0),
    'supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_frag': (128, 128, 'res', 0.05, True, 0.1, True, 0.0),
    'supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_w02_res01': (128, 128, 'res', 0.01, True, 0.2, False, 0.0),
    'supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_frag_w02': (128, 128, 'res', 0.05, True, 0.2, True, 0.0),
    'supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_res01': (128, 128, 'res', 0.01, True, 0.1, False, 0.0),
    'supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_res02': (128, 128, 'res', 0.02, True, 0.1, False, 0.0),
    'supcon_vib_dglsspp_corsupcon_distill_128_128':  (128, 128, 'ind', 0.0, False, 0.0, False, 0.1),
    'supcon_vib_dglsspp_corsupcon_corrfree_corrsc':  (128, 64, 'ind', 0.0, False, 0.0, False, 0.0),
}
for _m in DECOUPLE_VARIANTS:
    DGLSS_METHODS.add(_m)

# Frozen teacher for the _distill variants: the plain DGLSS++ medium checkpoint
# (the higher-ceiling extractor the teacher framework preserves geometry from).
TEACHER_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp'
TEACHER_METHOD = 'supcon_vib_dglsspp'

# Feedback-direction variants (Iteration-19.5 broad sweep). Each adds ONE new loss to
# the robust corsupcon base, attacking the ceiling WITHOUT making the whole
# representation more shifted (the dircons failure mode):
#   corrsc       : corruption-manifold multi-positive SupCon. A SECOND independently
#                  corrupted view is generated; supcon(z_aug2, z_aug) pulls same-class
#                  corrupted realizations together (instead of corr->clean), so each
#                  class forms a good CORRUPTED manifold. Weak clean-corrupted term
#                  retained. ("same class => same local corrupted manifold")
#   corrfree_corrsc: corrfree base (independent corr head, no LSCC on corr) + the
#                  corrupted-manifold SupCon applied to the CORR slice, giving the
#                  free branch the structured supervision it was missing (corrfree
#                  alone had freedom but no organization).
#   hdc          : HDC-aware soft-prototype loss. Differentiable surrogate on the
#                  binarized geometry: pull the corrupted view's soft HDC code toward
#                  the class's clean HDC prototype (margin in code space, not in the
#                  continuous space the dircons geometry lived in).
#   antianchor   : Iteration-19.8 diagnosis -- every objective we built ERASES the
#                  corruption shift (GMSIFC/LSCC/SupCon/dircons/corrsc/hdc all pull
#                  corrupted toward clean or toward a coherence). This variant is the
#                  inverse: it PENALIZES the corrupted->clean class cosine, so the
#                  network retains whatever shift develops naturally instead of
#                  un-learning it. The direct test of "plain DGLSS++ keeps its ceiling
#                  because it was never told to undo corruption".
CORRSC_VARIANTS = {
    'supcon_vib_dglsspp_corsupcon_corrsc': 0.1,
    'supcon_vib_dglsspp_corsupcon_corrfree_corrsc': 0.1,
}
# The corrsc direction is "corrupted-manifold compactness, weak clean anchor". For the
# single-branch corrsc the standard clean-anchor SupCon would otherwise run at 0.1 on
# the full z8 (dominating the 0.1 corr-manifold term) -- cap it at a weak 0.02 so the
# corrupted-manifold is the primary pull. (corrfree_corrsc is untouched: its standard
# anchor applies to the INV slice for TTA, which is the intended division.)
SUPCON_VARIANTS['supcon_vib_dglsspp_corsupcon_corrsc'] = {'weight': 0.02}
HDC_VARIANTS = {
    'supcon_vib_dglsspp_corsupcon_hdc': 0.1,
}
# Anti-anchor: penalize the corrupted->clean class cosine (positive term added to the
# loss), so the network is DISCOURAGED from erasing the corruption shift. The one
# objective none of the failed variants tried (Iteration-19.8 diagnosis).
ANTI_ANCHOR_VARIANTS = {
    'supcon_vib_dglsspp_antianchor': 0.1,
}
for _m in (*CORRSC_VARIANTS, *HDC_VARIANTS, *ANTI_ANCHOR_VARIANTS):
    DGLSS_METHODS.add(_m)
for _m in SUPCON_VARIANTS:
    DGLSS_METHODS.add(_m)

# InstanceNorm variants (Iteration-19.8 candidate): BatchNorm's running stats are a
# known covariate-shift failure point, and BN-statistic alignment was the best TTA
# lever (it FIXED the stats at test time). Training with InstanceNorm removes that
# sensitivity at the source. Two bases: plain DGLSS++ (beam-drop) and the robust
# corruption-view base, so the norm effect is isolated from the view effect.
NORM_VARIANTS = {
    'supcon_vib_dglsspp_instancenorm': {'norm': 'in'},
    'supcon_vib_dglsspp_cor_instancenorm': {'norm': 'in'},
}
for _m in NORM_VARIANTS:
    DGLSS_METHODS.add(_m)

# Input-normalization variants (Iteration-19.10 level-1 covariate shift): the 5-channel
# input is normalized by FIXED clean-data img_means/img_stds in the parser, so under
# fog/crosstalk the network receives inputs scaled against clean statistics. These
# variants normalize each scan's valid input channels by its OWN per-scan mean/std
# instead. _inputin = input-IN only (internal stays BatchNorm); _inputin_in = the
# stack (input-IN + internal InstanceNorm), attacking both covariate-shift levels.
# _inputin_in_chan = the Iteration-19.11.2 fix: per-scan normalization restricted to
# the RANGE and REMISSION channels (indices 0, 4) that carry crosstalk's statistics
# shift, LEAVING the xyz geometry channels (1-3) untouched so fog's shifted direction
# survives. (Fog's recoverable classes are shift-driven; crosstalk's are
# packing-driven -- this is the condition-aware split.)
INPUT_NORM_VARIANTS = {
    'supcon_vib_dglsspp_inputin': {'norm': 'bn'},
    'supcon_vib_dglsspp_inputin_in': {'norm': 'in'},
    'supcon_vib_dglsspp_inputin_in_chan': {'norm': 'in', 'norm_channels': (0, 4)},
    'supcon_vib_dglsspp_inputin_in_scale': {'norm': 'in', 'scale_only': True},
    # Iteration C8 levers (the continuous-features healthy-loss fix). C8 proved the
    # healthy-ceiling loss survives every decoding, so it is a continuous loss in the
    # cov-shift extractor's features. These three attack it training-side:
    #  - _scope: InstanceNorm only in the late stages (layer3/4 + bottleneck conv_1/2);
    #    the early geometry blocks keep BatchNorm so the healthy conditions' early-stage
    #    per-dimension anisotropy survives while fog/crosstalk robustness stays.
    #  - _scalein: scale-only internal InstanceNorm (divide by per-scan per-channel std
    #    without centering), preserving the per-dimension offset structure.
    #  - _scalereg: feature-scale regularizer in the trainer (clean view scale pulled
    #    toward the beam-drop view's), preventing the packing erosion.
    'supcon_vib_dglsspp_inputin_in_chan_scope': {'norm': 'in', 'norm_channels': (0, 4), 'norm_scope': 'in_late'},
    'supcon_vib_dglsspp_inputin_in_chan_scalein': {'norm': 'in', 'norm_channels': (0, 4), 'scale_in': True},
    'supcon_vib_dglsspp_inputin_in_chan_scalereg': {'norm': 'in', 'norm_channels': (0, 4), 'scale_reg': True},
}
for _m in INPUT_NORM_VARIANTS:
    DGLSS_METHODS.add(_m)

class GenTrainer(Trainer):
    def __init__(self, ARCH, DATA, datadir, logdir, path=None, method='baseline', cutoff_percent=1.0,
                 fragile_w=None, edl_kl_cap=0.005, edl_w=0.1, edl_kl_selective=True,
                 dglss_lam1=1.0, dglss_lam2=1.0, dglss_scc_norm=True):
        self.method = method
        self.cutoff_percent = cutoff_percent
        self.fragile_w = fragile_w if fragile_w is not None else FRAGILE_SUPCON_W
        self.edl_kl_cap = edl_kl_cap
        self.edl_w = edl_w
        self.edl_kl_selective = edl_kl_selective
        # DGLSS / DGLSS++ consistency weights (lambda_1 SIFC, lambda_2 SCC in the papers)
        self.dglss_lam1 = dglss_lam1
        self.dglss_lam2 = dglss_lam2
        # SCC / LSCC prototype normalization. True = the stable cosine-correlation form
        # (Gram entries in [-1, 1]); False reproduces the raw unnormalized-prototype
        # form whose Gram entries scale as ||z||^2 and diverge during VIB-free training.
        self.dglss_scc_norm = dglss_scc_norm

        # Decoupling variants: tell the model constructor to build the corr head and
        # the trainer to route the losses per-branch. Must be set BEFORE super().__init__
        # builds the network.
        dec = DECOUPLE_VARIANTS.get(self.method)
        if dec is not None:
            (self.inv_dim, self.corr_dim, self.corr_mode,
             self.res_w, self.lscc_corr, self.dir_w, self.dir_fragile,
             self.teacher_w) = dec
            ARCH.setdefault("train", {})["twobranch"] = {
                "inv_dim": self.inv_dim, "corr_dim": self.corr_dim, "corr_mode": self.corr_mode}
        else:
            self.inv_dim, self.corr_dim, self.corr_mode = 128, 0, 'ind'
            self.res_w, self.lscc_corr, self.dir_w = 0.0, True, 0.0
            self.dir_fragile = False
            self.teacher_w = 0.0

        # InstanceNorm variants: set the norm type in the twobranch config so the model
        # constructor builds with InstanceNorm instead of BatchNorm.
        if self.method in NORM_VARIANTS:
            tw = ARCH.setdefault("train", {}).setdefault("twobranch", {})
            tw["norm"] = NORM_VARIANTS[self.method]["norm"]
        if self.method in INPUT_NORM_VARIANTS:
            # input-IN variants: internal norm from the config, and the input-IN flag
            # (applied inside the model forward so it holds at train AND eval time).
            tw = ARCH.setdefault("train", {}).setdefault("twobranch", {})
            tw["norm"] = INPUT_NORM_VARIANTS[self.method]["norm"]
            tw["input_in"] = True
            tw["norm_channels"] = INPUT_NORM_VARIANTS[self.method].get("norm_channels")
            tw["scale_only"] = INPUT_NORM_VARIANTS[self.method].get("scale_only", False)
            # C8 levers
            tw["norm_scope"] = INPUT_NORM_VARIANTS[self.method].get("norm_scope", "all")
            tw["scale_in"] = INPUT_NORM_VARIANTS[self.method].get("scale_in", False)
            self.scale_reg = INPUT_NORM_VARIANTS[self.method].get("scale_reg", False)
        else:
            self.scale_reg = False
        self.input_in = self.method in INPUT_NORM_VARIANTS
        self._dir_ema = None  # per-class EMA displacement direction for _dircons

        # HDC-aware variants: build the seeded projection ONCE (get_hdc_projection
        # calls torch.manual_seed(42), which must NOT run inside the training loop --
        # it would reset the global RNG every step and destroy augmentation/subsample
        # randomness). Save/restore the RNG around the build to leave training state
        # untouched, and store the projection for reuse.
        self._hdc_proj = None
        if self.method in HDC_VARIANTS:
            from modules.oracle_core import get_hdc_projection
            rng_state = torch.get_rng_state()
            self._hdc_proj = get_hdc_projection(dim_in=128, dim_out=10000,
                                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            torch.set_rng_state(rng_state)

        # Frozen teacher for the _distill variants: load the plain DGLSS++ medium
        # extractor once, frozen, so its corrupted-view geometry is the distillation
        # target. Builds a second model via a fresh GenTrainer (cheap: eval-only load).
        # The teacher must be the plain 128D DGLSS++ -- temporarily clear the student's
        # twobranch config so the teacher model constructor builds a single-head net.
        self.teacher_model = None
        if self.teacher_w > 0:
            saved_tw = ARCH["train"].pop("twobranch", None)
            t = GenTrainer(ARCH, DATA, datadir, TEACHER_PATH, path=TEACHER_PATH,
                           method=TEACHER_METHOD)
            if saved_tw is not None:
                ARCH["train"]["twobranch"] = saved_tw
            self.teacher_model = t.model
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)
        
        # Call super with path=None to prevent it from immediately loading the checkpoint
        super().__init__(ARCH, DATA, datadir, logdir, None)
        
        # If VIB or any SupCon+VIB variant, initialize logvar_head and add to optimizer BEFORE loading checkpoint
        if self.method == 'vib' or self.method == 'supcon_vib_dglsspp_vib' or (self.method.startswith('supcon_vib') and self.method not in DGLSS_METHODS):
            self.logvar_head = nn.Conv2d(128, 128, kernel_size=1).to(self.device)
            # Add to optimizer so it has 2 param groups (matching the saved checkpoint)
            self.optimizer.add_param_group({'params': self.logvar_head.parameters()})
        else:
            self.logvar_head = None

        # Phase 25 Addition 2 (evidential head): a 1x1 conv on the 128D bottleneck outputting
        # per-pixel Dirichlet evidence. Trained so the augmented (corruption-hard) views carry
        # high epistemic uncertainty, giving the model intrinsic calibrated uncertainty for
        # pseudo-label gating. Saved via the optimizer state (like logvar_head).
        if self.method == 'supcon_vib_evidential':
            self.evidence_head = nn.Conv2d(128, self.parser.get_n_classes(), kernel_size=1).to(self.device)
            self.optimizer.add_param_group({'params': self.evidence_head.parameters()})
            self._edl_accum = {}
        else:
            self.evidence_head = None
            self._edl_accum = None

        # Phase 25.6 (direct loss prediction, Yoo & Kweon): a head that regresses the main
        # classifier's per-point loss on clean + augmented views. The per-point CE of the
        # semantic head is the supervision (no OOD labels); the predicted loss is the
        # gating/uncertainty signal. Condition-agnostic and EDL-trap-free.
        if self.method in ('supcon_vib_losspred', 'supcon_vib_hardneg'):
            self.losspred_head = nn.Conv2d(128, 1, kernel_size=1).to(self.device)
            self.optimizer.add_param_group({'params': self.losspred_head.parameters()})
        else:
            self.losspred_head = None
            
        # Now manually load the checkpoint
        self.path = path
        if self.path is not None:
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(self.path + "/SENet", map_location=lambda storage, loc: storage)
            # strict=False because logvar_head was not saved in the backbone state_dict
            self.model.load_state_dict(w_dict['state_dict'], strict=False)
            # The optimizer state is only needed for RESUMING training. Diagnostics that
            # load checkpoints for eval-only feature extraction don't use it, and a
            # param-group mismatch (e.g. checkpoint trained under a different optimizer
            # config or torch version) should not hard-fail them.
            try:
                self.optimizer.load_state_dict(w_dict['optimizer'])
            except (ValueError, RuntimeError) as e:
                print(f"WARNING: could not load optimizer state ({e}); continuing with "
                      f"a fresh optimizer (model weights loaded)")
            self.epoch = w_dict['epoch'] + 1
            if 'scheduler' in w_dict:
                try:
                    self.scheduler.load_state_dict(w_dict['scheduler'])
                except (ValueError, RuntimeError) as e:
                    print(f"WARNING: could not load scheduler state ({e})")
            print("dict epoch:", w_dict['epoch'])
            print("info", w_dict['info'])
            self.info = w_dict['info']

    def beam_drop(self, in_vol, p=0.5):
        """ Voxel Dropout (Sparsity) """
        bs, channels, h, w = in_vol.shape
        result = in_vol.clone()
        for b in range(bs):
            num_drop = int(h * p)
            indices = np.random.choice(h, num_drop, replace=False)
            result[b, :, indices, :] = 0
        return result

    def z_jitter(self, in_vol, std=0.2):
        """ Anisotropic Gaussian Jitter on depth """
        # in_vol[:, 0, :, :] is usually depth/range
        result = in_vol.clone()
        mask = result[:, 0, :, :] > 0
        noise = torch.randn_like(result[:, 0, :, :]) * std
        result[:, 0, :, :] += (noise * mask.float())
        return result
        
    def volumetric_noise_injection(self, in_vol, density=0.05):
        """ Additive Augmentation: Inject fake geometric returns into empty space """
        result = in_vol.clone()
        # Find empty space (where depth is 0)
        empty_mask = result[:, 0, :, :] == 0
        # Randomly select a percentage of empty space
        inject_mask = (torch.rand_like(empty_mask.float()) < density) & empty_mask
        
        # Inject uniformly distributed depth noise (e.g., between 0 and 50)
        # Assuming channel 0 is depth, which usually scales between 0 and some max.
        # We can just sample from uniform [0, 1] if it's normalized, or use random non-empty depths.
        noise = torch.rand_like(result[:, 0, :, :])
        
        # Broadcast inject mask across channels
        inject_mask_expanded = inject_mask.unsqueeze(1).expand_as(result)
        noise_expanded = torch.rand_like(result) * 2 - 1 # Random features for XYZ and remission
        noise_expanded[:, 0, :, :] = noise # Depth channel is strictly positive
        
        result[inject_mask_expanded] = noise_expanded[inject_mask_expanded]
        return result

    def sor_filter(self, in_vol):
        """ Pre-Network Spatial Filtering: Approximation of Radius Outlier Removal using 2D Pooling """
        valid = (in_vol[:, 0:1, :, :] > 0).float()
        # Count neighbors in 3x3 grid
        kernel = torch.ones(1, 1, 3, 3, device=in_vol.device)
        kernel[0, 0, 1, 1] = 0 # Don't count self
        
        # We use F.conv2d to count neighbors
        with torch.no_grad():
            neighbors = F.conv2d(valid, kernel, padding=1)
            
        # Keep points that have at least 1 neighbor
        keep = (neighbors >= 1).float()
        return in_vol * keep

    def get_augmented_view(self, in_vol):
        # Compose dropout, jitter, and density subsampling
        out = self.beam_drop(in_vol)
        out = self.z_jitter(out)
        
        # Density Subsampling (Randomly drop 20% of points to simulate lidar sparsity)
        mask = (torch.rand_like(out[:, :1, :, :]) > 0.2).float()
        out = out * mask
        
        if self.method == 'supcon_vib_additive':
            out = self.volumetric_noise_injection(out, density=0.05)

        if self.method == 'supcon_vib_losspred':
            # Crosstalk-style augmentation (Phase 25.6): sparse wrong-beam returns (low
            # injection density into empty space) so the loss-prediction head sees
            # crosstalk-hard points during training, not just the fog-ish views.
            out = self.volumetric_noise_injection(out, density=0.005)

        return out

    def get_extreme_view(self, in_vol):
        # Phase 25.7 (hard-negative SupCon): the MILD view plus a crosstalk-style sparse
        # wrong-beam injection. Used ONLY for the same-class repulsion term: extreme-
        # augmented points are pushed AWAY from the clean anchors of their class, carving
        # a distinct artifact sub-cluster instead of being absorbed into the class.
        out = self.get_augmented_view(in_vol)
        return self.volumetric_noise_injection(out, density=0.005)

    def supcon_loss(self, z8, z8_aug, proj_labels, tau=0.1, max_pts=2000,
                    blend_alpha=None, cond=False, channels=None):
        """Decoupled SupCon on the 128D bottleneck (clean <-> augmented, L2-normalized),
        the SAME term and temperature the supcon_vib branch uses (weight 0.1). This lets
        a DGLSS++ + SupCon direction test use the canonical mechanism rather than a
        variant-specific tuning.

        Direction-test variants (to check the anchoring trade-off is robust to tuning):
          - channels=k    : apply the pull to only the first k bottleneck channels,
                            leaving the rest free to retain the corruption shift.
          - blend_alpha   : pull each corrupted point toward its class's blended
                            anchor normalize((1-a)*clean_mean + a*corrupted_mean)
                            instead of the pairwise clean<->aug form.
          - cond=True     : weight each point's pull by its closeness to the clean
                            class anchor (near-clean points anchored, far/shifted
                            points anchored lightly so the recoverable shift survives).
        """
        mask = proj_labels > 0
        z_c = z8.permute(0, 2, 3, 1)[mask]
        z_a = z8_aug.permute(0, 2, 3, 1)[mask]
        lbl = proj_labels[mask]
        if len(lbl) == 0:
            return torch.tensor(0.0, device=z8.device)
        if len(lbl) > max_pts:
            idx = torch.randperm(len(lbl), device=z8.device)[:max_pts]
            z_c, z_a, lbl = z_c[idx], z_a[idx], lbl[idx]

        if channels is not None:
            z_c = z_c[:, :channels]
            z_a = z_a[:, :channels]

        z_c = F.normalize(z_c, p=2, dim=1)
        z_a = F.normalize(z_a, p=2, dim=1)

        if cond:
            # per-point anchor weight: closeness of the corrupted point to its clean
            # class centroid, normalized to [0,1] and detached (a gate, not a gradient
            # path). Far-from-clean (shifted) points get a small weight, so their
            # recoverable shift survives.
            Kc = int(lbl.max()) + 1
            centroid = torch.zeros(Kc, z_c.shape[1], device=z_c.device)
            centroid.scatter_add_(0, lbl.unsqueeze(1).expand(-1, z_c.shape[1]), z_c)
            cnt = torch.bincount(lbl, minlength=Kc).float().unsqueeze(1).clamp(min=1)
            clean_cent = F.normalize(centroid / cnt, p=2, dim=1)
            sim_c = (z_a * clean_cent[lbl]).sum(dim=1).detach()
            lo, hi = sim_c.min(), sim_c.max()
            w = ((sim_c - lo) / (hi - lo + 1e-8)).clamp(0, 1)
        else:
            w = None

        if blend_alpha is not None:
            # soft anchor: InfoNCE of corrupted points against the per-class blended
            # (clean + shifted) anchors, so the shift direction is retained.
            Kc = int(lbl.max()) + 1
            c_sum = torch.zeros(Kc, z_c.shape[1], device=z_c.device)
            a_sum = torch.zeros(Kc, z_a.shape[1], device=z_a.device)
            c_sum.scatter_add_(0, lbl.unsqueeze(1).expand(-1, z_c.shape[1]), z_c)
            a_sum.scatter_add_(0, lbl.unsqueeze(1).expand(-1, z_a.shape[1]), z_a)
            cnt = torch.bincount(lbl, minlength=Kc).float().unsqueeze(1).clamp(min=1)
            clean_cent = F.normalize(c_sum / cnt, p=2, dim=1)
            corr_cent = F.normalize(a_sum / cnt, p=2, dim=1)
            anchor = F.normalize((1 - blend_alpha) * clean_cent + blend_alpha * corr_cent, p=2, dim=1)
            sim = z_a @ anchor.T / tau
            pos = sim.gather(1, lbl.unsqueeze(1))
            max_sim, _ = torch.max(sim, dim=1, keepdim=True)
            loss = -(pos - (torch.logsumexp(sim - max_sim.detach(), dim=1) + max_sim.detach()))
        else:
            sim = z_c @ z_a.T / tau
            lbl_mat = lbl.unsqueeze(0) == lbl.unsqueeze(1)
            max_sim, _ = torch.max(sim, dim=1, keepdim=True)
            exp_sim = torch.exp(sim - max_sim.detach())
            pos_sum = (exp_sim * lbl_mat).sum(dim=1)
            all_sum = exp_sim.sum(dim=1)
            loss = -torch.log(pos_sum / (all_sum + 1e-8))

        if w is not None:
            return (loss * w).sum() / w.sum()
        return loss.mean()

    def vib_loss(self, z8, z8_aug):
        """VIB magnitude-bottleneck KL on the 128D bottleneck, the same form the
        supcon_vib branch uses (weight 0.01)."""
        logvar_aug = self.logvar_head(z8_aug)
        loss_kl_aug = -0.5 * torch.sum(1 + logvar_aug - z8_aug.pow(2) - logvar_aug.exp(), dim=1).mean()
        logvar_clean = self.logvar_head(z8)
        loss_kl_clean = -0.5 * torch.sum(1 + logvar_clean - z8.pow(2) - logvar_clean.exp(), dim=1).mean()
        return (loss_kl_clean + loss_kl_aug) / 2.0

    def nn_pull_loss(self, z, proj_labels, max_pts=2000):
        """Neighborhood-purity regularizer (Iteration 13.2): pull each point toward its
        nearest SAME-CLASS neighbor in the (corrupted) bottleneck view, i.e. minimize
        1 - cos(z_i, NN_same_class(z_i)). Directly maximizes the 1-NN same-class purity
        (nn1) that drives both the label ceiling and the AL readiness (rho(nn1, oracle)
        ~+0.7-0.9). Runs on the same subsample as the SupCon, so it adds ~1% to the
        step time."""
        mask = proj_labels > 0
        zf = z.permute(0, 2, 3, 1)[mask]
        lbl = proj_labels[mask]
        if len(lbl) == 0:
            return torch.tensor(0.0, device=z.device)
        if len(lbl) > max_pts:
            idx = torch.randperm(len(lbl), device=z.device)[:max_pts]
            zf, lbl = zf[idx], lbl[idx]
        zn = F.normalize(zf, p=2, dim=1)
        sim = zn @ zn.T
        n = len(zn)
        self_mask = ~torch.eye(n, dtype=torch.bool, device=zn.device)
        same = (lbl.unsqueeze(0) == lbl.unsqueeze(1)) & self_mask
        sim = sim.masked_fill(~same, -1e4)   # -1e9 overflows fp16 under autocast
        has = same.any(dim=1)
        if not has.any():
            return torch.tensor(0.0, device=z.device)
        best = sim[has].max(dim=1).values
        return (1.0 - best).mean()

    def ball_loss(self, z, proj_labels, max_pts=2000, momentum=0.99):
        """AL-oriented intra-class ball tightening (Iteration-11 line): pull each
        point toward its class's EMA center in cosine, i.e. minimize the angular
        radius of each class ball. The measured fat-blob geometry (intra-class cosine
        0.62-0.70, points 45-50 deg from their mean) is what makes: (a) the prototype
        (R1) metric fail (boundary flips), (b) class-mean estimation need k>=8 points,
        and (c) small T errors amplify into large W errors. Shrinking the ball attacks
        all three at the source. The EMA center is the class mean of the (corrupted)
        augmented view, so the tightening applies to the TTA-relevant view."""
        mask = proj_labels > 0
        zf = z.permute(0, 2, 3, 1)[mask]
        lbl = proj_labels[mask]
        if len(lbl) == 0:
            return torch.tensor(0.0, device=z.device)
        if len(lbl) > max_pts:
            idx = torch.randperm(len(lbl), device=z.device)[:max_pts]
            zf, lbl = zf[idx], lbl[idx]
        zn = F.normalize(zf, p=2, dim=1)
        loss = torch.tensor(0.0, device=z.device)
        n_terms = 0
        for c in torch.unique(lbl):
            mc = lbl == c
            if int(mc.sum().item()) < 2:
                continue
            zc = zn[mc]
            cent = F.normalize(zc.mean(dim=0), p=2, dim=0)
            loss = loss + (1.0 - (zc @ cent).mean())
            n_terms += 1
        return loss / max(1, n_terms)

    def spectrum_loss(self, z, max_pts=4000, eps=1e-4):
        """AL-oriented covariance conditioning (Iteration-11 line): penalize the
        condition number (lambda_max / lambda_min) of the centered feature
        covariance on a batch subsample. The measured ill-conditioning of the
        pool covariance (gain q99 ~50-130, the 4-6x ridge-relevant error, the
        fractional update needing beta < 1) is exactly the amplification the
        inverse covariance applies to label-statistic errors. Flattening the
        spectrum at TRAIN time makes the TTA/AL probe update less sensitive."""
        with torch.autocast(device_type='cuda', enabled=False):
            zf = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]).float()
            if len(zf) < 10:
                return torch.tensor(0.0, device=z.device)
            if len(zf) > max_pts:
                idx = torch.randperm(len(zf), device=z.device)[:max_pts]
                zf = zf[idx]
            zc = zf - zf.mean(dim=0, keepdim=True)
            cov = zc.t() @ zc / (len(zc) - 1 + 1e-8)
            eig = torch.linalg.eigvalsh(cov).clamp(min=eps)
            cond = eig[-1] / eig[0]
            return cond / (1.0 + cond)   # bounded in (0, 1)

    def dircons_loss(self, z8, z8_aug, proj_labels, max_pts=2000, momentum=0.99,
                     fragile_only=False):
        """Displacement-direction consistency (Iteration-15 idea #3, Iteration-16
        direction #2): each point's residual corr-shift dz = z_corr - z_inv is pulled
        toward its class's EMA displacement DIRECTION (detached target), so same-class
        corrupted points move coherently. This is deliberately DIRECTION-only (no
        magnitude constraint): it is the weakest structural commitment and directly
        targets rho(dir_retention, oracle) = +0.55..+0.81. The corr branch must be
        residual mode (corr = inv + dz); operates on the corrupted (augmented) view
        only -- the clean view's residual stays small via L_res. Runs on the same
        subsample as the SupCon, so ~1% of the step time.
        With fragile_only=True the consistency is applied ONLY to the casualty classes
        (2/7/13/14/15), leaving the healthy classes' residual uncoupled -- the
        Iteration-19 finding that the all-class coupling costs the healthy conditions."""
        mask = proj_labels > 0
        z_a = z8_aug.permute(0, 2, 3, 1)[mask]
        lbl = proj_labels[mask]
        if len(lbl) == 0:
            return torch.tensor(0.0, device=z8.device)
        if fragile_only:
            frag = torch.tensor(sorted(FRAGILE_CLASSES), device=lbl.device)
            keep = torch.isin(lbl, frag)
            if not keep.any():
                return torch.tensor(0.0, device=z8.device)
            z_a, lbl = z_a[keep], lbl[keep]
        if len(lbl) > max_pts:
            idx = torch.randperm(len(lbl), device=z8.device)[:max_pts]
            z_a, lbl = z_a[idx], lbl[idx]

        # residual shift on the corrupted view: dz = (z_a_corr - z_a_inv)
        dz = F.normalize(z_a[:, self.inv_dim:] - z_a[:, :self.corr_dim], p=2, dim=1)

        # per-class EMA of the corrupted displacement direction
        K = int(lbl.max()) + 1
        D = dz.shape[1]
        cur = torch.zeros(K, D, device=dz.device)
        cur.scatter_add_(0, lbl.unsqueeze(1).expand(-1, D), dz)
        cnt = torch.bincount(lbl, minlength=K).float().unsqueeze(1).clamp(min=1)
        # normalize per present class; classes absent from this batch get a zero row
        # (F.normalize(0)=NaN would poison the EMA)
        norms = cur.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        cur = cur / norms
        if self._dir_ema is None or self._dir_ema.shape[0] != K:
            self._dir_ema = cur.detach()
        else:
            self._dir_ema = F.normalize(
                momentum * self._dir_ema + (1 - momentum) * cur, p=2, dim=1).detach()

        target = self._dir_ema[lbl]
        return (1.0 - (dz * target).sum(dim=1)).mean()

    def distill_loss(self, in_vol_aug, z8, proj_labels, max_pts=2000):
        """Teacher-preserved ceiling branch (feedback direction): distill the CORR
        branch (z8[:, inv_dim:]) toward the FROZEN plain-DGLSS++ corrupted-view
        features via cosine geometry, on the corrupted (augmented) view only. Unlike
        dircons, the target is a KNOWN-good extractor's geometry, not the network's
        own EMA displacement -- so the branch is asked to reproduce a fixed target
        rather than invent one. TTA stays on the inv branch untouched; the HDC
        decoder reads the concat [inv, corr]."""
        mask = proj_labels > 0
        with torch.no_grad():
            t_out = self.teacher_model(in_vol_aug)
            if len(t_out) == 3:
                _, _, z_t = t_out
            else:
                _, z_t = t_out
        z_c = z8[:, self.inv_dim:].permute(0, 2, 3, 1)[mask]
        z_t = z_t.permute(0, 2, 3, 1)[mask]
        lbl = proj_labels[mask]
        if len(lbl) == 0:
            return torch.tensor(0.0, device=z8.device)
        if len(lbl) > max_pts:
            idx = torch.randperm(len(lbl), device=z8.device)[:max_pts]
            z_c, z_t = z_c[idx], z_t[idx]
        z_c = F.normalize(z_c, p=2, dim=1)
        z_t = F.normalize(z_t, p=2, dim=1)
        return (1.0 - (z_c * z_t).sum(dim=1)).mean()

    def hdc_loss(self, z8, z8_aug, proj_labels, max_pts=2000, tau=0.1):
        """HDC-aware soft-prototype loss (feedback direction 4): train the continuous
        features so their BINARIZED code has class margin. Differentiable surrogate:
        per class, pool a clean HDC prototype sign(z @ R) (detached), then pull each
        corrupted point's SOFT code (the pre-sign continuous projection, normalized)
        toward its class prototype via a cosine CE. Optimizes the exact geometry the
        decoder reads, rather than the continuous-space geometry the dircons line
        attacked. R is the same seeded projection the eval uses, built ONCE in
        __init__ (get_hdc_projection resets the global RNG, so it must not run in
        the training loop)."""
        if self._hdc_proj is None or self._hdc_proj.shape[0] != z8.shape[1]:
            # fallback for an unexpected dim (shouldn't happen; HDC_VARIANTS are 128D)
            from modules.oracle_core import get_hdc_projection
            rng_state = torch.get_rng_state()
            self._hdc_proj = get_hdc_projection(dim_in=z8.shape[1], dim_out=10000,
                                                device=z8.device)
            torch.set_rng_state(rng_state)
        proj = self._hdc_proj
        mask = proj_labels > 0
        zc = z8.permute(0, 2, 3, 1)[mask]
        za = z8_aug.permute(0, 2, 3, 1)[mask]
        lbl = proj_labels[mask]
        if len(lbl) == 0:
            return torch.tensor(0.0, device=z8.device)
        if len(lbl) > max_pts:
            idx = torch.randperm(len(lbl), device=z8.device)[:max_pts]
            zc, za, lbl = zc[idx], za[idx], lbl[idx]

        # clean class HDC prototypes (detached): sign code, L2-normalized per class
        hc = torch.sign(zc @ proj).float()
        K = int(lbl.max()) + 1
        protos = torch.zeros(K, proj.shape[1], device=z8.device)
        protos.scatter_add_(0, lbl.unsqueeze(1).expand(-1, proj.shape[1]), hc)
        cnt = torch.bincount(lbl, minlength=K).float().unsqueeze(1).clamp(min=1)
        protos = F.normalize(protos / cnt, p=2, dim=1).detach()
        # corrupted SOFT code (differentiable pre-sign projection), cosine to prototypes
        ua = F.normalize(za @ proj, p=2, dim=1)
        logits = ua @ protos.T / tau
        return F.cross_entropy(logits, lbl)

    def anti_anchor_loss(self, z8, z8_aug, proj_labels, max_pts=2000):
        """Anti-anchor (Iteration-19.8 diagnosis): PENALIZE the corrupted->clean class
        cosine so the network retains the corruption shift instead of erasing it.
        Every objective in the family (GMSIFC/LSCC/SupCon/dircons/corrsc/hdc) pulls
        corrupted toward clean or toward a coherence; this is the inverse. For each
        corrupted point, compute the cosine to its CLEAN class centroid (mean of the
        clean view's same-class features, detached) and ADD that cosine to the loss --
        so the optimizer is discouraged from moving the corrupted feature onto the
        clean centroid. This directly tests whether plain DGLSS++'s ceiling comes
        from never being told to undo corruption."""
        mask = proj_labels > 0
        zc = z8.permute(0, 2, 3, 1)[mask]
        za = z8_aug.permute(0, 2, 3, 1)[mask]
        lbl = proj_labels[mask]
        if len(lbl) == 0:
            return torch.tensor(0.0, device=z8.device)
        if len(lbl) > max_pts:
            idx = torch.randperm(len(lbl), device=z8.device)[:max_pts]
            zc, za, lbl = zc[idx], za[idx], lbl[idx]
        # clean class centroids (detached) from the clean view
        K = int(lbl.max()) + 1
        D = zc.shape[1]
        csum = torch.zeros(K, D, device=z8.device)
        csum.scatter_add_(0, lbl.unsqueeze(1).expand(-1, D), zc)
        cnt = torch.bincount(lbl, minlength=K).float().unsqueeze(1).clamp(min=1)
        centroids = F.normalize(csum / cnt, p=2, dim=1).detach()
        zn_a = F.normalize(za, p=2, dim=1)
        cos2clean = (zn_a * centroids[lbl]).sum(dim=1)
        # penalize the corrupted->clean cosine: high cosine (aligned) is penalized,
        # so the shift is retained. Clamp to [0,1] so it never goes negative/explodes.
        return cos2clean.clamp(min=0.0, max=1.0).mean()

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, color_fn, report=10, show_scans=False):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        if self.method == 'supcon_vib_evidential':
            self._edl_accum = {}

        if self.gpu:
            torch.cuda.empty_cache()

        evaluator.reset()
        model.train()
        
        scaler = torch.amp.GradScaler('cuda')
        max_steps = int(len(train_loader) * self.cutoff_percent)

        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(train_loader), total=max_steps):
            if i >= max_steps:
                break
            
            if self.gpu:
                in_vol, proj_labels = in_vol.cuda(), proj_labels.cuda().long()

            # Create augmented view for all methods. DGLSS / DGLSS++ use the pure
            # sparsity (beam-drop) view their consistency losses are defined on.
            if self.method.startswith('supcon_vib_dglsspp_cor'):
                # Robust DGLSS++ arm: corruption-targeted augmented view (fog depth
                # jitter + density sparsity from get_augmented_view, then crosstalk
                # fake-return injection) instead of the pure beam-drop view, so the
                # GMSIFC/LSCC consistency constraints learn invariance to the exact
                # corruptions that collapse the minority classes.
                in_vol_aug = self.get_augmented_view(in_vol)
                in_vol_aug = self.volumetric_noise_injection(in_vol_aug, density=0.005)
            elif self.method in DGLSS_METHODS:
                in_vol_aug = get_dglss_view(in_vol)
            else:
                in_vol_aug = self.get_augmented_view(in_vol)

            # corrsc: a SECOND independently corrupted view, so the corrupted-manifold
            # SupCon has two realizations of the same class to pull together.
            in_vol_aug2 = None
            if self.method in CORRSC_VARIANTS:
                in_vol_aug2 = self.get_augmented_view(in_vol)
                in_vol_aug2 = self.volumetric_noise_injection(in_vol_aug2, density=0.005)

            # SupCon+VIB+SOR: mirror the eval-time SOR pre-filter on both clean and augmented inputs
            if self.method == 'supcon_vib_sor':
                in_vol = self.sor_filter(in_vol)
                in_vol_aug = self.sor_filter(in_vol_aug)

            with torch.amp.autocast('cuda'):
                # Forward pass clean. The standard-implementation arm additionally
                # requests the deepest encoder stage (x_4) for the encoder-level SIFC.
                if self.method == 'supcon_vib_dglss_enc':
                    if self.ARCH["train"]["aux_loss"]:
                        output, aux_list, z8, x4 = model(in_vol, return_stage4=True)
                        output_aug, aux_list_aug, z8_aug, x4_aug = model(in_vol_aug, return_stage4=True)
                    else:
                        output, z8, x4 = model(in_vol, return_stage4=True)
                        output_aug, z8_aug, x4_aug = model(in_vol_aug, return_stage4=True)
                elif self.ARCH["train"]["aux_loss"]:
                    output, aux_list, z8 = model(in_vol)
                    output_aug, aux_list_aug, z8_aug = model(in_vol_aug)
                else:
                    output, z8 = model(in_vol)
                    output_aug, z8_aug = model(in_vol_aug)

                if in_vol_aug2 is not None:
                    if self.ARCH["train"]["aux_loss"]:
                        output_aug2, aux_list_aug2, z8_aug2 = model(in_vol_aug2)
                    else:
                        output_aug2, z8_aug2 = model(in_vol_aug2)

                # Standard semantic segmentation loss
                loss_ce = criterion(torch.log(output.clamp(min=1e-8)), proj_labels)
                loss_ce_aug = criterion(torch.log(output_aug.clamp(min=1e-8)), proj_labels)
                loss_sem = (loss_ce + loss_ce_aug) / 2.0
                
                loss_total = loss_sem
                
                # --- The 3 Methodologies ---
                
                if self.method == 'supcon':
                    # Unnormalized Supervised Contrastive
                    # We subsample points to avoid OOM
                    mask = proj_labels > 0
                    z_c = z8.permute(0, 2, 3, 1)[mask]
                    z_a = z8_aug.permute(0, 2, 3, 1)[mask]
                    lbl = proj_labels[mask]
                    
                    if len(lbl) > 2000:
                        idx = torch.randperm(len(lbl))[:2000]
                        z_c, z_a, lbl = z_c[idx], z_a[idx], lbl[idx]
                    
                    if len(lbl) > 0:
                        # Since features are unnormalized with magnitude 5-11, tau=0.1 blows up. 
                        # Unnormalized contrastive should use tau=1.0 or adaptive scaling.
                        tau = 1.0
                        sim_matrix = torch.matmul(z_c, z_a.T) / tau
                        lbl_matrix = lbl.unsqueeze(0) == lbl.unsqueeze(1)
                        
                        max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
                        exp_sim = torch.exp(sim_matrix - max_sim.detach())
                        pos_sum = (exp_sim * lbl_matrix).sum(dim=1)
                        all_sum = exp_sim.sum(dim=1)
                        loss_supcon = -torch.log(pos_sum / (all_sum + 1e-8)).mean()
                        
                        loss_total = loss_total + 0.1 * loss_supcon

                elif self.method == 'vib':
                    # Variational Information Bottleneck for BOTH clean and augmented
                    if self.logvar_head is None:
                        self.logvar_head = nn.Conv2d(z8_aug.shape[1], z8_aug.shape[1], kernel_size=1).to(self.device)
                        self.optimizer.add_param_group({'params': self.logvar_head.parameters()})

                    mu_aug = z8_aug
                    logvar_aug = self.logvar_head(z8_aug)
                    loss_kl_aug = -0.5 * torch.sum(1 + logvar_aug - mu_aug.pow(2) - logvar_aug.exp(), dim=1).mean()
                    
                    mu_clean = z8
                    logvar_clean = self.logvar_head(z8)
                    loss_kl_clean = -0.5 * torch.sum(1 + logvar_clean - mu_clean.pow(2) - logvar_clean.exp(), dim=1).mean()
                    
                    loss_kl = (loss_kl_clean + loss_kl_aug) / 2.0
                    
                    # We sample for the classification pass
                    std_aug = torch.exp(0.5 * logvar_aug)
                    eps_aug = torch.randn_like(std_aug)
                    z_sampled_aug = mu_aug + eps_aug * std_aug
                    
                    std_clean = torch.exp(0.5 * logvar_clean)
                    eps_clean = torch.randn_like(std_clean)
                    z_sampled_clean = mu_clean + eps_clean * std_clean
                    
                    # Route through the classification head to enforce the bottleneck
                    if hasattr(model, 'module'):
                        logits_sampled_aug = model.module.semantic_output(z_sampled_aug)
                        logits_sampled_clean = model.module.semantic_output(z_sampled_clean)
                    else:
                        logits_sampled_aug = model.semantic_output(z_sampled_aug)
                        logits_sampled_clean = model.semantic_output(z_sampled_clean)
                        
                    pred_sampled_aug = F.softmax(logits_sampled_aug, dim=1)
                    pred_sampled_clean = F.softmax(logits_sampled_clean, dim=1)
                    
                    loss_ce_aug = criterion(torch.log(pred_sampled_aug.clamp(min=1e-8)), proj_labels)
                    loss_ce_clean = criterion(torch.log(pred_sampled_clean.clamp(min=1e-8)), proj_labels)
                    
                    loss_sem = (loss_ce_clean + loss_ce_aug) / 2.0
                    loss_total = loss_sem + 0.01 * loss_kl

                elif self.method in DGLSS_METHODS:
                    # DGLSS / DGLSS++ (VIB-free, plain bottleneck, no reparameterization
                    # and no KL): the representation constraint is SIFC/GMSIFC + SCC/LSCC,
                    # applied to the 128D bottleneck (the HDC-input space) or, for the
                    # standard-implementation arm, SIFC on the deepest encoder stage x_4
                    # with SCC on the decoded bottleneck (matching the paper's split of
                    # SIFC on Phi_enc(F) and SCC on Psi(Phi_dec(F))).
                    # Variants (all keep the default beam-drop view, isolating the added
                    # mechanism; only _cor / _corsupcon* swap in the corruption view):
                    #   _supcon : + 0.1 * decoupled SupCon on the bottleneck
                    #   _bal    : class-balanced GMSIFC + LSCC contrastive
                    #   _vib    : + 0.01 * VIB magnitude-bottleneck KL
                    #   _corsupcon* : corruption view + SupCon; the _nogmsifc / _nolscc /
                    #                _nocons suffixes drop the GMSIFC / LSCC / both
                    #                consistency terms to ablate the DGLSS++ stack.
                    gmsifc = self.method.startswith('supcon_vib_dglsspp')
                    class_bal = self.method == 'supcon_vib_dglsspp_bal'
                    no_sifc = 'nogmsifc' in self.method or 'nocons' in self.method
                    no_scc = 'nolscc' in self.method or 'nocons' in self.method
                    loss_total = loss_sem
                    if not no_sifc:
                        if self.method == 'supcon_vib_dglss_enc':
                            loss_sifc = dglss_sifc_loss(x4, x4_aug, proj_labels, in_vol, in_vol_aug,
                                                        masked=False, tau=DGLSS_TAU)
                        elif self.corr_dim > 0:
                            # GMSIFC on the INVARIANT slice only: the corr branch must
                            # NOT be clean-view-aligned (that is what erases the shift).
                            loss_sifc = dglss_sifc_loss(z8[:, :self.inv_dim], z8_aug[:, :self.inv_dim],
                                                        proj_labels, in_vol, in_vol_aug,
                                                        masked=gmsifc, tau=DGLSS_TAU, class_bal=class_bal)
                        else:
                            loss_sifc = dglss_sifc_loss(z8, z8_aug, proj_labels, in_vol, in_vol_aug,
                                                        masked=gmsifc, tau=DGLSS_TAU, class_bal=class_bal)
                        loss_total = loss_total + self.dglss_lam1 * loss_sifc
                    if not no_scc:
                        if self.corr_dim > 0 and self.lscc_corr:
                            # LSCC on BOTH slices (decode structure for both branches).
                            loss_scc = (dglss_scc_loss(z8[:, :self.inv_dim], z8_aug[:, :self.inv_dim],
                                                       proj_labels, in_vol, in_vol_aug,
                                                       local=gmsifc, normalize=self.dglss_scc_norm,
                                                       class_bal=class_bal)
                                        + dglss_scc_loss(z8[:, self.inv_dim:], z8_aug[:, self.inv_dim:],
                                                         proj_labels, in_vol, in_vol_aug,
                                                         local=gmsifc, normalize=self.dglss_scc_norm,
                                                         class_bal=class_bal))
                        elif self.corr_dim > 0:
                            # _corrfree: DROP LSCC on the corr slice (Iteration-16:
                            # LSCC is a clean-view alignment term that re-anchors corr;
                            # leave CE on the full concat as corr's only clean pull).
                            loss_scc = dglss_scc_loss(z8[:, :self.inv_dim], z8_aug[:, :self.inv_dim],
                                                      proj_labels, in_vol, in_vol_aug,
                                                      local=gmsifc, normalize=self.dglss_scc_norm,
                                                      class_bal=class_bal)
                        else:
                            loss_scc = dglss_scc_loss(z8, z8_aug, proj_labels, in_vol, in_vol_aug,
                                                      local=gmsifc, normalize=self.dglss_scc_norm,
                                                      class_bal=class_bal)
                        loss_total = loss_total + self.dglss_lam2 * loss_scc
                    if 'corsupcon' in self.method:
                        cfg = SUPCON_VARIANTS.get(self.method, {})
                        supcon_kw = {k: v for k, v in cfg.items()
                                     if k not in ('weight', 'coclust_w', 'nnpull_w',
                                                  'ball_w', 'spec_w')}
                        if self.corr_dim > 0:
                            # SupCon on the INVARIANT slice only (the clean anchor is
                            # exactly what erases the corr branch's recoverable shift).
                            z8_anchor = z8[:, :self.inv_dim]
                            z8_aug_anchor = z8_aug[:, :self.inv_dim]
                        else:
                            z8_anchor, z8_aug_anchor = z8, z8_aug
                        loss_total = loss_total + cfg.get('weight', 0.1) * self.supcon_loss(
                            z8_anchor, z8_aug_anchor, proj_labels, **supcon_kw)
                        if 'ball_w' in cfg:
                            # AL-oriented: intra-class ball tightening. Pull each point
                            # toward its class's EMA center (cosine), directly shrinking
                            # the fat-blob radius (intra-cos 0.62-0.70) that drives the
                            # mean-estimation sample complexity, the prototype metric's
                            # viability, and the T-error -> W-error amplification.
                            loss_total = loss_total + cfg['ball_w'] * self.ball_loss(
                                z8_aug_anchor, proj_labels)
                        if 'spec_w' in cfg:
                            # AL-oriented: covariance conditioning. Penalize the batch
                            # covariance's condition number (lambda_max / lambda_min of
                            # the centered feature covariance + eps), flattening the
                            # spectrum that the inverse covariance amplifies (the 4-6x
                            # ridge-relevant error, Iteration 8-10).
                            loss_total = loss_total + cfg['spec_w'] * self.spectrum_loss(
                                z8_aug_anchor)
                        if 'coclust_w' in cfg:
                            # corrupted-only clustering: pull the corrupted points
                            # toward their CORRUPTED class centroids (blend_alpha=1.0),
                            # maximizing intra-corrupted packing while leaving the
                            # shifted direction intact (Iteration-12 ceiling drivers).
                            loss_total = loss_total + cfg['coclust_w'] * self.supcon_loss(
                                z8_anchor, z8_aug_anchor, proj_labels, blend_alpha=1.0)
                        if 'nnpull_w' in cfg:
                            # neighborhood-purity regularizer: pull each corrupted point
                            # toward its nearest SAME-CLASS neighbor, directly raising
                            # the 1-NN purity that drives the ceiling and AL readiness.
                            loss_total = loss_total + cfg['nnpull_w'] * self.nn_pull_loss(
                                z8_aug_anchor, proj_labels)
                        if self.corr_dim > 0 and self.corr_mode == 'res':
                            # residual-shift penalty: keep the corr deformation small
                            # (used only when the corruption needs it).
                            loss_total = loss_total + self.res_w * (
                                (z8[:, self.inv_dim:] - z8[:, :self.corr_dim]).pow(2).mean()
                                + (z8_aug[:, self.inv_dim:] - z8_aug[:, :self.corr_dim]).pow(2).mean()) / 2.0
                        if self.corr_dim > 0 and self.dir_w > 0:
                            # displacement-direction consistency (idea #3): same-class
                            # corrupted points move coherently (direction-only).
                            loss_total = loss_total + self.dir_w * self.dircons_loss(
                                z8, z8_aug, proj_labels, fragile_only=self.dir_fragile)
                        if self.corr_dim > 0 and self.teacher_w > 0:
                            # teacher-preserved ceiling branch: distill the corr branch
                            # toward the frozen plain-DGLSS++ corrupted geometry.
                            loss_total = loss_total + self.teacher_w * self.distill_loss(
                                in_vol_aug, z8, proj_labels)
                        if self.method in CORRSC_VARIANTS and in_vol_aug2 is not None:
                            # corruption-manifold multi-positive SupCon: pull same-class
                            # CORRUPTED realizations together (not corr->clean), so each
                            # class forms a good corrupted manifold. Applies to the corr
                            # slice when a corrfree corr head exists, else the full view.
                            cw = CORRSC_VARIANTS[self.method]
                            if self.corr_dim > 0:
                                za1 = z8_aug2[:, self.inv_dim:]
                                za2 = z8_aug[:, self.inv_dim:]
                                z_anchor = z8[:, self.inv_dim:]
                                z_anchor_aug = z8_aug[:, self.inv_dim:]
                                # corrfree_corrsc: weak clean anchor on the CORR slice
                                # (the standard block anchors the INV slice at 0.1 for
                                # TTA; this weak term keeps the free corr branch from
                                # drifting unanchored).
                                loss_total = loss_total + 0.02 * self.supcon_loss(
                                    z_anchor, z_anchor_aug, proj_labels)
                            else:
                                za1, za2 = z8_aug2, z8_aug
                                # single-branch corrsc: the standard SupCon block already
                                # provides the weak 0.02 clean anchor (SUPCON_VARIANTS
                                # cap); do not double-add here.
                            loss_total = loss_total + cw * self.supcon_loss(
                                za1, za2, proj_labels)
                        if self.method in HDC_VARIANTS:
                            # HDC-aware soft-prototype loss: pull the corrupted view's
                            # soft HDC code toward the class's clean HDC prototype
                            # (margin in the binarized geometry, not the continuous one).
                            loss_total = loss_total + HDC_VARIANTS[self.method] * self.hdc_loss(
                                z8, z8_aug, proj_labels)
                        if self.method in ANTI_ANCHOR_VARIANTS:
                            # anti-anchor: penalize the corrupted->clean cosine so the
                            # corruption shift is retained (Iteration-19.8 diagnosis).
                            loss_total = loss_total + ANTI_ANCHOR_VARIANTS[self.method] * self.anti_anchor_loss(
                                z8, z8_aug, proj_labels)
                    if self.method == 'supcon_vib_dglsspp_vib':
                        loss_total = loss_total + 0.01 * self.vib_loss(z8, z8_aug)
                elif self.method.startswith('supcon_vib'):
                    # Decoupled SupCon + VIB
                    # 1. VIB Magnitude Bottleneck (Absolute Space)
                    if self.logvar_head is None:
                        self.logvar_head = nn.Conv2d(z8_aug.shape[1], z8_aug.shape[1], kernel_size=1).to(self.device)
                        self.optimizer.add_param_group({'params': self.logvar_head.parameters()})

                    mu_aug = z8_aug
                    logvar_aug = self.logvar_head(z8_aug)
                    loss_kl_aug = -0.5 * torch.sum(1 + logvar_aug - mu_aug.pow(2) - logvar_aug.exp(), dim=1).mean()
                    
                    mu_clean = z8
                    logvar_clean = self.logvar_head(z8)
                    loss_kl_clean = -0.5 * torch.sum(1 + logvar_clean - mu_clean.pow(2) - logvar_clean.exp(), dim=1).mean()
                    
                    loss_kl = (loss_kl_clean + loss_kl_aug) / 2.0
                    
                    std_aug = torch.exp(0.5 * logvar_aug)
                    eps_aug = torch.randn_like(std_aug)
                    z_sampled_aug = mu_aug + eps_aug * std_aug
                    
                    std_clean = torch.exp(0.5 * logvar_clean)
                    eps_clean = torch.randn_like(std_clean)
                    z_sampled_clean = mu_clean + eps_clean * std_clean
                    
                    if hasattr(model, 'module'):
                        logits_sampled_aug = model.module.semantic_output(z_sampled_aug)
                        logits_sampled_clean = model.module.semantic_output(z_sampled_clean)
                    else:
                        logits_sampled_aug = model.semantic_output(z_sampled_aug)
                        logits_sampled_clean = model.semantic_output(z_sampled_clean)
                        
                    pred_sampled_aug = F.softmax(logits_sampled_aug, dim=1)
                    pred_sampled_clean = F.softmax(logits_sampled_clean, dim=1)
                    
                    loss_ce_aug = criterion(torch.log(pred_sampled_aug.clamp(min=1e-8)), proj_labels)
                    loss_ce_clean = criterion(torch.log(pred_sampled_clean.clamp(min=1e-8)), proj_labels)
                    
                    loss_sem = (loss_ce_clean + loss_ce_aug) / 2.0
                    
                    # 2. SupCon Angular Margins (Normalized Space)
                    mask = proj_labels > 0
                    z_c = mu_clean.permute(0, 2, 3, 1)[mask]
                    z_a = mu_aug.permute(0, 2, 3, 1)[mask]
                    lbl = proj_labels[mask]
                    
                    loss_supcon = torch.tensor(0.0, device=z8.device)
                    subsampled = False
                    if len(lbl) > 2000:
                        idx = torch.randperm(len(lbl))[:2000]
                        z_c, z_a, lbl = z_c[idx], z_a[idx], lbl[idx]
                        subsampled = True
                        
                    if self.method == 'supcon_vib_hardneg':
                        # Phase 25.7: the extreme (crosstalk-injected) view, aligned to the
                        # same subsample, for the same-class repulsion term.
                        out_ext = model(self.get_extreme_view(in_vol))
                        z8_ext = out_ext[2] if len(out_ext) == 3 else out_ext[1]
                        output_ext = out_ext[0]
                        z_ext = z8_ext.permute(0, 2, 3, 1)[mask]
                        if subsampled:
                            z_ext = z_ext[idx]

                    if len(lbl) > 0:
                        # CRITICAL FIX: L2 Normalize features for SupCon to prevent gradient tug-of-war with VIB
                        z_c_norm = F.normalize(z_c, p=2, dim=1)
                        z_a_norm = F.normalize(z_a, p=2, dim=1)
                        
                        tau = 0.1
                        sim_matrix = torch.matmul(z_c_norm, z_a_norm.T) / tau
                        lbl_matrix = lbl.unsqueeze(0) == lbl.unsqueeze(1)
                        
                        max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
                        exp_sim = torch.exp(sim_matrix - max_sim.detach())
                        pos_sum = (exp_sim * lbl_matrix).sum(dim=1)
                        all_sum = exp_sim.sum(dim=1)
                        if self.method == 'supcon_vib_fragile':
                            # Phase 25 Addition 1: per-anchor weighting that up-weights the
                            # casualty classes (2/7/13/14/15) so their corrupted-view
                            # separability gets the contrastive signal. Target: move their
                            # per-class fog LP corrupt accuracy off ~0 (Iteration 4B).
                            frag = torch.tensor(sorted(FRAGILE_CLASSES), device=lbl.device)
                            anchor_w = torch.where(torch.isin(lbl, frag),
                                                   self.fragile_w, 1.0).float()
                            loss_supcon = (-(anchor_w * torch.log(pos_sum / (all_sum + 1e-8)))
                                           .sum() / anchor_w.sum())
                        else:
                            loss_supcon = -torch.log(pos_sum / (all_sum + 1e-8)).mean()

                        # Phase 25.7 (hard-negative SupCon): push each extreme-augmented point
                        # AWAY from its class's clean centroid (the clean anchor) above the
                        # margin, so crosstalk-style artifacts form a distinct sub-cluster
                        # instead of being absorbed into the class centroid. Keeps the
                        # mild-view attraction (robustness).
                        if self.method == 'supcon_vib_hardneg':
                            z_ext_norm = F.normalize(z_ext, p=2, dim=1)
                            Kc = int(lbl.max()) + 1
                            centroid = torch.zeros(Kc, z_c_norm.shape[1], device=z_c_norm.device)
                            centroid.scatter_add_(0, lbl.unsqueeze(1).expand(-1, z_c_norm.shape[1]),
                                                  z_c_norm)
                            counts = torch.bincount(lbl, minlength=Kc).float().unsqueeze(1)
                            centroid = F.normalize(centroid / counts.clamp(min=1), p=2, dim=1)
                            sim_centroid = (z_ext_norm * centroid[lbl]).sum(dim=1)
                            loss_repel = F.relu(sim_centroid - HARDNEG_MARGIN).mean()
                            loss_total = loss_total + self.edl_w * loss_repel
                        
                    # VIB pressure variants (Phase 17: 5x at medium scale over-collapsed
                    # the clean manifold; midvib = 3x as the intermediate probe)
                    kl_weight = {'supcon_vib_strongvib': 0.05,
                                 'supcon_vib_midvib': 0.03}.get(self.method, 0.01)
                    loss_total = loss_sem + kl_weight * loss_kl + 0.1 * loss_supcon

                    # Phase 25 Addition 2 (evidential head): Dirichlet evidence on the 128D
                    # bottleneck. Two terms on the valid pixels:
                    #   - evidential cross-entropy (expected log-likelihood under the Dirichlet)
                    #     on BOTH views, so the head classifies;
                    #   - a KL-to-uniform regularizer on the AUGMENTED view ONLY, forcing high
                    #     epistemic uncertainty on the corruption-hard points (the Phase 22.2
                    #     confident-and-wrong failure). Annealed in per Sensoy et al.
                    if self.method == 'supcon_vib_evidential':
                        m = proj_labels > 0
                        al = (F.softplus(self.evidence_head(z8)) + 1.0).permute(0, 2, 3, 1)[m]
                        al_a = (F.softplus(self.evidence_head(z8_aug)) + 1.0).permute(0, 2, 3, 1)[m]
                        lbl_e = proj_labels[m]
                        if len(lbl_e) > 0:
                            S = al.sum(dim=1)
                            Sa = al_a.sum(dim=1)
                            al_t = al.gather(1, lbl_e.unsqueeze(1)).squeeze(1)
                            al_a_t = al_a.gather(1, lbl_e.unsqueeze(1)).squeeze(1)
                            loss_edl = (torch.digamma(S) - torch.digamma(al_t)).mean()
                            loss_edl_aug = (torch.digamma(Sa) - torch.digamma(al_a_t)).mean()
                            y_onehot = F.one_hot(lbl_e, num_classes=al_a.shape[1]).float()
                            atilde = al_a * (1 - y_onehot) + 1.0
                            if self.edl_kl_selective:
                                # Fix (b), Phase 25.4: apply the KL only to augmented points the
                                # head CURRENTLY predicts wrong, so correct points build evidence
                                # while hard points get pushed to high uncertainty. Condition-
                                # agnostic ("be uncertain where wrong"), which is the gating signal
                                # needed on BOTH fog and crosstalk (the blanket KL calibrated fog
                                # but not crosstalk).
                                wrong = al_a.argmax(dim=1) != lbl_e
                                if int(wrong.sum().item()) > 0:
                                    atilde = atilde[wrong]
                                else:
                                    atilde = torch.zeros(0, al_a.shape[1], device=al.device)
                            St = atilde.sum(dim=1, keepdim=True)
                            Kc = al_a.shape[1]
                            kl_aug = (torch.lgamma(St)
                                      - torch.lgamma(atilde).sum(dim=1, keepdim=True)
                                      + ((atilde - 1) * (torch.digamma(atilde)
                                                         - torch.digamma(St))).sum(dim=1, keepdim=True)
                                      + torch.lgamma(torch.tensor(Kc, device=al.device))).mean() if len(atilde) > 0 else torch.tensor(0.0, device=al.device)
                            lam_kl = min(self.edl_kl_cap, epoch / 100.0)
                            loss_total = loss_total + self.edl_w * (loss_edl + loss_edl_aug) + lam_kl * kl_aug
                            # running loss-component log (KL-domination diagnostic)
                            for k, v in [('edl', loss_edl.item()), ('edl_aug', loss_edl_aug.item()),
                                         ('kl_aug', kl_aug.item()), ('kl_w', lam_kl),
                                         ('edl_ratio', (loss_edl.item() + loss_edl_aug.item()) /
                                          max(loss_sem.item(), 1e-6)),
                                         ('kl_ratio', (kl_aug.item() * lam_kl) /
                                          max(loss_sem.item(), 1e-6))]:
                                self._edl_accum[k] = self._edl_accum.get(k, 0.0) + v

                    # Phase 25.6/25.7 (direct loss prediction): regress the main classifier's
                    # per-point CE. For the hardneg method, ALSO regress the EXTREME view's CE
                    # (the artifact-point error), directly tying the head to the separated
                    # artifact sub-cluster the hard-neg repulsion creates.
                    if self.method in ('supcon_vib_losspred', 'supcon_vib_hardneg'):
                        m = proj_labels > 0
                        # Targets detached: the head regresses the model's error, it does
                        # not steer it (canonical loss-prediction, Yoo & Kweon). Undetached
                        # targets pushed the model to make its loss PREDICTABLE, fighting loss_sem.
                        target_c = F.cross_entropy(output, proj_labels, reduction='none')[m].detach()
                        target_a = F.cross_entropy(output_aug, proj_labels, reduction='none')[m].detach()
                        pred_c = F.softplus(self.losspred_head(z8)).permute(0, 2, 3, 1)[m, 0]
                        pred_a = F.softplus(self.losspred_head(z8_aug)).permute(0, 2, 3, 1)[m, 0]
                        if len(target_c) > 0:
                            loss_lp = (F.smooth_l1_loss(pred_c, target_c)
                                       + F.smooth_l1_loss(pred_a, target_a))
                            if self.method == 'supcon_vib_hardneg':
                                target_x = F.cross_entropy(output_ext, proj_labels, reduction='none')[m].detach()
                                pred_x = F.softplus(self.losspred_head(z8_ext)).permute(0, 2, 3, 1)[m, 0]
                                loss_lp = loss_lp + F.smooth_l1_loss(pred_x, target_x)
                            loss_total = loss_total + self.edl_w * loss_lp

                elif self.method == 'smoothness':
                    # Local Smoothness (Dirichlet Energy)
                    # We gate the difference penalty to only apply if the adjacent pixels share the same class label
                    diff_y = torch.norm(z8_aug[:, :, 1:, :] - z8_aug[:, :, :-1, :], dim=1)
                    mask_y = (proj_labels[:, 1:, :] == proj_labels[:, :-1, :]) & (proj_labels[:, 1:, :] > 0)
                    diff_y = (diff_y * mask_y.float()).sum() / (mask_y.sum() + 1e-8)
                    
                    diff_x = torch.norm(z8_aug[:, :, :, 1:] - z8_aug[:, :, :, :-1], dim=1)
                    mask_x = (proj_labels[:, :, 1:] == proj_labels[:, :, :-1]) & (proj_labels[:, :, 1:] > 0)
                    diff_x = (diff_x * mask_x.float()).sum() / (mask_x.sum() + 1e-8)
                    
                    loss_smooth = diff_y + diff_x
                    loss_total = loss_total + 0.5 * loss_smooth

            optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            with torch.no_grad():
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()

            losses.update(loss_total.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            if i % report == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                      f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                      f'IoU {iou.val:.3f} ({iou.avg:.3f})')
                if self.method == 'supcon_vib_evidential' and self._edl_accum:
                    n = max(i + 1, 1)
                    comp = " ".join(f"{k} {v / n:.4f}" for k, v in self._edl_accum.items())
                    print(f"    [evidential] {comp}")

        return acc.avg, iou.avg, losses.avg

