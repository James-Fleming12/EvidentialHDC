from torchhd import functional
from torchhd import embeddings

import numpy as np
import copy
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_log = logging.getLogger("HDC_utils")

# ==========================================================================
# GATING CONSTANTS  (see docs/method_details.md section 8)
# ==========================================================================
# Dirichlet uncertainty is u = C/E with E >= C, so u is bounded in (0, 1].
# That bound matters: with u_th=0.5, u_coef=1.5 the epistemic decay only ever
# falls to exp(-0.75) = 0.472 at u = 1. The previously shipped 'epistemic' mode
# used exp(-2.0*relu(u-0.1)), which reaches exp(-1.8) = 0.165. The old
# no_dual_gating ablation therefore compared two gates in different REGIMES,
# not merely at different offsets. Both presets are exposed so this can be swept
# rather than assumed.
PRESETS = {
    "soft":  {"u_th": 0.5, "u_coef": 1.5},   # what soft_dual_weight used
    "sharp": {"u_th": 0.1, "u_coef": 2.0},   # what the old 'epistemic' used
}

GATE_CFG = {
    "u_th": 0.5,          # epistemic uncertainty threshold
    "u_coef": 1.5,        # epistemic decay rate
    "z_th": 0.5,          # geometric z-score threshold
    "z_coef": 1.0,        # geometric decay rate
    # rescue_gate only
    "u_veto_th": 0.5,     # epistemic VETOES when u > this. Keyed on the raw
                          # uncertainty, NOT on the decay: under the "soft"
                          # preset, decay < 0.5 requires u > 0.962, which for
                          # u in (0,1] essentially never fires.
    "rescue_z_th": 0.0,   # rescue keys on z <= this (tight => high precision)
    "rescue_min": 0.60,   # minimum rescue weight worth admitting
    "rescue_scale": 1.0,  # global trust discount on rescued points
}

def fuse_uncertainties(epistemic, geometric, method="soft_dual_weight", cfg=None):
    """Corrected uncertainty fusion.

    Both inputs are 'higher = worse' scores: `epistemic` is the Dirichlet
    uncertainty u = C/E in (0,1]; `geometric` is the distance z-score
    (dist - mean_c)/std_c, unbounded, higher = farther from the class centroid.

    Every method is built from the SAME two factors:

        u_dec = exp(-u_coef * relu(u - u_th))
        z_dec = exp(-z_coef * relu(z - z_th))

    so ablating the geometric term isolates exactly one thing. Previously
    'epistemic' and 'soft_dual_weight' used different thresholds AND different
    coefficients, so 'no dual gating' changed three things at once.
    """
    c = dict(GATE_CFG)
    if cfg:
        c.update(cfg)

    if geometric is None:
        geometric = torch.zeros_like(epistemic)

    u_dec = torch.exp(-c["u_coef"] * torch.relu(epistemic - c["u_th"]))
    z_dec = torch.exp(-c["z_coef"] * torch.relu(geometric - c["z_th"]))

    if method == "uniform":
        return torch.ones_like(epistemic)
    if method == "epistemic":
        return u_dec
    if method == "geometric":
        return z_dec
    if method == "soft_dual_weight":
        # exp(-a)*exp(-b) == exp(-a-b): identical algebra to the shipped
        # version, but the epistemic factor now matches gate_mode='epistemic'.
        return u_dec * z_dec
    if method == "and_gate":
        return torch.minimum(u_dec, z_dec)
    if method == "or_gate":
        return torch.maximum(u_dec, z_dec)
    if method == "ellipsoid_gate":
        ue = torch.relu(epistemic - c["u_th"])
        ze = torch.relu(geometric - c["z_th"])
        return torch.exp(-(c["u_coef"] * ue ** 2 + c["z_coef"] * 0.5 * ze ** 2))
    if method == "rescue_gate":
        # Asymmetric cascade. Epistemic is primary; where it vetoes, allow a
        # BOUNDED geometric rescue keyed on LOW z (close to the centroid).
        #
        # The previously shipped version was
        #     where((u < 0.5) & (z >= 0.8), z, u)
        # which (i) rescued points FAR from the centroid -- inverted sign -- and
        # (ii) returned the raw unbounded z-score as a weight while the
        # else-branch returned an uncertainty in [0,1]. That is why it over-fired
        # at 83-84%. Hypothesis A was never actually tested.
        rescue = torch.exp(-c["z_coef"] * torch.relu(geometric - c["rescue_z_th"]))
        rescue = torch.where(rescue >= c["rescue_min"],
                             rescue * c["rescue_scale"],
                             torch.zeros_like(rescue))
        epi_vetoes = epistemic > c["u_veto_th"]
        return torch.where(epi_vetoes, torch.maximum(u_dec, rescue), u_dec)

    raise ValueError(f"unknown gate method '{method}'. known: uniform, epistemic, "
                     f"geometric, soft_dual_weight, and_gate, or_gate, "
                     f"ellipsoid_gate, rescue_gate")

class GainController:
    """Label-free domain-gap estimate used to scale the global learning rate.

    Motivation: across the 8-corruption panel,
        corr(frozen mIoU, adaptation gain) = -0.48
    i.e. adaptation helps where the frozen model is weak (crosstalk +3.77) and
    hurts where it is already strong (wet_ground -5.11). The pipeline had no
    notion of "how much adaptation does this domain need".

        gap  = EMA of mean epistemic uncertainty over the stream
        gain = clip((gap - gap_lo) / (gap_hi - gap_lo), min_gain, 1.0)

    gap_lo should be the mean uncertainty the frozen model shows on CLEAN source
    data. Calibrate once with `ablation_kitti-c.py --calibrate_gap`.
    """

    def __init__(self, gap_lo=0.35, gap_hi=0.75, beta=0.99, min_gain=0.0):
        self.gap_lo = float(gap_lo)
        self.gap_hi = float(gap_hi)
        self.beta = float(beta)
        self.min_gain = float(min_gain)
        self.gap = None
        self.n = 0
        self._log = []

    def update(self, uncertainty):
        if uncertainty is None or uncertainty.numel() == 0:
            return self.gain()
        m = float(uncertainty.mean().item())
        self.gap = m if self.gap is None else self.beta * self.gap + (1 - self.beta) * m
        self.n += 1
        g = self.gain()
        self._log.append(g)
        return g

    def gain(self):
        if self.gap is None:
            return 1.0
        denom = max(1e-6, self.gap_hi - self.gap_lo)
        return float(min(1.0, max(self.min_gain, (self.gap - self.gap_lo) / denom)))
        
    def gap_value(self):
        return self.gap if self.gap is not None else 0.0

    def summary(self):
        if not self._log:
            return "GainController: never invoked"
        return (f"GainController: frames={self.n} final_gap={self.gap:.4f} "
                f"mean_gain={sum(self._log)/len(self._log):.4f} "
                f"final_gain={self.gain():.4f} "
                f"(gap_lo={self.gap_lo}, gap_hi={self.gap_hi})")

# Call counters so the runner can detect whether evaluate_and_adapt actually
# routes through these methods, or still uses its own inline gating.
CALL_COUNTERS = {"fuse": 0, "update": 0, "confidence": 0}

def reset_counters():
    for k in CALL_COUNTERS:
        CALL_COUNTERS[k] = 0

def counters():
    return dict(CALL_COUNTERS)

class Model(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
        super(Model, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.hd_dim = 10000
        self.temperature = 0.01
        self.flatten = torch.nn.Flatten()
        self.input_dim = 128
        self.ARCH = ARCH

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.net = HarDNet(self.num_classes, self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"])

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.net = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

        w_dict = torch.load(modeldir + "/SENet_valid_best",
                            map_location=lambda storage, loc: storage)
        self.net.load_state_dict(w_dict['state_dict'], strict=True)
        self.net.eval()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.gpu = True
            self.net.cuda()

        self.hd_encoder = hd_encoder
        if self.hd_encoder == 'rp':
            self.projection = embeddings.Projection(self.input_dim, self.hd_dim)
        elif self.hd_encoder == 'idlevel':
            self.value = embeddings.Level(num_levels, self.hd_dim, randomness=randomness)
            self.position = embeddings.Random(self.input_dim, self.hd_dim)
        elif self.hd_encoder == 'nonlinear':
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)
        else:
            self.hd_dim = self.input_dim

        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)
        self.classify.weight.data.fill_(0.0)
        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        with torch.cuda.amp.autocast(enabled=True):
            x = self.net(x, True)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 128)

        if PERCENTAGE is not None:
            num_samples = int(x.shape[0] * PERCENTAGE)
            if is_wrong is not None:
                wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
                if wrong_indices.numel() >= num_samples:
                    selected_indices = wrong_indices[torch.randperm(wrong_indices.shape[0], device=x.device)[:num_samples]]
                    is_wrong[selected_indices] = False
                else:
                    non_wrong_indices = torch.nonzero(~is_wrong, as_tuple=False).squeeze()
                    remaining = num_samples - wrong_indices.numel()
                    fill_indices = non_wrong_indices[torch.randperm(non_wrong_indices.shape[0], device=x.device)[:remaining]]
                    selected_indices = torch.cat([wrong_indices, fill_indices], dim=0)
                    is_wrong[selected_indices] = False
            else:
                selected_indices = torch.randperm(x.shape[0], device=x.device)[:num_samples]

            selected_indices, _ = selected_indices.sort()
            x = x[selected_indices]
            assert x.shape[0] == num_samples, f"Expected {num_samples} samples, got {x.shape[0]}"
        else:
            selected_indices = torch.arange(x.shape[0], device=x.device)

        sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device, dtype=x.dtype)

        if self.hd_encoder == 'rp':
            if x.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(x.dtype).to(self.device)
            sample_hv[:, mask] = self.projection(x)[:, mask]
        elif self.hd_encoder == 'idlevel':
            tmp_hv = functional.bind(self.position.weight[:, mask], self.value(x)[:, :, mask])
            sample_hv[:, mask] = functional.multiset(tmp_hv)
        elif self.hd_encoder == 'nonlinear':
            sample_hv[:, mask] = self.nonlinear_projection(x)[:, mask]
        else:
            return x

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        return sample_hv, selected_indices, is_wrong

    def forward(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)
        enc, indices, is_wrong_left = self.encode(x, mask, PERCENTAGE, is_wrong)
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc, dim=1))
        return logits, F.normalize(enc, dim=1), indices, is_wrong_left

    def get_predictions(self, enc):
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        return self.classify(F.normalize(enc, dim=1))

    def extract_class_hv(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)
        if self.method == 'LifeHD':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        else:
            class_hv = self.classify.weight[:, mask]
        return class_hv.detach().cpu().numpy()

    def extract_pair_simil(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)
        if self.method == 'LifeHD' or self.method == 'LifeHDsemi':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        elif self.method == 'BasicHD':
            class_hv = self.classify.weight[:, mask]
        else:
            raise ValueError('method not supported: {}'.format(self.method))
        pair_simil = class_hv @ class_hv.T
        if self.method == 'LifeHDsemi':
            pair_simil[:self.num_classes, :self.num_classes] = torch.eye(self.num_classes)
        return pair_simil.detach().cpu().numpy(), class_hv.detach().cpu().numpy()

def set_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
    return Model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)

class DualGateModel(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes,
                 device, gauss_rp=True, use_adaptor=True):
        super(DualGateModel, self).__init__()

        self.device = device
        self.use_adaptor = use_adaptor
        self.num_classes = num_classes
        self.hd_dim = 10000
        self.temperature = 0.01
        self.flatten = torch.nn.Flatten()
        self.input_dim = 128
        self.ARCH = ARCH

        # per-model gate configuration; None => module-level GATE_CFG
        self.gate_cfg = None
        self.gain_controller = None

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.net = HarDNet(self.num_classes, self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"],
                                     use_adaptor=self.use_adaptor)

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"])
                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "pointpillar":
                from modules.HDC_cl import PointPillarEncoder

                class _PointPillarEncoder4D(PointPillarEncoder):
                    def forward(self, batch, only_feat=False):
                        return super().forward(batch).unsqueeze(-1).unsqueeze(-1)

                self.net = _PointPillarEncoder4D(
                    in_channels=self.ARCH["train"].get("pointpillar_in_channels", 4),
                    bev_shape=tuple(self.ARCH["train"].get("pointpillar_bev_shape", [512, 512])),
                )

        if self.ARCH["train"]["pipeline"] != "pointpillar":
            w_dict = torch.load(modeldir + "/SENet_valid_best",
                                map_location=lambda storage, loc: storage)
            state_dict = w_dict['state_dict']
            model_state = self.net.state_dict()
            for k in list(state_dict.keys()):
                if k in model_state and state_dict[k].shape != model_state[k].shape:
                    del state_dict[k]
            self.net.load_state_dict(state_dict, strict=False)
            self.net.eval()
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.gpu = True
                self.net.cuda()

        self.hd_encoder = hd_encoder
        if self.hd_encoder == 'rp':
            torch_rng_state = torch.get_rng_state()
            numpy_rng_state = np.random.get_state()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()

            torch.manual_seed(42)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)

            if not gauss_rp:
                self.projection = nn.Linear(self.input_dim, self.hd_dim, bias=False)
                with torch.no_grad():
                    gaussian_matrix = torch.randn(self.hd_dim, self.input_dim)
                    self.projection.weight.copy_(gaussian_matrix / np.sqrt(self.input_dim))
            else:
                self.projection = nn.Linear(self.input_dim, self.hd_dim, bias=False)
                with torch.no_grad():
                    gaussian_matrix = torch.randn(self.hd_dim, self.input_dim)
                    q, _ = torch.linalg.qr(gaussian_matrix)
                    self.projection.weight.copy_(q * torch.sqrt(torch.tensor(self.hd_dim)))

            torch.set_rng_state(torch_rng_state)
            np.random.set_state(numpy_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)

        elif self.hd_encoder == 'idlevel':
            self.value = embeddings.Level(num_levels, self.hd_dim, randomness=randomness)
            self.position = embeddings.Random(self.input_dim, self.hd_dim)
        elif self.hd_encoder == 'nonlinear':
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)
        else:
            self.hd_dim = self.input_dim

        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)
        self.classify.weight.data.fill_(0.0)
        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)
        self.gauss_rp = gauss_rp

        self.register_buffer('proto_momentum', torch.zeros_like(self.classify.weight.data))

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None, chunk_idx=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        with torch.amp.autocast('cuda', enabled=True):
            x = self.net(x, only_feat=True)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 128)

        if chunk_idx is not None:
            start, end = chunk_idx
            x = x[start:end]

        if PERCENTAGE is not None:
            wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
            num_samples = int(x.shape[0] * PERCENTAGE)
            if wrong_indices.numel() >= num_samples:
                selected_indices = wrong_indices[torch.randperm(wrong_indices.shape[0], device=x.device)[:num_samples]]
                is_wrong[selected_indices] = False
            else:
                non_wrong_indices = torch.nonzero(~is_wrong, as_tuple=False).squeeze()
                remaining = num_samples - wrong_indices.numel()
                fill_indices = non_wrong_indices[torch.randperm(non_wrong_indices.shape[0], device=x.device)[:remaining]]
                selected_indices = torch.cat([wrong_indices, fill_indices], dim=0)
                is_wrong[selected_indices] = False
            selected_indices, _ = selected_indices.sort()
            x = x[selected_indices]
            assert x.shape[0] == num_samples, f"Expected {num_samples} samples, got {x.shape[0]}"
        else:
            selected_indices = torch.arange(x.shape[0], device=x.device)

        sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device, dtype=x.dtype)

        if self.hd_encoder == 'rp':
            if x.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(x.dtype).to(self.device)
            sample_hv[:, mask] = self.projection(x)[:, mask]
        elif self.hd_encoder == 'idlevel':
            tmp_hv = functional.bind(self.position.weight[:, mask], self.value(x)[:, :, mask])
            sample_hv[:, mask] = functional.multiset(tmp_hv)
        elif self.hd_encoder == 'nonlinear':
            sample_hv[:, mask] = self.nonlinear_projection(x)[:, mask]
        else:
            return x

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        return sample_hv, selected_indices, is_wrong

    def forward(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)
        enc, indices, is_wrong_left = self.encode(x, mask, PERCENTAGE, is_wrong)
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc, dim=1))
        return logits, F.normalize(enc, dim=1), indices, is_wrong_left

    def get_predictions(self, enc):
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        return self.classify(F.normalize(enc, dim=1))

    def extract_class_hv(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)
        if self.method == 'LifeHD':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        else:
            class_hv = self.classify.weight[:, mask]
        return class_hv.detach().cpu().numpy()

    def extract_pair_simil(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)
        if self.method == 'LifeHD' or self.method == 'LifeHDsemi':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        elif self.method == 'BasicHD':
            class_hv = self.classify.weight[:, mask]
        else:
            raise ValueError('method not supported: {}'.format(self.method))
        pair_simil = class_hv @ class_hv.T
        if self.method == 'LifeHDsemi':
            pair_simil[:self.num_classes, :self.num_classes] = torch.eye(self.num_classes)
        return pair_simil.detach().cpu().numpy(), class_hv.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_confidence(self, enc, preds=None, method='soft_dual_weight', uncertainty=None,
                       z_score=None, logits=None, active_mu_cos=None, active_sigma_cos=None,
                       dynamic_geom=True, view_preds=None, view_var=None,
                       base_conf=None, **kwargs):
        """Compute the gating weight. Returns (weights, uncertainty, z_score).

        `base_conf` (NEW, optional): the softmax confidence term
        softmax(cos_sims * 100).max(dim=1). The inline path in unsup_kitti-c.py
        computed update_weights = base_conf * decay; this method previously
        returned the decay ALONE, so the two paths weighted points differently
        even under identical gating. Pass base_conf to reproduce the inline
        semantics; leave it None to keep the decay-only behaviour.
        """
        CALL_COUNTERS["confidence"] += 1
        if uncertainty is None:
            uncertainty = self._get_epistemic_uncertainty(
                enc, logits=logits, active_mu_cos=active_mu_cos,
                active_sigma_cos=active_sigma_cos)
        if z_score is None and preds is not None:
            z_score = self._get_geometric_confidence(enc, preds, dynamic_geom=dynamic_geom)
        elif z_score is None:
            z_score = torch.zeros_like(uncertainty)

        base_weights = self._fuse_uncertainties(uncertainty, None, z_score, method=method)
        if base_conf is not None:
            base_weights = base_weights * base_conf
        return base_weights, uncertainty, z_score

    @torch.no_grad()
    def online_update(self, enc, preds, update_weights, update_method='bm_ic4',
                      ic_method='ic4', uncertainty=None, update_lr=0.01,
                      normalize_weights=False, view_preds=None, fire_th=0.0, **kwargs):
        """Prototype momentum update.

        CHANGES vs the previously shipped version:
          * `fire_th` (NEW). The old code used `fired_mask = update_weights > 0`.
            Fusion weights come from exp(...), which is strictly positive, so
            that mask was ALWAYS all-True: there was no veto at all, every
            background point contributed, and the reported firing rate was
            necessarily ~100%. Set fire_th > 0 for a real threshold.
          * `update_lr` default changed 0.005 -> 0.01 to match the inline path in
            unsup_kitti-c.py. If you were relying on the old default, every
            previous run through this method used HALF the intended step size.
          * gain control: if self.gain_controller is set, the effective learning
            rate is scaled by the label-free domain-gap estimate.
        """
        CALL_COUNTERS["update"] += 1

        if ic_method == 'ic4' and uncertainty is not None:
            update_weights = update_weights * uncertainty

        if self.gain_controller is not None:
            update_lr = update_lr * self.gain_controller.update(uncertainty)

        fired_mask = update_weights > fire_th
        if not fired_mask.any():
            return fired_mask

        valid_enc = enc[fired_mask]
        labels_fired = preds[fired_mask]
        weights_fired = update_weights[fired_mask]

        sums_c = torch.zeros((self.num_classes, valid_enc.shape[1]),
                             dtype=valid_enc.dtype, device=self.device)
        weighted_enc = (valid_enc * weights_fired.unsqueeze(1)).to(valid_enc.dtype)
        sums_c.index_add_(0, labels_fired, weighted_enc)

        counts_c = torch.bincount(labels_fired, minlength=self.num_classes).float()
        weights_sum_c = torch.bincount(labels_fired, weights=weights_fired.float(),
                                       minlength=self.num_classes)

        valid_c = (counts_c > 0)
        if valid_c.any():
            c_update_norm = F.normalize(sums_c, p=2, dim=1)
            step_mags = update_lr * (weights_sum_c / torch.clamp(counts_c, min=1.0))
            update_delta = torch.where(
                valid_c.unsqueeze(1),
                step_mags.unsqueeze(1) * c_update_norm.to(self.classify.weight.dtype),
                torch.zeros_like(self.classify.weight))
            self.classify.weight.data += update_delta
            if normalize_weights:
                self.classify.weight.data = F.normalize(self.classify.weight.data, p=2, dim=1)

            if not hasattr(self, 'class_update_counts'):
                self.class_update_counts = torch.zeros(self.num_classes, device=self.device)
            self.class_update_counts += valid_c.long()
            if not hasattr(self, '_update_magnitude_log'):
                self._update_magnitude_log = []
            self._update_magnitude_log.extend(step_mags[valid_c].detach().cpu().tolist())

        return fired_mask

    @torch.no_grad()
    def _get_epistemic_uncertainty(self, enc, logits=None, active_mu_cos=None,
                                   active_sigma_cos=None):
        """Network uncertainty: Dirichlet evidence decay, u = C/E in (0, 1]."""
        if active_mu_cos is None:
            active_mu_cos = getattr(self, 'source_mu_cos', None)
        if active_sigma_cos is None:
            active_sigma_cos = getattr(self, 'source_sigma_cos', None)

        if active_mu_cos is not None and active_sigma_cos is not None:
            if logits is None:
                norm_enc = F.normalize(enc.float(), dim=1)
                norm_weights = F.normalize(self.classify.weight.float(), dim=1)
                logits = F.linear(norm_enc, norm_weights)
            z = (logits - active_mu_cos) / (active_sigma_cos + 1e-8)
            ev = F.softplus(5.0 * z)
            S_val = torch.sum(ev + 1.0, dim=1)
            return self.num_classes / S_val
        elif logits is not None:
            probs = torch.softmax(logits, dim=1)
            conf, _ = torch.max(probs, dim=1)
            return conf
        return torch.ones(enc.shape[0], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _get_spatial_consistency(self, enc, preds, proj_xyz=None, **kwargs):
        return torch.ones_like(preds, dtype=torch.bool)

    @torch.no_grad()
    def _get_geometric_confidence(self, enc, preds, dynamic_geom=True):
        """Geometric density: isotropic Euclidean distance z-score in 128D.

        NOTE ON NAMING: this is isotropic Euclidean distance with one scalar
        scale per class, not a Mahalanobis distance -- there is no covariance
        anywhere. Papers/tables should say "isotropic Euclidean z-score density".
        """
        if (hasattr(self, 'class_latent_means') and self.class_latent_means is not None
                and hasattr(self, 'source_density_std') and self.source_density_std is not None):
            if not hasattr(self, 'source_density_mean') or self.source_density_mean is None:
                raise ValueError(
                    "source_density_mean is missing or None. A stale checkpoint or cache here "
                    "silently restores the uncentred kernel: in 128D, distances concentrate at "
                    "~16 std from zero, so exp(-d^2/2s^2) ~ exp(-128) rejects 100% of even clean "
                    "in-distribution points.")
            pred_means = self.class_latent_means[preds]
            dist = torch.norm(enc.float() - pred_means.float(), p=2, dim=1)

            if dynamic_geom:
                if not hasattr(self, 'running_density_mean') or self.running_density_mean is None:
                    self.running_density_mean = self.source_density_mean.clone().to(self.device)
                if not hasattr(self, 'running_density_std') or self.running_density_std is None:
                    self.running_density_std = self.source_density_std.clone().to(self.device)
                self.running_density_mean = self.running_density_mean.to(self.device)
                self.running_density_std = self.running_density_std.to(self.device)

                counts = torch.bincount(preds, minlength=self.num_classes).float()
                sum_dist = torch.bincount(preds, weights=dist, minlength=self.num_classes)
                sum_sq_dist = torch.bincount(preds, weights=dist ** 2, minlength=self.num_classes)

                valid_c = (counts > 1)
                batch_means = torch.where(valid_c, sum_dist / torch.clamp(counts, min=1.0),
                                          torch.zeros_like(sum_dist))
                batch_vars = torch.where(
                    valid_c,
                    (sum_sq_dist - sum_dist ** 2 / torch.clamp(counts, min=1.0))
                    / torch.clamp(counts - 1.0, min=1.0),
                    torch.zeros_like(sum_sq_dist))
                batch_stds = torch.sqrt(torch.clamp(batch_vars, min=0.0))

                self.running_density_mean = torch.where(
                    valid_c, 0.95 * self.running_density_mean + 0.05 * batch_means,
                    self.running_density_mean)
                valid_std = valid_c & (batch_stds > 0)
                self.running_density_std = torch.where(
                    valid_std, 0.95 * self.running_density_std + 0.05 * batch_stds,
                    self.running_density_std)

                mean_c = self.running_density_mean[preds]
                std_c = self.running_density_std[preds] + 1e-8
            else:
                self.source_density_mean = self.source_density_mean.to(self.device)
                self.source_density_std = self.source_density_std.to(self.device)
                mean_c = self.source_density_mean[preds]
                std_c = self.source_density_std[preds] + 1e-8

            return (dist - mean_c) / std_c
        raise ValueError("Geometric confidence requires class_latent_means and "
                         "source_density_std from source pretraining.")

    @torch.no_grad()
    def _fuse_uncertainties(self, epistemic, consistency, geometric,
                            method='soft_dual_weight'):
        CALL_COUNTERS["fuse"] += 1
        return fuse_uncertainties(epistemic, geometric, method=method,
                                  cfg=getattr(self, 'gate_cfg', None))

def set_uq_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes,
                 device, subcluster_type='bipolar'):
    return DualGateModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)

UQModel = DualGateModel

class MV_TTAModel(DualGateModel):
    """Multi-view TTA variant: spatial augmentation consensus (veto_disagree)
    and cross-view softmax variance gating (view_var_gate)."""

    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes,
                 device, gauss_rp=True, use_adaptor=True, mv_tta='veto_disagree'):
        super(MV_TTAModel, self).__init__(ARCH, modeldir, hd_encoder, num_levels,
                                          randomness, num_classes, device,
                                          gauss_rp=gauss_rp, use_adaptor=use_adaptor)
        self.mv_tta = mv_tta

    @torch.no_grad()
    def _get_spatial_consistency(self, enc, preds, proj_xyz=None, view_preds=None,
                                 view_var=None, **kwargs):
        if view_preds is not None and len(view_preds) >= 2:
            pred_m1, pred_m2 = view_preds[0], view_preds[1]
            view_disagreement = (preds != pred_m1) | (preds != pred_m2)
            return ~view_disagreement
        return super()._get_spatial_consistency(enc, preds, proj_xyz=proj_xyz, **kwargs)

    @torch.no_grad()
    def get_confidence(self, enc, preds=None, method='soft_dual_weight', uncertainty=None,
                       z_score=None, view_preds=None, view_var=None, **kwargs):
        base_weights, uncertainty, z_score = super().get_confidence(
            enc, preds=preds, method=method, uncertainty=uncertainty, z_score=z_score,
            view_preds=view_preds, view_var=view_var, **kwargs)

        if method == 'view_var_gate' and view_var is not None:
            cfg = dict(GATE_CFG)
            if getattr(self, 'gate_cfg', None):
                cfg.update(self.gate_cfg)
            view_var_decay = torch.exp(-2.0 * torch.relu(view_var - 0.05))
            u_dec = torch.exp(-cfg["u_coef"] * torch.relu(uncertainty - cfg["u_th"]))
            base_weights = base_weights * torch.min(u_dec, view_var_decay)

        if self.mv_tta == 'veto_disagree' and view_preds is not None:
            agree_mask = self._get_spatial_consistency(enc, preds, view_preds=view_preds)
            base_weights = base_weights * agree_mask.float()

        return base_weights, uncertainty, z_score

    @torch.no_grad()
    def online_update(self, enc, preds, update_weights, update_method='bm_ic4',
                      ic_method='ic4', uncertainty=None, update_lr=0.01,
                      normalize_weights=False, view_preds=None, fire_th=0.0, **kwargs):
        if self.mv_tta == 'veto_disagree' and view_preds is not None:
            agree_mask = self._get_spatial_consistency(enc, preds, view_preds=view_preds)
            update_weights = update_weights * agree_mask.float()
        return super().online_update(enc, preds, update_weights, update_method=update_method,
                                     ic_method=ic_method, uncertainty=uncertainty,
                                     update_lr=update_lr, normalize_weights=normalize_weights,
                                     view_preds=view_preds, fire_th=fire_th, **kwargs)

MVTTAModel = MV_TTAModel

def set_dual_gate_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes,
                        device, subcluster_type='bipolar'):
    return DualGateModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)

def set_mv_tta_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes,
                     device, mv_tta='veto_disagree'):
    return MV_TTAModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes,
                       device, mv_tta=mv_tta)