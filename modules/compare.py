import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ConformalHDC(nn.Module):
    """
    ConformalHDC: Uncertainty-Aware Hyperdimensional Computing via Conformal Inference.
    
    Ref: Angelopoulos et al., "ConformalHDC: Uncertainty-Aware Hyperdimensional Computing via Conformal Inference" (conformalhdc.pdf).
    
    This module implements the CHDC-discount nonconformity score and uses conformal prediction sets
    to gate (veto) and scale online prototype adaptations during test-time adaptation.
    """
    def __init__(self, feature_extractor, num_classes=19, feature_dim=128, proj_dim=1024, source_prototypes=None, alpha=0.10, lr=0.01, *args, **kwargs):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.alpha = alpha  # Significance level (default 0.10 for 90% coverage)
        self.lr = lr
        
        # Disable bias on semantic output if present
        if hasattr(self.feature_extractor, 'semantic_output') and hasattr(self.feature_extractor.semantic_output, 'bias'):
            if self.feature_extractor.semantic_output.bias is not None:
                self.feature_extractor.semantic_output.bias.data.zero_()
                
        # Initialize source prototypes
        if source_prototypes is not None:
            self.source_prototypes = F.normalize(source_prototypes.clone().float(), dim=1)
        elif hasattr(self.feature_extractor, 'semantic_output'):
            source_weight = self.feature_extractor.semantic_output.weight.data.clone().float()
            self.source_prototypes = F.normalize(source_weight.view(num_classes, feature_dim), dim=1)
        else:
            self.source_prototypes = F.normalize(torch.randn(num_classes, feature_dim), dim=1)
            
        # Running prototypes for online adaptation
        self.prototypes = nn.Parameter(self.source_prototypes.clone(), requires_grad=False)
        self.running_quantile = None
        self.alpha_momentum = 0.90

    def _flatten_features(self, feat):
        """Universal shape handling to ensure 2D feature matrix [N, feature_dim]."""
        if feat.dim() == 4:
            return feat.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        elif feat.dim() == 3:
            return feat.reshape(-1, self.feature_dim)
        return feat

    def compute_conformity_scores(self, h):
        """
        Computes CHDC-discount conformity scores (inverse of nonconformity Eq. 3 in conformalhdc.pdf).
        
        For each point z and class c:
            sim(z, w_c) = (cosine_similarity(z, w_c) + 1) / 2  in [0, 1]
            ratio(z, c) = sim(z, w_c) / sum_y sim(z, w_y)
            conformity V(z, c) = ratio(z, c) * sim(z, w_c)
        """
        device = h.device
        Z = F.normalize(h.float(), dim=1)
        W = F.normalize(self.prototypes.to(device).float(), dim=1)
        
        # Raw cosine similarities mapped to non-negative range [0, 1]
        cos_sim = Z @ W.T
        sim = (cos_sim + 1.0) / 2.0 + 1e-6
        
        # Ratio score (likelihood of class c vs all others)
        sim_sum = sim.sum(dim=1, keepdim=True)
        ratio = sim / sim_sum
        
        # Conformity score (larger = stronger evidence for class c)
        conformity = ratio * sim
        return conformity, cos_sim

    def forward(self, x, xyz=None, *args, **kwargs):
        with torch.no_grad():
            out = self.feature_extractor(x)
            if isinstance(out, tuple):
                if len(out) == 2:
                    base_pred, feat = out
                else:
                    base_pred = out[0]
                    feat = out[-1]
            else:
                feat = out
                base_pred = None
                
            feat_flat = self._flatten_features(feat)
            
            # Compute similarity logits against adapted prototypes
            Z = F.normalize(feat_flat.float(), dim=1)
            W = F.normalize(self.prototypes.to(feat_flat.device).float(), dim=1)
            logits = Z @ W.T
            
        return logits, None, torch.arange(logits.shape[0], device=logits.device), feat_flat

    def inference_update(self, h, predictions=None, xyz=None, *args, **kwargs):
        """
        Online adaptation update gated and scaled by conformal prediction sets.
        """
        if h is None or h.numel() == 0:
            return
            
        device = h.device
        h_flat = self._flatten_features(h)
        h_valid = h_flat[h_flat.sum(dim=1) != 0] if h_flat.dim() == 2 else h_flat
        if h_valid.shape[0] < 10:
            return
            
        conformity, cos_sim = self.compute_conformity_scores(h_valid)
        pred_labels = cos_sim.argmax(dim=1)
        
        # Get conformity scores for predicted labels
        pred_conformity = conformity[torch.arange(h_valid.shape[0], device=device), pred_labels]
        
        # Compute empirical quantile Q_alpha for target coverage (1 - alpha)
        if pred_conformity.numel() == 0:
            return
        batch_quantile = torch.quantile(pred_conformity.float(), self.alpha)
        if self.running_quantile is None:
            self.running_quantile = batch_quantile.item()
        else:
            self.running_quantile = self.alpha_momentum * self.running_quantile + (1.0 - self.alpha_momentum) * batch_quantile.item()
            
        q_thresh = self.running_quantile
        
        # Construct Conformal Prediction Sets C_alpha(z) = { c | V(z, c) >= Q_alpha }
        in_set_mask = (conformity >= q_thresh)
        set_sizes = in_set_mask.sum(dim=1)
        
        # Gating (Veto): Require singleton prediction sets (|C_alpha| == 1) containing the predicted label
        # Points with |C_alpha| > 1 lie in overlapping decision boundaries (ambiguous) and are vetoed.
        # Points with |C_alpha| == 0 are OOD outliers and are vetoed.
        singleton_mask = (set_sizes == 1) & in_set_mask[torch.arange(h_valid.shape[0], device=device), pred_labels]
        
        if not singleton_mask.any():
            return
            
        # Scaling: Scale admitted updates by their conformal margin over threshold
        valid_indices = torch.nonzero(singleton_mask, as_tuple=True)[0]
        valid_feats = F.normalize(h_valid[valid_indices].float(), dim=1)
        valid_preds = pred_labels[valid_indices]
        valid_conf = pred_conformity[valid_indices]
        
        # Scale factor in [0, 1] proportional to certainty above threshold
        scale_factors = torch.clamp((valid_conf - q_thresh) / (1.0 - q_thresh + 1e-6), min=0.1, max=1.0)
        
        # Prototype Momentum Update
        with torch.no_grad():
            self.prototypes.data = self.prototypes.data.to(device)
            for c in range(self.num_classes):
                c_mask = (valid_preds == c)
                if c_mask.sum() > 0:
                    c_feats = valid_feats[c_mask]
                    c_scales = scale_factors[c_mask].unsqueeze(1)
                    weighted_step = (c_feats * c_scales).mean(dim=0)
                    self.prototypes[c].data = F.normalize(self.prototypes[c].data + self.lr * weighted_step, dim=0)

    # Alias methods for interface uniformity
    update = inference_update
    adapt = inference_update

class HyperDUM(nn.Module):
    """
    HyperDUM: Hyperdimensional Uncertainty Weighting Module for Multimodal/Single-modal Perception.
    
    Ref: "Multimodal Perception with Hyperdimensional Uncertainty Weighting" (hyperdum.pdf).
    
    This module implements Channel-wise Projection/Bundling (CPB) and Uncertainty Weighting Modules (Omega)
    to reweight latent features based on similarity dispersion before prototype adaptation.
    """
    def __init__(self, feature_extractor, num_classes=19, feature_dim=128, proj_dim=1024, source_prototypes=None, lr=0.01, gamma=2.0, uncert_thresh=0.70, *args, **kwargs):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.lr = lr
        self.gamma = gamma  # Damping exponent in weighting module Omega(z, u) = (1 - u)^gamma
        self.uncert_thresh = uncert_thresh
        
        if hasattr(self.feature_extractor, 'semantic_output') and hasattr(self.feature_extractor.semantic_output, 'bias'):
            if self.feature_extractor.semantic_output.bias is not None:
                self.feature_extractor.semantic_output.bias.data.zero_()
                
        if source_prototypes is not None:
            self.source_prototypes = F.normalize(source_prototypes.clone().float(), dim=1)
        elif hasattr(self.feature_extractor, 'semantic_output'):
            source_weight = self.feature_extractor.semantic_output.weight.data.clone().float()
            self.source_prototypes = F.normalize(source_weight.view(num_classes, feature_dim), dim=1)
        else:
            self.source_prototypes = F.normalize(torch.randn(num_classes, feature_dim), dim=1)
            
        self.prototypes = nn.Parameter(self.source_prototypes.clone(), requires_grad=False)
        
        # Learnable channel-wise weighting projection (CPB Eq. 6)
        self.channel_weights = nn.Parameter(torch.zeros(1, feature_dim), requires_grad=False)

    def _flatten_features(self, feat):
        """Universal shape handling to ensure 2D feature matrix [N, feature_dim]."""
        if feat.dim() == 4:
            return feat.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        elif feat.dim() == 3:
            return feat.reshape(-1, self.feature_dim)
        return feat

    def compute_uncertainty_and_reweight(self, h):
        """
        Computes similarity uncertainty U_m (Eq. 2 & Eq. 9 in hyperdum.pdf) and applies 
        the Uncertainty Weighting Module Omega(z, u) to output uncertainty-aware features z_hat.
        """
        device = h.device
        Z = F.normalize(h.float(), dim=1)
        W = F.normalize(self.prototypes.to(device).float(), dim=1)
        
        # Similarity calculation against bundled class prototypes
        cos_sim = Z @ W.T
        
        # Similarity uncertainty u: normalized entropy of similarity distribution
        probs = F.softmax(cos_sim / 0.1, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        max_entropy = math.log(self.num_classes)
        u = torch.clamp(entropy / max_entropy, 0.0, 1.0)
        
        # Channel-wise weighting module Omega(z, u) (Section 3.1 & Figure 5)
        # Reweights input features between 0-1 times original value based on uncertainty
        channel_scale = torch.sigmoid(5.0 * self.channel_weights.to(device))
        u_factor = ((1.0 - u) ** self.gamma).unsqueeze(1)
        
        z_hat = Z * channel_scale * u_factor
        return z_hat, u, cos_sim, Z, W

    def forward(self, x, xyz=None, *args, **kwargs):
        with torch.no_grad():
            out = self.feature_extractor(x)
            if isinstance(out, tuple):
                if len(out) == 2:
                    base_pred, feat = out
                else:
                    base_pred = out[0]
                    feat = out[-1]
            else:
                feat = out
                base_pred = None
                
            feat_flat = self._flatten_features(feat)
            
            Z = F.normalize(feat_flat.float(), dim=1)
            W = F.normalize(self.prototypes.to(feat_flat.device).float(), dim=1)
            logits = Z @ W.T
            
        return logits, None, torch.arange(logits.shape[0], device=logits.device), feat_flat

    def inference_update(self, h, predictions=None, xyz=None, *args, **kwargs):
        """
        Online adaptation update using HyperDUM reweighted features z_hat gated by similarity uncertainty.
        Includes dynamic CPB channel-wise uncertainty learning.
        """
        if h is None or h.numel() == 0:
            return
            
        device = h.device
        h_flat = self._flatten_features(h)
        h_valid = h_flat[h_flat.sum(dim=1) != 0] if h_flat.dim() == 2 else h_flat
        if h_valid.shape[0] < 10:
            return
            
        z_hat, u, cos_sim, Z, W = self.compute_uncertainty_and_reweight(h_valid)
        pred_labels = cos_sim.argmax(dim=1)
        
        # Gating: Veto points whose uncertainty exceeds the adaptive threshold
        q_thresh = torch.quantile(u.float(), self.uncert_thresh)
        gate_mask = (u <= q_thresh)
        
        if not gate_mask.any():
            return
            
        valid_indices = torch.nonzero(gate_mask, as_tuple=True)[0]
        valid_z_hat = z_hat[valid_indices]
        valid_preds = pred_labels[valid_indices]
        
        # Dynamic CPB channel weight adaptation: learn feature channel reliability in target domain
        with torch.no_grad():
            self.channel_weights.data = self.channel_weights.data.to(device)
            self.prototypes.data = self.prototypes.data.to(device)
            
            channel_alignment = (Z[valid_indices] * W[valid_preds]).mean(dim=0, keepdim=True)
            self.channel_weights.data = 0.95 * self.channel_weights.data + 0.05 * channel_alignment
            
            # Prototype Adaptation using uncertainty-weighted features z_hat
            for c in range(self.num_classes):
                c_mask = (valid_preds == c)
                if c_mask.sum() > 0:
                    c_feats = valid_z_hat[c_mask]
                    weighted_step = c_feats.mean(dim=0)
                    self.prototypes[c].data = F.normalize(self.prototypes[c].data + self.lr * weighted_step, dim=0)

    # Alias methods for interface uniformity
    update = inference_update
    adapt = inference_update

logger = logging.getLogger("baselines")

BASELINE_METHODS = ("d3ctta", "conformalhdc", "hyperdum")

class _FeatOnly(nn.Module):
    """Forces the backbone to return 128D features rather than logits."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        # expose BN modules so D3CTTA's get_last_bn_stats() still works
        self.modules_src = net
        if hasattr(net, 'semantic_output'):
            self.semantic_output = net.semantic_output   # lets D3CTTA find [17,128] head weights

    def forward(self, x, *a, **kw):
        with torch.amp.autocast('cuda', enabled=True):
            return self.net(x, only_feat=True).float()

    def modules(self):
        return self.net.modules()

def get_adapter(model, name, num_classes, device, feature_dim=128):
    """Lazily build (and cache on `model`) the baseline adapter."""
    if getattr(model, "_baseline_adapter_name", None) == name and \
            getattr(model, "_baseline_adapter", None) is not None:
        return model._baseline_adapter

    if getattr(model, "class_latent_means", None) is None:
        raise ValueError(
            "baselines need model.class_latent_means ([num_classes, 128]) as source "
            "prototypes. The HDC classify weights are [num_classes, 10000] and are the "
            "wrong space -- seeding with them is what made the old sync silently break.")
    src_proto = model.class_latent_means.detach().clone().to(device).float()
    assert src_proto.shape == (num_classes, feature_dim), \
        f"source prototypes {tuple(src_proto.shape)} != ({num_classes}, {feature_dim})"

    backbone = _FeatOnly(model.net)

    if name == "d3ctta":
        from modules.D3CTTA import D3CTTA
        adapter = D3CTTA(backbone, num_classes=num_classes, feature_dim=feature_dim,
                         source_prototypes=src_proto)
    elif name == "conformalhdc":
        adapter = ConformalHDC(backbone, num_classes=num_classes, feature_dim=feature_dim,
                               source_prototypes=src_proto)
    elif name == "hyperdum":
        adapter = HyperDUM(backbone, num_classes=num_classes, feature_dim=feature_dim,
                           source_prototypes=src_proto)
    else:
        raise ValueError(f"unknown baseline '{name}'; known: {BASELINE_METHODS}")

    adapter = adapter.to(device)
    model._baseline_adapter = adapter
    model._baseline_adapter_name = name
    logger.info(f"[baselines] built {name}: feature_dim={feature_dim}, "
                f"source_prototypes={tuple(src_proto.shape)}")
    return adapter

@torch.no_grad()
def baseline_forward(model, name, proj_in, proj_xyz, num_classes, device):
    """Run the baseline's OWN forward. Returns (logits, state).

    `state` carries whatever the update step needs (e.g. D3CTTA's projected h).
    """
    adapter = get_adapter(model, name, num_classes, device)

    out = adapter(proj_in, xyz=proj_xyz)
    if isinstance(out, (tuple, list)):
        logits = out[0]
        h = out[-1] if len(out) >= 4 else None
    else:
        logits, h = out, None

    if logits.dim() == 4:                       # [B,C,H,W] -> [N,C]
        logits = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
    elif logits.dim() == 3:
        logits = logits.reshape(-1, num_classes)

    return logits, {"h": h, "adapter": adapter}

@torch.no_grad()
def baseline_update(state, predictions, proj_xyz):
    """Run the baseline's own online adaptation."""
    adapter = state["adapter"]
    h = state["h"]
    if h is None:
        logger.warning("[baselines] adapter forward returned no projected features; "
                       "skipping update this frame")
        return
    adapter.inference_update(h, predictions, proj_xyz)

def reset(model):
    for a in ("_baseline_adapter", "_baseline_adapter_name"):
        if hasattr(model, a):
            try:
                delattr(model, a)
            except AttributeError:
                setattr(model, a, None)