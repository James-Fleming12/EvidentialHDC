import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np

from modules.trainer import Trainer
from common.avgmeter import AverageMeter

# Phase 24.2 casualty list: the classes whose corrupted features collapse and absorb
# into neighbors (Road, Building, Other-ground, Traffic-sign, Bicycle). Phase 25 probe:
# up-weight these anchors in the SupCon loss so the contrastive signal concentrates on
# the classes that need corrupted-view separability the most.
FRAGILE_CLASSES = {2, 7, 13, 14, 15}
FRAGILE_SUPCON_W = 3.0
# Phase 25.7: same-class (clean, extreme-aug) pairs repel above this cosine margin
HARDNEG_MARGIN = 0.5

class GenTrainer(Trainer):
    def __init__(self, ARCH, DATA, datadir, logdir, path=None, method='baseline', cutoff_percent=1.0,
                 fragile_w=None, edl_kl_cap=0.005, edl_w=0.1, edl_kl_selective=True):
        self.method = method
        self.cutoff_percent = cutoff_percent
        self.fragile_w = fragile_w if fragile_w is not None else FRAGILE_SUPCON_W
        self.edl_kl_cap = edl_kl_cap
        self.edl_w = edl_w
        self.edl_kl_selective = edl_kl_selective
        
        # Call super with path=None to prevent it from immediately loading the checkpoint
        super().__init__(ARCH, DATA, datadir, logdir, None)
        
        # If VIB or any SupCon+VIB variant, initialize logvar_head and add to optimizer BEFORE loading checkpoint
        if self.method == 'vib' or self.method.startswith('supcon_vib'):
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
        if self.method == 'supcon_vib_losspred':
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
            self.optimizer.load_state_dict(w_dict['optimizer'])
            self.epoch = w_dict['epoch'] + 1
            if 'scheduler' in w_dict:
                self.scheduler.load_state_dict(w_dict['scheduler'])
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

            # Create augmented view for all methods
            in_vol_aug = self.get_augmented_view(in_vol)

            # SupCon+VIB+SOR: mirror the eval-time SOR pre-filter on both clean and augmented inputs
            if self.method == 'supcon_vib_sor':
                in_vol = self.sor_filter(in_vol)
                in_vol_aug = self.sor_filter(in_vol_aug)

            with torch.amp.autocast('cuda'):
                # Forward pass clean
                if self.ARCH["train"]["aux_loss"]:
                    output, aux_list, z8 = model(in_vol)
                    output_aug, aux_list_aug, z8_aug = model(in_vol_aug)
                else:
                    output, z8 = model(in_vol)
                    output_aug, z8_aug = model(in_vol_aug)

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
                    if len(lbl) > 2000:
                        idx = torch.randperm(len(lbl))[:2000]
                        z_c, z_a, lbl = z_c[idx], z_a[idx], lbl[idx]
                        
                    if self.method == 'supcon_vib_hardneg':
                        # Phase 25.7: the extreme (crosstalk-injected) view, aligned to the
                        # same subsample, for the same-class repulsion term.
                        out_ext = model(self.get_extreme_view(in_vol))
                        z8_ext = out_ext[2] if len(out_ext) == 3 else out_ext[1]
                        z_ext = z8_ext.permute(0, 2, 3, 1)[mask]
                        if len(lbl) > 2000:
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

                    # Phase 25.6 (direct loss prediction): regress the main classifier's
                    # per-point CE on clean + augmented views. The predicted loss is the
                    # gating/uncertainty signal. No OOD labels, no KL, condition-agnostic.
                    if self.method == 'supcon_vib_losspred':
                        m = proj_labels > 0
                        target_c = F.cross_entropy(output, proj_labels, reduction='none')[m]
                        target_a = F.cross_entropy(output_aug, proj_labels, reduction='none')[m]
                        pred_c = F.softplus(self.losspred_head(z8)).permute(0, 2, 3, 1)[m, 0]
                        pred_a = F.softplus(self.losspred_head(z8_aug)).permute(0, 2, 3, 1)[m, 0]
                        if len(target_c) > 0:
                            loss_lp = (F.smooth_l1_loss(pred_c, target_c)
                                       + F.smooth_l1_loss(pred_a, target_a))
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

