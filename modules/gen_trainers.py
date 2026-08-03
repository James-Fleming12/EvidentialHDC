import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np

from modules.trainer import Trainer
from common.avgmeter import AverageMeter

class GenTrainer(Trainer):
    def __init__(self, ARCH, DATA, datadir, logdir, path=None, method='baseline', cutoff_percent=1.0):
        self.method = method
        self.cutoff_percent = cutoff_percent
        
        # Call super with path=None to prevent it from immediately loading the checkpoint
        super().__init__(ARCH, DATA, datadir, logdir, None)
        
        # If VIB or any SupCon+VIB variant, initialize logvar_head and add to optimizer BEFORE loading checkpoint
        if self.method == 'vib' or self.method.startswith('supcon_vib'):
            self.logvar_head = nn.Conv2d(128, 128, kernel_size=1).to(self.device)
            # Add to optimizer so it has 2 param groups (matching the saved checkpoint)
            self.optimizer.add_param_group({'params': self.logvar_head.parameters()})
        else:
            self.logvar_head = None
            
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
            
        return out

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, color_fn, report=10, show_scans=False):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()

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
                        loss_supcon = -torch.log(pos_sum / (all_sum + 1e-8)).mean()
                        
                    loss_total = loss_sem + 0.01 * loss_kl + 0.1 * loss_supcon

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

        return acc.avg, iou.avg, losses.avg

