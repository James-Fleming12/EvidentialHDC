import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np

from modules.trainer import Trainer
from common.avgmeter import AverageMeter

class GenTrainer(Trainer):
    def __init__(self, ARCH, DATA, datadir, logdir, path=None, method='baseline'):
        super().__init__(ARCH, DATA, datadir, logdir, path)
        self.method = method
        
        # If VIB, we need an extra projection to get logvar
        if self.method == 'vib':
            # HarDNet / ResNet z8 bottleneck is usually 128 channels before final classification
            # Let's dynamically add a variance head
            self.logvar_head = nn.Conv2d(128, 128, kernel_size=1).to(self.device)
            # Add to optimizer
            self.optimizer.add_param_group({'params': self.logvar_head.parameters()})

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

    def get_augmented_view(self, in_vol):
        # Compose dropout and jitter
        out = self.beam_drop(in_vol)
        out = self.z_jitter(out)
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

        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(train_loader), total=int(len(train_loader)*0.1)):
            if i > len(train_loader) * 0.1: 
                break

            if self.gpu:
                in_vol, proj_labels = in_vol.cuda(), proj_labels.cuda().long()

            # Create augmented view for all methods
            in_vol_aug = self.get_augmented_view(in_vol)

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
                        tau = 0.1
                        sim_matrix = torch.matmul(z_c, z_a.T) / tau
                        lbl_matrix = lbl.unsqueeze(0) == lbl.unsqueeze(1)
                        
                        max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
                        exp_sim = torch.exp(sim_matrix - max_sim.detach())
                        pos_sum = (exp_sim * lbl_matrix).sum(dim=1)
                        all_sum = exp_sim.sum(dim=1)
                        loss_supcon = -torch.log(pos_sum / (all_sum + 1e-8)).mean()
                        
                        loss_total = loss_total + 0.1 * loss_supcon

                elif self.method == 'vib':
                    # Variational Information Bottleneck
                    mu = z8_aug
                    logvar = self.logvar_head(z8_aug)
                    
                    # KL Divergence to N(0, I)
                    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                    
                    # We sample for the classification pass
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z_sampled = mu + eps * std
                    
                    # Route through the classification head to enforce the bottleneck
                    if hasattr(model, 'module'):
                        logits_sampled = model.module.semantic_output(z_sampled)
                    else:
                        logits_sampled = model.semantic_output(z_sampled)
                    pred_sampled = F.softmax(logits_sampled, dim=1)
                    
                    loss_ce_aug = criterion(torch.log(pred_sampled.clamp(min=1e-8)), proj_labels)
                    loss_sem = (loss_ce + loss_ce_aug) / 2.0
                    
                    loss_total = loss_sem + 0.01 * loss_kl

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

