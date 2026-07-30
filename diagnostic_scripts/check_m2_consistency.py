import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random

try:
    from modules.HDC_utils import HDC_Net
    from models.pointnet2.pointnet2_msg import Pointnet2MSG as Pointnet2
    import datasets.kitti_c_utils as ukc
except ImportError:
    print("Please run this script from the EvidentialHDC root directory.")
    exit(1)

def strong_augmentation(proj_in, drop_prob=0.2, noise_std=0.05):
    """
    Applies a strong augmentation to the LiDAR range projection.
    proj_in: [B, 5, H, W] tensor (e.g., [x, y, z, remission, range])
    """
    aug_proj = proj_in.clone()
    
    # 1. Random Point Dropout (simulated by setting features to 0)
    # We apply this mask to the range map and features
    B, C, H, W = aug_proj.shape
    drop_mask = (torch.rand(B, 1, H, W, device=aug_proj.device) > drop_prob).float()
    aug_proj = aug_proj * drop_mask
    
    # 2. Gaussian Noise on XYZ channels (channels 0, 1, 2)
    noise = torch.randn(B, 3, H, W, device=aug_proj.device) * noise_std
    aug_proj[:, 0:3, :, :] += (noise * drop_mask) # Only add noise to non-dropped points
    
    return aug_proj

def run_m2_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing model...")
    
    net = Pointnet2(num_classes=17, use_xyz=True).to(device)
    model = HDC_Net(net, num_classes=17, device=device).to(device)
    
    ckpt_path = 'logs/kitti_pretrain/hdc_sub.pth'
    if not os.path.exists(ckpt_path):
        print(f"Cannot find {ckpt_path}.")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, strict=False)
    model.eval()

    # Load a clean sequence (or wet_ground) to see if consistency holds for REAL points
    ds = ukc.SemanticKITTI_C(corruption='wet_ground', severity=3, split='test')
    subset = torch.utils.data.Subset(ds, range(20)) # Just 20 frames
    dl = DataLoader(subset, batch_size=1, shuffle=False)
    
    print("\nValidating Strong Augmentation Semantics on wet_ground...")
    
    total_points = 0
    consistent_points = 0
    clean_acc = 0
    aug_acc = 0
    
    with torch.no_grad():
        for batch in tqdm(dl):
            proj_in = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            
            if proj_in.shape[1] == 0:
                continue
                
            # Clean pass
            clean_logits, _, clean_indices, _ = model(proj_in)
            clean_preds = clean_logits.argmax(dim=1)
            clean_labels = labels[clean_indices]
            valid_mask = (clean_labels >= 0) & (clean_labels < 17)
            
            # Strong Aug pass
            aug_in = strong_augmentation(proj_in)
            aug_logits, _, aug_indices, _ = model(aug_in)
            aug_preds = aug_logits.argmax(dim=1)
            aug_labels = labels[aug_indices]
            aug_valid_mask = (aug_labels >= 0) & (aug_labels < 17)
            
            # For simplicity, compare points that survived encoding in both (assuming indices align roughly)
            # In practice, HDC encodes valid points. We compare accuracy to GT.
            clean_acc += (clean_preds[valid_mask] == clean_labels[valid_mask]).sum().item()
            aug_acc += (aug_preds[aug_valid_mask] == aug_labels[aug_valid_mask]).sum().item()
            total_points += valid_mask.sum().item()
            
    clean_accuracy = clean_acc / total_points
    aug_accuracy = aug_acc / total_points
    
    print("\n=== M2 Strong Augmentation Diagnostic ===")
    print(f"Clean Model Accuracy (20 frames): {clean_accuracy:.2%}")
    print(f"Strong Aug Model Accuracy:        {aug_accuracy:.2%}")
    print(f"Accuracy Drop:                    {(clean_accuracy - aug_accuracy)*100:.2f}%")
    
    if (clean_accuracy - aug_accuracy) < 0.15:
        print("\n[VERDICT] SUCCESS: The strong augmentation is well-defined! It drops accuracy slightly but preserves core semantics. M2 consistency gating is viable.")
    else:
        print("\n[VERDICT] WARNING: The strong augmentation destroyed too much signal (>15% drop). You must tune 'drop_prob' and 'noise_std' before running M2.")

if __name__ == '__main__':
    run_m2_diagnostic()
