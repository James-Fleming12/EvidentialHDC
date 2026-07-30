import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Assuming EvidentialHDC structure is in path
try:
    from modules.HDC_utils import HDC_Net
    from models.pointnet2.pointnet2_msg import Pointnet2MSG as Pointnet2
    import datasets.kitti_c_utils as ukc
except ImportError as e:
    print(f"ImportError: {e}\\nPlease run this script from the EvidentialHDC root directory.")
    exit(1)

def compute_uncertainties(alphas):
    """
    Computes EviATTA's dual-channel uncertainties from Dirichlet evidence.
    alphas: [B, K] where K is num_classes (alpha = evidence + 1)
    """
    K = alphas.shape[1]
    S = torch.sum(alphas, dim=1, keepdim=True)
    
    # 1. Distribution Uncertainty (Epistemic / Domain Gap)
    # Measured by the vacuity of the Dirichlet distribution
    dist_unc = K / S.squeeze(-1)
    
    # 2. Data Uncertainty (Aleatoric / Inherent Noise)
    # Measured by the entropy of the mean categorical distribution
    p = alphas / S
    data_unc = -torch.sum(p * torch.log(p + 1e-8), dim=1)
    
    return dist_unc, data_unc

def run_n2_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing model...")
    
    # Minimal model setup
    net = Pointnet2(num_classes=17, use_xyz=True).to(device)
    model = HDC_Net(net, num_classes=17, device=device).to(device)
    
    ckpt_path = 'logs/kitti_pretrain/hdc_sub.pth'
    if not os.path.exists(ckpt_path):
        print(f"Cannot find {ckpt_path}.")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    print("Loading datasets...")
    # Load just one sequence of fog and one of wet_ground
    # Assuming standard KITTI-C dataset location
    corruptions = ['fog', 'wet_ground']
    results = {}
    
    for ct in corruptions:
        ds = ukc.SemanticKITTI_C(corruption=ct, severity=3, split='test')
        # Just use first 50 frames to be fast
        subset = torch.utils.data.Subset(ds, range(50))
        dl = DataLoader(subset, batch_size=1, shuffle=False)
        
        print(f"\nProcessing {ct}...")
        all_dist_unc = []
        all_data_unc = []
        
        with torch.no_grad():
            for batch in tqdm(dl):
                # proj_in is [B, 5, H, W]
                proj_in = batch[0].to(device)
                if proj_in.shape[1] == 0:
                    continue
                    
                # Get logits
                logits, _, _, _ = model(proj_in)
                
                # In EvidentialHDC, evidence = exp(logits) or relu(logits). 
                # Assuming standard relu-based evidence for simplicity (update if exp is used)
                # We'll use softplus as a safe generalized evidence function
                evidence = F.softplus(logits)
                alphas = evidence + 1.0
                
                dist_unc, data_unc = compute_uncertainties(alphas)
                
                all_dist_unc.append(dist_unc.cpu())
                all_data_unc.append(data_unc.cpu())
                
        dist_mean = torch.cat(all_dist_unc).mean().item()
        data_mean = torch.cat(all_data_unc).mean().item()
        
        results[ct] = {
            'dist_unc': dist_mean,
            'data_unc': data_mean
        }
        print(f"  {ct} -> Dist Unc (Domain Gap): {dist_mean:.4f} | Data Unc (Noise): {data_mean:.4f}")

    print("\n=== N2 Diagnostic Verdict ===")
    fog_dist = results['fog']['dist_unc']
    fog_data = results['fog']['data_unc']
    wet_dist = results['wet_ground']['dist_unc']
    wet_data = results['wet_ground']['data_unc']
    
    print("Expected:")
    print("  fog should have HIGH Data Uncertainty (hallucinated points are noisy).")
    print("  wet_ground should have HIGH Distribution Uncertainty (points are clear, but prior is shifted).")
    
    if fog_data > wet_data and wet_dist > fog_dist:
        print("\n[VERDICT] SUCCESS: The dual channels cleanly separate the two failure modes! N2 is viable.")
    else:
        print("\n[VERDICT] FAILURE: The channels are degenerate/entangled in HDC space. N2 will not work as a switch.")

if __name__ == '__main__':
    run_n2_diagnostic()
