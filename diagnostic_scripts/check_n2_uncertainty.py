import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    import unsup_main
    from dataset.kitti.parser import Parser
except ImportError as e:
    print(f"ImportError: {e}\nPlease run this script from the EvidentialHDC root directory.")
    exit(1)

def compute_uncertainties(alphas):
    """
    Computes EviATTA's dual-channel uncertainties from Dirichlet evidence.
    """
    K = alphas.shape[1]
    S = torch.sum(alphas, dim=1, keepdim=True)
    
    # 1. Distribution Uncertainty (Epistemic / Domain Gap)
    dist_unc = K / S.squeeze(-1)
    
    # 2. Data Uncertainty (Aleatoric / Inherent Noise)
    p = alphas / S
    data_unc = -torch.sum(p * torch.log(p + 1e-8), dim=1)
    
    return dist_unc, data_unc

def run_n2_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing model...")
    
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA = yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))
    
    model = unsup_main.train_hdc(ARCH, DATA, epochs=0, return_extractor=False)
    model = model.to(device)
    
    ckpt_path = 'logs/kitti_pretrain/hdc_sub.pth'
    if not os.path.exists(ckpt_path):
        print(f"Cannot find {ckpt_path}.")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, strict=False)
    model.eval()

    print("Loading datasets...")
    corruptions = ['fog', 'wet_ground']
    results = {}
    
    for ct in corruptions:
        corruption_root = f"/mnt/alpha/jmfleming/KITTI/dataset/sequences/08/corruption/{ct}/3/"
        if not os.path.exists(corruption_root):
            # Fallback for generic structure
            corruption_root = f"/mnt/alpha/jmfleming/KITTI/dataset/sequences/08/"
            
        parser_obj = Parser(root=corruption_root,
                            train_sequences=None,
                            valid_sequences=None,
                            test_sequences=['08'],
                            labels=DATA["labels"],
                            color_map=DATA["color_map"],
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"],
                            batch_size=1,
                            workers=4,
                            gt=True,
                            shuffle_train=False)
        ds = parser_obj.get_test_set()
        subset = torch.utils.data.Subset(ds, range(20)) # Just 20 frames for diagnostic
        dl = DataLoader(subset, batch_size=1, shuffle=False)
        
        print(f"\nProcessing {ct}...")
        all_dist_unc = []
        all_data_unc = []
        
        with torch.no_grad():
            for batch in tqdm(dl):
                proj_in = batch[0].to(device)
                if proj_in.shape[1] == 0:
                    continue
                    
                logits, _, _, _ = model(proj_in)
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
