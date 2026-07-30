import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    import importlib
    ukc = importlib.import_module("unsup_kitti-c")
except ImportError as e:
    print(f"ImportError: {e}\nPlease run this script from the EvidentialHDC root directory.")
    exit(1)

def compute_uncertainties(alphas):
    K = alphas.shape[1]
    S = torch.sum(alphas, dim=1) # [B, H, W]
    dist_unc = K / S
    p = alphas / S.unsqueeze(1)
    data_unc = -torch.sum(p * torch.log(p + 1e-8), dim=1)
    return dist_unc, data_unc

def run_n2_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing model...")
    
    ckpt_path = 'logs/kitti_pretrain/hdc_sub.pth'
    if not os.path.exists(ckpt_path):
        print(f"Cannot find {ckpt_path}.")
        return
        
    model = ukc.load_hdc_model(ckpt_path, num_classes=17, mv_tta='none')
    model.eval()

    ARCH = ukc.ARCH
    DATA = ukc.DATA

    print("Loading datasets...")
    corruptions = ['fog', 'wet_ground']
    results = {}
    
    for ct in corruptions:
        corruption_root = f"/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/{ct}/heavy"
        if not os.path.exists(corruption_root):
            corruption_root = "/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/"            
        parser_obj = ukc.Parser(root=corruption_root,
                                train_sequences=['08'],
                                valid_sequences=['08'],
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
        ds = parser_obj.validloader.dataset
        subset = torch.utils.data.Subset(ds, range(20)) # Just 20 frames for speed
        dl = DataLoader(subset, batch_size=1, shuffle=False)
        
        print(f"\nProcessing {ct}...")
        all_dist_unc = []
        all_data_unc = []
        
        with torch.no_grad():
            for batch in tqdm(dl):
                proj_in = batch[0].to(device)
                proj_mask = batch[1].to(device) # [B, H, W]
                if proj_in.shape[1] == 0:
                    continue
                    
                logits, _, _, _ = model(proj_in)
                evidence = F.softplus(logits)
                alphas = evidence + 1.0
                
                dist_unc, data_unc = compute_uncertainties(alphas)
                
                valid_mask = proj_mask > 0
                if valid_mask.sum() > 0:
                    all_dist_unc.append(dist_unc[valid_mask].cpu())
                    all_data_unc.append(data_unc[valid_mask].cpu())
                
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
