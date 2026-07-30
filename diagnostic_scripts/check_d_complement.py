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

ARCH = ukc.ARCH
DATA = ukc.DATA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_epistemic_uncertainty(logits):
    # Logits should be [N, C] and scaled by tau=15.0 to break blockade
    scaled_logits = logits * 15.0
    evidence = F.softplus(scaled_logits)
    alphas = evidence + 1.0
    S = torch.sum(alphas, dim=1)
    K = alphas.shape[1]
    dist_unc = K / S # [N]
    return dist_unc

def run_d_complement():
    print("Initializing model...")
    model = ukc.load_hdc_model("logs/kitti_pretrain/hdc_sub.pth", num_classes=17)
    model.eval()
    
    corruptions = ['fog', 'crosstalk']
    
    for ct in corruptions:
        print(f"\nRunning D-COMPLEMENT on {ct}...")
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
        subset = torch.utils.data.Subset(ds, range(20)) # Just 20 frames
        dl = DataLoader(subset, batch_size=1, shuffle=False)
        
        confident_agree_correct = 0
        confident_agree_total = 0
        
        confident_disagree_correct = 0
        confident_disagree_total = 0
        
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
                
                # Compute network uncertainty
                dist_unc = compute_epistemic_uncertainty(clean_logits)
                
                # We define confident points as those with uncertainty < 0.5
                # (0.5 is the exact threshold our gating uses before it starts decaying weights)
                confident_mask = dist_unc < 0.5 
                
                # Strong Augmentation (20% spatial dropout, 0.05 XYZ noise)
                aug_in = proj_in.clone()
                drop_mask = torch.rand_like(aug_in[:, 0:1, :, :]) > 0.20
                aug_in = aug_in * drop_mask
                noise = torch.randn_like(aug_in[:, 1:4, :, :]) * 0.05
                aug_in[:, 1:4, :, :] += noise * drop_mask
                
                aug_logits, _, aug_indices, _ = model(aug_in)
                if len(aug_indices) == 0:
                    continue
                aug_preds = aug_logits.argmax(dim=1)
                
                # Align aug_preds to clean_preds
                B, C_ch, H, W = proj_in.shape
                full_aug_preds = torch.full((B * H * W,), -1, device=device, dtype=clean_preds.dtype)
                full_aug_preds[aug_indices] = aug_preds
                aligned_aug_preds = full_aug_preds[clean_indices]
                
                agree_mask = (clean_preds == aligned_aug_preds)
                disagree_mask = ~agree_mask
                correct_mask = (clean_preds == clean_labels)
                
                # Filter valid ground truth
                valid_gt = (clean_labels >= 0) & (clean_labels < 17)
                
                mask_ca = confident_mask & agree_mask & valid_gt
                mask_cd = confident_mask & disagree_mask & valid_gt
                
                confident_agree_total += mask_ca.sum().item()
                confident_agree_correct += (mask_ca & correct_mask).sum().item()
                
                confident_disagree_total += mask_cd.sum().item()
                confident_disagree_correct += (mask_cd & correct_mask).sum().item()
                
        prec_agree = confident_agree_correct / max(1, confident_agree_total)
        prec_disagree = confident_disagree_correct / max(1, confident_disagree_total)
        
        print(f"\n=== D-COMPLEMENT Verdict ({ct}) ===")
        print(f"Confident & Agree:    N={confident_agree_total:7d}, Precision: {prec_agree:.4f}")
        print(f"Confident & Disagree: N={confident_disagree_total:7d}, Precision: {prec_disagree:.4f}")

if __name__ == "__main__":
    run_d_complement()
