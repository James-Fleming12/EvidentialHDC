import torch
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

def strong_augmentation(proj_in, drop_prob=0.2, noise_std=0.05):
    aug_proj = proj_in.clone()
    B, C, H, W = aug_proj.shape
    drop_mask = (torch.rand(B, 1, H, W, device=aug_proj.device) > drop_prob).float()
    aug_proj = aug_proj * drop_mask
    noise = torch.randn(B, 3, H, W, device=aug_proj.device) * noise_std
    aug_proj[:, 0:3, :, :] += (noise * drop_mask) 
    return aug_proj

def run_m2_diagnostic():
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

    print("\nValidating Strong Augmentation Semantics on wet_ground...")
    
    corruption_root = f"/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/wet_ground/heavy"
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
    
    total_points = 0
    clean_acc = 0
    aug_acc = 0
    
    with torch.no_grad():
        for batch in tqdm(dl):
            proj_in = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            
            if proj_in.shape[1] == 0:
                continue
                
            clean_logits, _, clean_indices, _ = model(proj_in)
            clean_preds = clean_logits.argmax(dim=1)
            clean_labels = labels[clean_indices]
            valid_mask = (clean_labels >= 0) & (clean_labels < 17)
            
            aug_in = strong_augmentation(proj_in)
            aug_logits, _, aug_indices, _ = model(aug_in)
            aug_preds = aug_logits.argmax(dim=1)
            aug_labels = labels[aug_indices]
            aug_valid_mask = (aug_labels >= 0) & (aug_labels < 17)
            
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
