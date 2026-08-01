import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import yaml
from dataset.kitti.parser import Parser
from modules.HDC_utils import set_uq_model

@torch.no_grad()
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Loading architecture and model...")
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    
    # Load model
    model = set_uq_model(ARCH, 'logs/kitti_pretrain', 'rp', 0, 0, 17, device)
    model.to(device)
    model.eval()

    # Load baseline dataset
    print("Loading baseline dataset...")
    baseline_parser = Parser(
        root="/mnt/alpha/jmfleming/KITTI",
        train_sequences=["08"],
        valid_sequences=["08"],
        test_sequences=["08"],
        labels=yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["labels"],
        color_map={},
        learning_map=yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["learning_map"],
        learning_map_inv=yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))["learning_map_inv"],
        sensor=ARCH["dataset"]["sensor"],
        max_points=ARCH["dataset"]["max_points"],
        batch_size=1,
        workers=4,
        gt=True,
        shuffle_train=False
    )
    dataloader = baseline_parser.get_valid_set()

    print("Extracting a large pool of clean features...")
    all_enc = []
    all_true = []
    
    for i, batch_data in enumerate(dataloader):
        if i >= 10: # 10 frames is ~1.2M points, plenty for N=1000
            break
            
        proj_in = batch_data[0].to(device)
        proj_mask = batch_data[1].to(device) if len(batch_data) > 1 else None
        proj_labels = batch_data[2].to(device)
        
        valid_mask = (proj_mask.reshape(-1) > 0)
        true_valid = proj_labels.reshape(-1)[valid_mask]
        
        raw_enc = model.encode(proj_in)[0].reshape(-1, 10000)[valid_mask]
        norm_enc = torch.nn.functional.normalize(raw_enc, dim=1).to(torch.float32)
        
        all_enc.append(norm_enc.cpu())
        all_true.append(true_valid.cpu())
        
    all_enc = torch.cat(all_enc, dim=0)
    all_true = torch.cat(all_true, dim=0)
    
    print("\n=== Marginal Utility Curve (Accuracy vs N) ===")
    
    # We will test two classes: Road (Class 9 - Head), Bicycle (Class 1 - Tail)
    # We will create a test set of 1000 points per class.
    # We will vary the memory bank size N = [10, 50, 100, 500, 1000]
    
    test_size = 1000
    N_list = [10, 50, 100, 500, 1000]
    
    for c, c_name in [(9, 'Road (Head)'), (1, 'Bicycle (Tail)')]:
        mask_c = (all_true == c)
        enc_c = all_enc[mask_c]
        
        if enc_c.size(0) < test_size + max(N_list):
            print(f"Not enough points for {c_name} (Found {enc_c.size(0)}). Skipping.")
            continue
            
        # Split into train/test
        indices = torch.randperm(enc_c.size(0))
        enc_c = enc_c[indices]
        
        test_enc = enc_c[:test_size].to(device)
        train_enc = enc_c[test_size:]
        
        # Test against itself and other classes? 
        # Actually to measure *accuracy*, we should query a mixed test set.
        # Let's create a mixed test set of 5000 points.
        pass

    # Better approach: Mix 5000 test points of all classes
    test_indices = torch.randperm(all_enc.size(0))[:5000]
    test_enc = all_enc[test_indices].to(device)
    test_true = all_true[test_indices].to(device)
    
    for N in N_list:
        # Build memory bank with N points PER CLASS
        bank = []
        bank_labels = []
        for c in range(17):
            mask_c = (all_true == c)
            enc_c = all_enc[mask_c]
            
            if enc_c.size(0) == 0:
                continue
                
            n_c = min(N, enc_c.size(0))
            idx_c = torch.randperm(enc_c.size(0))[:n_c]
            
            bank.append(enc_c[idx_c])
            bank_labels.append(torch.full((n_c,), c))
            
        bank = torch.cat(bank, dim=0).to(device)
        bank_labels = torch.cat(bank_labels, dim=0).to(device)
        
        # 1-NN Retrieval
        sim = torch.mm(test_enc, bank.t())
        pred_idx = sim.argmax(dim=1)
        pred = bank_labels[pred_idx]
        
        acc = (pred == test_true).float().mean().item()
        print(f"Memory Bank Size (N={N} per class): Overall Accuracy = {acc*100:.2f}%")

if __name__ == '__main__':
    main()
