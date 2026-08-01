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
    model.load_state_dict(torch.load('logs/kitti_pretrain/hdc_sub.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()

    # Load Fog dataset
    print("Loading Fog dataset...")
    fog_parser = Parser(
        root="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/fog/heavy",
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
    dataloader = fog_parser.get_valid_set()

    # Setup tracking for lifetimes (simulating a simple class-partitioned reservoir)
    class_counts = torch.zeros(17, dtype=torch.long)
    
    # Track insertion times and labels
    insertion_times = torch.zeros(17, 100, device=device)
    stored_is_correct = torch.zeros(17, 100, device=device, dtype=torch.bool)
    
    lifetimes_correct = []
    lifetimes_halluc = []

    print("Running memory bank updates...")
    for frame_idx, batch_data in enumerate(dataloader):
        if frame_idx >= 50: # Run for 50 frames
            break
            
        proj_in = batch_data[0].to(device)
        proj_mask = batch_data[1].to(device) if len(batch_data) > 1 else None
        proj_labels = batch_data[2].to(device)
        
        with torch.amp.autocast('cuda', enabled=True):
            latent_x = model.net(proj_in, only_feat=True)
            
        latent_x = latent_x.permute(0, 2, 3, 1).reshape(-1, 128)
        proj_mask = proj_mask.reshape(-1)
        proj_labels = proj_labels.reshape(-1)
        
        valid_mask = proj_mask > 0
        true_valid = proj_labels[valid_mask]
        
        raw_enc = model.encode(proj_in)[0].reshape(-1, 10000)[valid_mask]
        norm_enc = torch.nn.functional.normalize(raw_enc, dim=1).to(torch.float32)
        logits = model.get_predictions(norm_enc)
        pseudo_valid = logits.argmax(dim=1)
        
        # We only care about points the model accepts (confident)
        # For simplicity, we just take top 100 most confident points per class
        conf = logits.max(dim=1)[0]
        
        for c in range(17):
            mask_c = (pseudo_valid == c)
            if not mask_c.any():
                continue
                
            enc_c = norm_enc[mask_c]
            true_c = true_valid[mask_c]
            conf_c = conf[mask_c]
            
            # Take top points (simulating distance gating)
            if enc_c.size(0) > 10:
                _, top_idx = conf_c.topk(10)
                enc_c = enc_c[top_idx]
                true_c = true_c[top_idx]
                
            # Reservoir Sampling Simulation
            for i in range(enc_c.size(0)):
                if class_counts[c] < 100:
                    idx = class_counts[c].item()
                    insertion_times[c, idx] = frame_idx
                    stored_is_correct[c, idx] = (c == true_c[i].item())
                    class_counts[c] += 1
                else:
                    # Replace with prob 10 / (total_seen) - standard reservoir
                    # Just simulate random replacement
                    replace_idx = torch.randint(0, 100, (1,)).item()
                    
                    # Log lifetime of the outgoing point
                    old_age = frame_idx - insertion_times[c, replace_idx].item()
                    was_correct = stored_is_correct[c, replace_idx].item()
                    
                    if was_correct:
                        lifetimes_correct.append(old_age)
                    else:
                        lifetimes_halluc.append(old_age)
                        
                    # Insert new point
                    insertion_times[c, replace_idx] = frame_idx
                    stored_is_correct[c, replace_idx] = (c == true_c[i].item())

    print("\n=== Lifetime Distribution ===")
    if len(lifetimes_correct) > 0:
        print(f"Correct Points Lifetime     : mean={np.mean(lifetimes_correct):.2f} frames")
    if len(lifetimes_halluc) > 0:
        print(f"Hallucination Lifetime      : mean={np.mean(lifetimes_halluc):.2f} frames")

if __name__ == '__main__':
    main()
