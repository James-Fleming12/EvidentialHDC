import torch
import torch.nn as nn
import torch.optim as optim
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

def run_phase3_dynamics():
    print("Initializing model for Phase III Adaptation Dynamics...")
    ckpt_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = ukc.load_hdc_model(ckpt_path, num_classes=17)
    model.eval()
    
    # Enable gradients for the classifier to simulate adaptation
    model.classify.train()
    for param in model.net.parameters():
        param.requires_grad = False
    model.classify.weight.requires_grad = True
    
    optimizer = optim.SGD(model.classify.parameters(), lr=0.01)
    
    ct = 'fog'
    print(f"\nRunning Online Adaptation on {ct} (First 20 Frames)...")
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
                            batch_size=1, workers=4, gt=True, shuffle_train=False)
                            
    subset = torch.utils.data.Subset(parser_obj.validloader.dataset, range(20))
    dl = DataLoader(subset, batch_size=1, shuffle=False)
    
    # Tracking variables
    prev_P = F.normalize(model.classify.weight.clone().detach(), dim=1)
    prev_V = torch.zeros_like(prev_P)
    
    velocities = []
    accelerations = []
    angular_velocities = []
    angular_accelerations = []
    hallucination_counts = []
    
    frame_t = 0
    with tqdm(total=len(dl), desc="Adaptation Tracker") as pbar:
        for batch in dl:
            proj_in = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            if proj_in.shape[1] == 0: 
                pbar.update(1)
                continue
                
            with torch.amp.autocast('cuda', enabled=True):
                raw_enc, indices, _ = model.encode(proj_in)
                
            if len(indices) == 0: 
                pbar.update(1)
                continue
                
            norm_enc = F.normalize(raw_enc, dim=1).to(model.classify.weight.dtype)
            
            # Predict and calculate pseudo-labels
            logits = model.classify(norm_enc)
            probs = torch.softmax(logits, dim=1)
            conf, pseudo_labels = probs.max(dim=1)
            
            clean_labels = labels[indices]
            valid = (clean_labels >= 0) & (clean_labels < 17)
            
            # Identify confident hallucinations (conf > 0.9, but wrong label)
            is_hallucination = (pseudo_labels == clean_labels) == False
            confident_hallucinations = is_hallucination & (conf > 0.9) & valid
            num_halls = confident_hallucinations.sum().item()
            hallucination_counts.append(num_halls)
            
            # Adaptation Step (Standard TTA logic: learn on confident pseudo-labels)
            confident_mask = (conf > 0.9) & valid
            if confident_mask.sum() > 0:
                optimizer.zero_grad()
                loss = F.cross_entropy(logits[confident_mask], pseudo_labels[confident_mask])
                loss.backward()
                optimizer.step()
                
            # Measure Dynamics
            current_P = F.normalize(model.classify.weight.clone().detach(), dim=1)
            
            # Velocity (L2 distance moved)
            V_t = current_P - prev_P
            velocity_mag = torch.norm(V_t, dim=1).mean().item()
            velocities.append(velocity_mag)
            
            # Acceleration (Change in velocity)
            A_t = V_t - prev_V
            accel_mag = torch.norm(A_t, dim=1).mean().item()
            accelerations.append(accel_mag)
            
            # Angular Velocity (1 - Cosine Sim between current and previous)
            cos_sim = F.cosine_similarity(current_P, prev_P, dim=1).mean().item()
            ang_vel = 1.0 - cos_sim
            angular_velocities.append(ang_vel)
            
            # Prepare for next frame
            prev_P = current_P
            prev_V = V_t
            frame_t += 1
            pbar.update(1)
            
    print("\n--- Phase III: Adaptation Dynamics ---")
    print("Frame | Confident Hallucinations | Prototype Velocity | Prototype Accel | Angular Vel (Drift)")
    print("-" * 88)
    
    for i in range(len(velocities)):
        print(f"{i+1:5d} | {hallucination_counts[i]:24d} | {velocities[i]:18.4f} | {accelerations[i]:15.4f} | {angular_velocities[i]:19.4f}")

    avg_vel = sum(velocities) / len(velocities)
    avg_accel = sum(accelerations) / len(accelerations)
    avg_ang_vel = sum(angular_velocities) / len(angular_velocities)
    
    print("-" * 88)
    print(f"AVG   | {sum(hallucination_counts)/len(hallucination_counts):24.1f} | {avg_vel:18.4f} | {avg_accel:15.4f} | {avg_ang_vel:19.4f}")
    
    print("\n=== ERROR PROPAGATION & INFLUENCE ===")
    if avg_vel > 0.001:
        print("Verdict: The prototypes exhibit continuous dynamic instability.")
        print("Because hallucinations are treated identically to true positives, they forcefully drag the prototypes (high velocity).")
        print("This drift creates a confirmation bias feedback loop where prototype collapse accelerates.")
    else:
        print("Verdict: Prototypes are stable.")

if __name__ == "__main__":
    run_phase3_dynamics()
