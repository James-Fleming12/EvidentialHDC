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
    
    model = set_uq_model(ARCH, 'logs/kitti_pretrain', 'rp', 0, 0, 17, device)
    model.load_state_dict(torch.load('logs/kitti_pretrain/hdc_sub.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()

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

    margin_correct = []
    margin_hallucination = []
    
    print("Extracting Margins from Fog dataset...")
    
    for i, batch_data in enumerate(dataloader):
        if i >= 10: # Sample 10 frames
            break
            
        proj_in = batch_data[0].to(device)
        proj_mask = batch_data[1].to(device) if len(batch_data) > 1 else None
        proj_labels = batch_data[2].to(device)
        
        valid_mask = (proj_mask.reshape(-1) > 0)
        true_valid = proj_labels.reshape(-1)[valid_mask]
        
        # Get embeddings
        raw_enc = model.encode(proj_in)[0].reshape(-1, 10000)[valid_mask]
        norm_enc = torch.nn.functional.normalize(raw_enc, dim=1).to(torch.float32)
        
        # Get cosine similarities to prototypes (from Linear layer)
        logits = model.get_predictions(norm_enc)
        
        # Top-1 and Top-2 similarities
        topk_sims, topk_idx = logits.topk(k=2, dim=1)
        margin = topk_sims[:, 0] - topk_sims[:, 1]
        
        pseudo_valid = topk_idx[:, 0]
        
        is_correct = (pseudo_valid == true_valid)
        
        margin_correct.append(margin[is_correct].cpu())
        margin_hallucination.append(margin[~is_correct].cpu())

    margin_correct = torch.cat(margin_correct, dim=0).numpy()
    margin_hallucination = torch.cat(margin_hallucination, dim=0).numpy()
    
    print("\n=== Margin Distribution (s_1 - s_2) ===")
    print(f"Correct Points Margin      : mean={np.mean(margin_correct):.4f}, median={np.median(margin_correct):.4f}")
    print(f"Hallucination Margin       : mean={np.mean(margin_hallucination):.4f}, median={np.median(margin_hallucination):.4f}")
    
    # Calculate % of hallucinations with margin > 0.1
    high_margin_halluc = (margin_hallucination > 0.1).sum() / len(margin_hallucination) * 100
    print(f"\n% of Hallucinations with Margin > 0.1: {high_margin_halluc:.2f}%")
    if high_margin_halluc > 50:
        print("Verdict: Margins are massive. Hallucinations are deep inside the wrong cluster. Margin thresholding will fail.")

if __name__ == '__main__':
    main()
