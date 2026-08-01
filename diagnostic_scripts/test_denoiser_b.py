import torch
import torch.nn as nn
import yaml
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.kitti.parser import Parser
from modules.HDC_utils import set_uq_model
from modules.AdaptMemModel import HDCDenoiser

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA = yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))
    model = set_uq_model(ARCH, 'logs/kitti_pretrain', 'rp', 0, 0, 17, device)
    model.load_state_dict(torch.load('logs/kitti_pretrain/hdc_sub.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    print("Running populate source stats to train denoiser...")
    cache = model.populate_source_statistics("/mnt/alpha/jmfleming/KITTI", ARCH, DATA, device, dry_run=True)
    
    denoiser = HDCDenoiser(hd_dim=10000, hidden_dim=256).to(device)
    denoiser.load_state_dict(cache['denoiser_state_dict'])
    denoiser.eval()
    
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
    
    errors = []
    with torch.no_grad():
        for i, batch_data in enumerate(fog_parser.get_valid_set()):
            if i >= 5: break
            proj_in = batch_data[0].to(device)
            mask = (batch_data[1].to(device).reshape(-1) > 0)
            
            raw_enc = model.encode(proj_in)[0].reshape(-1, 10000)[mask]
            norm_enc = torch.nn.functional.normalize(raw_enc, dim=1).to(torch.float32)
            
            bin_enc = torch.sign(norm_enc)
            bin_enc[bin_enc == 0] = 1.0
            
            recon = denoiser(bin_enc)
            err = 1.0 - torch.cosine_similarity(bin_enc, recon, dim=1)
            errors.append(err.cpu())
            
    all_err = torch.cat(errors)
    print(f"Binarized Fog Mean Error: {all_err.mean():.4f}")
    print(f"Binarized Fog Min Error: {all_err.min():.4f}")
    print(f"Binarized Fog Max Error: {all_err.max():.4f}")

if __name__ == '__main__':
    main()
