import torch
import os
import sys

# Mock the old class names so torch.load can unpickle the full model objects
import modules.HDC_utils
modules.HDC_utils.Model = modules.HDC_utils.UQModel
modules.HDC_utils.KNNModel = modules.HDC_utils.UQModel
modules.HDC_utils.DensityModel = modules.HDC_utils.UQModel

def convert_weights(input_path, output_path):
    print(f"Processing {input_path}...")
    try:
        # Load the old data (using weights_only=False because we are unpickling an object, not just a dict)
        data = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # If it was saved as a full model object (which `trainer.train` did), extract state_dict
        if isinstance(data, torch.nn.Module):
            print("  -> Detected full model object. Extracting state_dict...")
            state_dict = data.state_dict()
        elif isinstance(data, dict):
            print("  -> Detected state_dict. No structural extraction needed.")
            state_dict = data
        else:
            print(f"  -> Unknown data type: {type(data)}. Skipping.")
            return
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(state_dict, output_path)
        print(f"  -> Successfully saved clean state_dict to {output_path}")
        
    except Exception as e:
        print(f"  -> Failed to convert {input_path}: {e}")

if __name__ == "__main__":
    # The user has the logs copied into kitti_pretrain (and potentially nusc_pretrain)
    paths_to_convert = [
        "logs/kitti_pretrain/hdc.pth",
        "logs/kitti_pretrain/hdc_sub.pth",
        "logs/kitti_pretrain/hdc_sub_aug.pth",
        "logs/kitti_pretrain/feature_optimizer.pth",
        "logs/kitti_pretrain/SENet_valid_best",
        "logs/kitti_pretrain/SENet_train_best",
        "logs/nusc_pretrain/hdc.pth",
        "logs/nusc_pretrain/hdc_sub.pth",
        "logs/nusc_pretrain/hdc_sub_aug.pth",
        "logs/nusc_pretrain/feature_optimizer.pth",
        "logs/nusc_pretrain/SENet_valid_best",
        "logs/nusc_pretrain/SENet_train_best",
    ]
    
    # We will read from EvidentialHDC and write back to EvidentialHDC (in-place conversion)
    base_dir = "/home/jmfleming/EvidentialHDC"
    
    for rel_path in paths_to_convert:
        file_p = os.path.join(base_dir, rel_path)
        
        if os.path.exists(file_p):
            convert_weights(file_p, file_p)
        else:
            print(f"Source file {file_p} not found. Skipping.")
