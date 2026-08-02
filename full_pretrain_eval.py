import os
import yaml
import argparse
from modules.gen_trainers import GenTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of full-scale epochs to run")
    args = parser.parse_args()

    # Load configurations
    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))

    methods = ['supcon', 'supcon_vib']

    for method in methods:
        print(f"\n{'='*60}")
        print(f" Starting Full-Scale Pretraining: {method.upper()}")
        print(f"{'='*60}")
        
        # Give each method its own distinct logging and weight-saving directory
        log_dir = os.path.join(args.log_dir, f"full_pretrain_{method}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Instantiate GenTrainer (inherits from standard PyTorch trainer, automatically handles checkpoint saving)
        # It will save standard weights to log_dir just like unsup_main.py (e.g. SENet_valid_best)
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, path=None, method=method)
        
        # Run the full PyTorch loop for N epochs
        trainer.train(epochs=args.epochs)
        
        print(f"Finished {method}. Weights saved to: {log_dir}")

if __name__ == "__main__":
    main()
