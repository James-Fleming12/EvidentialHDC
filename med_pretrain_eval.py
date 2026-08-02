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
    parser.add_argument("--epochs", type=int, default=3, help="Number of medium-scale epochs to run")
    args = parser.parse_args()

    # Load configurations
    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))

    methods = ['baseline', 'vib', 'supcon', 'supcon_vib']

    for method in methods:
        print(f"\n{'='*60}")
        print(f" Starting Medium-Scale Pretraining: {method.upper()}")
        print(f" Running {args.epochs} epochs on 100% of data (Scheduler-Safe)")
        print(f"{'='*60}")
        
        # Give each method its own distinct logging and weight-saving directory
        log_dir = os.path.join(args.log_dir, f"med_pretrain_{method}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Instantiate GenTrainer (100% dataset for safe scheduler convergence)
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, path=None, method=method)
        
        # Run the full PyTorch loop for N epochs
        trainer.train(epochs=args.epochs)
        
        print(f"Finished {method}. Weights saved to: {log_dir}")

if __name__ == "__main__":
    main()
