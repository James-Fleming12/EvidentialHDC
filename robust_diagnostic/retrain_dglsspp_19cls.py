"""retrain_dglsspp_19cls.py: retrain the DGLSS++ extractor on GeoID's 19-class
map (config/labels/semantic-kitti-19.yaml) so the decoder evaluation can be
scored on the SAME label space and mIoU convention as GeoID
(thirdparty/GeoID/utils/eval.py, fixed-19 mean).

The 17-class-trained DGLSS++ head cannot be evaluated on the 19-class map (its
classes are MERGED: manmade = building+fence+pole+sign+structure, driveable =
road+parking+lane-marking, vegetation+trunk, pedestrian = person+bicyclist+
motorcyclist). Retraining with a 19-class head gives the encoder a chance to
learn the fine distinctions, after which the decoder harness runs with
--map19 against this checkpoint.

Recipe follows the established DGLSS++ medium run (robust_iterations.md: the
supcon_vib_dglsspp checkpoint was 24 epochs at 100% of the data). Train split
comes from the config (0-7, 9, 10; valid 8), so seq 08 is never seen in
training.

Usage:
  uv run python robust_diagnostic/retrain_dglsspp_19cls.py \
    --log_dir robust_diagnostic/logs/dglsspp_19cls --epochs 24
Then evaluate against the retrained checkpoint:
  CKPT="robust_diagnostic/logs/dglsspp_19cls" MAP19=1 \
    CONDS="fog,crosstalk,wet_ground" bash run_lp_three_decoder.sh 3
"""
import os, sys, argparse, yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--data_cfg", type=str, default="config/labels/semantic-kitti-19.yaml")
    ap.add_argument("--log_dir", type=str, default="robust_diagnostic/logs/dglsspp_19cls")
    ap.add_argument("--method", type=str, default="supcon_vib_dglsspp")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--cutoff", type=float, default=1.0, help="fraction of each epoch's data (1.0 = 100%)")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.gen_trainers import GenTrainer

    ARCH = yaml.safe_load(open(args.arch, 'r'))
    DATA = yaml.safe_load(open(args.data_cfg, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} | arch {args.arch} | data {args.data_cfg}")
    print(f"  method {args.method} | epochs {args.epochs} | cutoff {args.cutoff} | log {args.log_dir}")
    print(f"  train split {DATA['split']['train']} | valid {DATA['split']['valid']} | "
          f"{len(DATA['learning_map_inv'])} classes")

    os.makedirs(args.log_dir, exist_ok=True)
    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.log_dir, method=args.method,
                         cutoff_percent=args.cutoff)
    trainer.train(epochs=args.epochs)
    print(f"\nRetrain done -> {args.log_dir} (SENet_valid_best).")
    print("Evaluate with:")
    print(f"  CKPT=\"{args.log_dir}\" MAP19=1 CONDS=\"fog,crosstalk,wet_ground\" "
          f"bash run_lp_three_decoder.sh 3")


if __name__ == "__main__":
    main()
