#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil

if __name__ == '__main__':
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        default=None,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        required=True,
        default=None,
        help='Directory to put the predictions. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--train_seq', '-t', 
        type=str, 
        required=False,
        default=None,
        help='Comma-separated list of training sequences (e.g., "0,1,2")'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % FLAGS.model)
        # ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
        ARCH = yaml.safe_load(open("config/arch/senet-512.yml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        # DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
        DATA = yaml.safe_load(open("config/labels/semantic-kitti.yaml", 'r'))
        # DATA = yaml.safe_load(open("config/labels/nuscenes.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # # create log folder
    # try:
    #     if os.path.isdir(FLAGS.log):
    #         shutil.rmtree(FLAGS.log)
    #     os.makedirs(FLAGS.log)
    #     os.makedirs(os.path.join(FLAGS.log, "sequences"))
    #     for seq in DATA["split"]["train"]:
    #         seq = '{0:02d}'.format(int(seq))
    #         print("train", seq)
    #         os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
    #         os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    #     for seq in DATA["split"]["valid"]:
    #         seq = '{0:02d}'.format(int(seq))
    #         print("valid", seq)
    #         os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
    #         os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    #     for seq in DATA["split"]["test"]:
    #         seq = '{0:02d}'.format(int(seq))
    #         print("test", seq)
    #         os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
    #         os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    # except Exception as e:
    #     print(e)
    #     print("Error creating log directory. Check permissions!")
    #     raise

    # except Exception as e:
    #     print(e)
    #     print("Error creating log directory. Check permissions!")
    #     quit()

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()
    
    if FLAGS.train_seq is not None:
        try:
            DATA["split"]["train"] = [int(s.strip()) for s in FLAGS.train_seq.split(",")]
            print(f"[INFO] Overriding training sequences: {DATA['split']['train']}")
        except ValueError as e:
            print(f"[ERROR] Failed to parse --train_seq: {e}")
            exit(1)
    else:
        print("[INFO] Using default training sequences from data config.")

    from modules.Basic_HD import BasicHD
    BasicHD = BasicHD(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, None)
    BasicHD.start()

    # from modules.Basic_Conv import BasicConv
    # BasicConv = BasicConv(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, None)
    # BasicConv.start()

