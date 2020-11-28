from __future__ import print_function

from misc.config import Config
from dataset_mimic import build_dataset
from trainer import JoImTeR as trainer

import os
# import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import pickle
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

cfg  = Config()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a JoImTeR network')
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()


    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    torch.manual_seed(cfg.seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(cfg.seed)
        
    ########################################
    
    
    
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
#     LAMBDA_FT,LAMBDA_FI,LAMBDA_DAMSM=01,50,10
    output_dir = '../output/%s_%s_%s'%(cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    

    data_set = build_dataset('train', cfg)
    train_loader = torch.utils.data.DataLoader(
                    data_set, batch_size=cfg.batch_size, drop_last=True,
                    shuffle=True, num_workers=cfg.num_workers)
    
    
    data_set = build_dataset('val', cfg)
    val_loader = torch.utils.data.DataLoader(
                    data_set, batch_size=cfg.val_batch_size, drop_last=False,
                    shuffle=False, num_workers=cfg.num_workers)
    
    
    
    
    # Define models and go to train/evaluate
    algo = trainer(output_dir, train_loader, val_loader)

    start_t = time.time()
    
    algo.train()
    
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
