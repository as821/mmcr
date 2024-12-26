from mmcr.cifar_stl.data import get_datasets, CifarBatchTransform
from mmcr.cifar_stl.augment import loss_function, log_model_jacobian, generate_aug_probs, calc_aug_ev_var, calc_aug_var_decomp

from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm
import einops
import wandb
import pdb
import time
import os
import numpy as np


parser = ArgumentParser()
parser.add_argument("--out_dir", type=str, default="/ssd1/aug_var/cifar_stl")
parser.add_argument("--batch_size", type=int, default=192)

args = parser.parse_args()

dir = args.out_dir + "/" + str(int(time.time()))
os.mkdir(dir)

n_workers = 16 if torch.cuda.is_available() else 0
train_dataset, _, _ = get_datasets(dataset="cifar10", n_aug=1, strong_aug=False, diffusion_aug=False, weak_aug=False, strongest_aug=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers, pin_memory=True, drop_last=False)

aug_prob_map = generate_aug_probs([3, 32, 32])
device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.no_grad():
    for step, data_tuple in enumerate(tqdm(train_loader)):
        img_batch, labels = data_tuple
        img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W").to(device, non_blocking=True)
        intermediate = calc_aug_var_decomp(img_batch, aug_prob_map).cpu()

        for idx in range(intermediate.shape[0]):
            num = idx + step * args.batch_size
            name = dir + "/" + str(num) + ".pt"
            
            # .clone is ESSENTIAL here, otherwise torch tries to be "smart" and save full intermediate each time
            # https://github.com/pytorch/pytorch/issues/21926
            torch.save(intermediate[idx].clone(), name)



