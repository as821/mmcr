import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import random
from typing import Tuple

import sys


class MMCR_Loss(nn.Module):
    def __init__(self, lmbda: float, n_aug: int, distributed: bool = False):
        super(MMCR_Loss, self).__init__()
        self.lmbda = lmbda
        self.n_aug = n_aug
        self.distributed = distributed
        self.first_time = True

    def forward(self, z: Tensor) -> Tuple[Tensor, dict]:
        # print(f"{z.max()} {z.min()}")
        
        z = F.normalize(z, dim=-1)
        z_local_ = einops.rearrange(z, "(B N) C -> B C N", N=self.n_aug)

        # gather across devices into list
        if self.distributed:
            z_list = [
                torch.zeros_like(z_local_)
                for i in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(z_list, z_local_, async_op=False)
            z_list[torch.distributed.get_rank()] = z_local_

            # append all
            z_local = torch.cat(z_list)

        else:
            z_local = z_local_

        centroids = torch.mean(z_local, dim=-1)

        # print(f"{z.max()} {z.min()} {centroids.max()} {centroids.min()}")
        # print(f"{torch.linalg.cond(centroids)}")

        if self.lmbda != 0.0:
            local_nuc = torch.linalg.svdvals(z_local).sum()
        else:
            local_nuc = torch.tensor(0.0)
        global_sing_vals = torch.linalg.svdvals(centroids)
        global_nuc = global_sing_vals.sum()

        batch_size = z_local.shape[0]
        loss = self.lmbda * local_nuc / batch_size - global_nuc

        loss_dict = {
            "loss": loss.item(),
            "local_nuc": local_nuc.item(),
            "global_nuc": global_nuc.item(),
            "global_sing_vals" : global_sing_vals.detach().cpu().numpy(), 
        }

        self.first_time = False

        return loss, loss_dict
