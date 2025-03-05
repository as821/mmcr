import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torch import Tensor
from typing import Tuple


from mmcr.cifar_stl.vision_transformer import vit_tiny, trunc_normal_

import pdb


class Model(nn.Module):
    def __init__(self, projector_dims, dataset):
        super(Model, self).__init__()

        # self.f = []
        # for name, module in resnet18().named_children():
        #     if name == "conv1":
        #         module = nn.Conv2d(
        #             3, 64, kernel_size=3, stride=1, padding=1, bias=False
        #         )
        #     if dataset == "cifar10" or "cifar100":
        #         if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
        #             self.f.append(module)
        #     elif dataset == "stl10":
        #         if not isinstance(module, nn.Linear):
        #             self.f.append(module)
        # # encoder
        # self.f = nn.Sequential(*self.f)

        self.use_cls_token = False
        if self.use_cls_token:
            self.f = vit_tiny(patch_size=4, enable_cls=True)
        else:
            self.f = vit_tiny(patch_size=24, patch_stride=1)

        # projection head (Following exactly barlow twins offical repo)
        projector_dims = [192] + projector_dims
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=False)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=False))
        self.g = nn.Sequential(*layers)
        
        # self.g.apply(self._init_weights)

        # TODO: try the DINO projection head as well


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.use_cls_token:
            feature = self.f(x)
            out = self.g(feature)
        else:
            feature = self.f(x)
            out = self.g(torch.flatten(feature, start_dim=0, end_dim=1))
            out = torch.unflatten(out, 0, (feature.shape[0], feature.shape[1]))
        return feature, out
