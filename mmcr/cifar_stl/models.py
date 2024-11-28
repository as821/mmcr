import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torch import Tensor
from typing import Tuple



class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()
        projector_dims = [1024, 256, 1]
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=False)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=False))
        
        self.g = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.g(torch.concat([x, y], dim=-1))

class Model(nn.Module):
    def __init__(self, projector_dims: list = [512, 128], dataset: str = "cifar10"):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == "conv1":
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
            if dataset == "cifar10" or "cifar100":
                if not isinstance(module, nn.Linear) and not isinstance(
                    module, nn.MaxPool2d
                ):
                    self.f.append(module)
            elif dataset == "stl10":
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)

        # projection head (Following exactly barlow twins offical repo)
        # projector_dims = [512] + projector_dims
        # layers = []
        # for i in range(len(projector_dims) - 2):
        #     layers.append(
        #         nn.Linear(projector_dims[i], projector_dims[i + 1], bias=False)
        #     )
        #     layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
        #     layers.append(nn.ReLU())
        # layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=False))
        # self.g = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        # out = self.g(feature)
        out = None
        return feature, out
