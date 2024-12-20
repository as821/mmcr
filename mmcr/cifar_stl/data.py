from __future__ import annotations

import sys

sys.path.append("..")
from PIL import Image
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF
from torchvision.datasets import CIFAR10
import torch

import random
from PIL import Image, ImageOps, ImageFilter


def get_datasets(dataset, n_aug, batch_transform=True, supervised=False, strong_aug=False, diffusion_aug=False, weak_aug=False, strongest_aug=False, **kwargs):
    data_dir = "./datasets/"
    if dataset == "stl10":
        train_split = "train" if supervised else "train+unlabeled"
        train_data = torchvision.datasets.STL10(
            root=data_dir,
            split=train_split,
            transform=StlBatchTransform(
                train_transform=True, n_transform=n_aug, batch_transform=batch_transform
            ),
            download=False,
        )
        memory_data = torchvision.datasets.STL10(
            root=data_dir,
            split="train",
            transform=StlBatchTransform(
                train_transform=False, batch_transform=False, n_transform=n_aug
            ),
            download=False,
        )
        test_data = torchvision.datasets.STL10(
            root=data_dir,
            split="test",
            transform=StlBatchTransform(
                train_transform=False, batch_transform=False, n_transform=n_aug
            ),
            download=False,
        )
    elif dataset == "cifar10":
        train_data = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=CifarBatchTransform(
                train_transform=True,
                batch_transform=batch_transform,
                n_transform=n_aug,
                strong_aug=strong_aug,
                strongest_aug=strongest_aug,
                weak_aug=weak_aug,
                diffusion_aug=diffusion_aug,
                **kwargs,
            ),
            download=True,
        )
        memory_data = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=CifarBatchTransform(
                train_transform=False,
                batch_transform=False,
                n_transform=n_aug,
                strong_aug=strong_aug,
                strongest_aug=strongest_aug,
                weak_aug=weak_aug,
                diffusion_aug=diffusion_aug,
                **kwargs,
            ),
            download=True,
        )
        test_data = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            transform=CifarBatchTransform(
                train_transform=False,
                batch_transform=False,
                n_transform=n_aug,
                strong_aug=strong_aug,
                strongest_aug=strongest_aug,
                weak_aug=weak_aug,
                diffusion_aug=diffusion_aug,
                **kwargs,
            ),
            download=True,
        )
    elif dataset == "cifar100":
        train_data = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            transform=CifarBatchTransform(
                train_transform=True,
                batch_transform=batch_transform,
                n_transform=n_aug,
                strong_aug=strong_aug,
                strongest_aug=strongest_aug,
                weak_aug=weak_aug,
                diffusion_aug=diffusion_aug,
                **kwargs,
            ),
            download=True,
        )
        memory_data = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            transform=CifarBatchTransform(
                train_transform=False,
                batch_transform=False,
                n_transform=n_aug,
                strong_aug=strong_aug,
                strongest_aug=strongest_aug,
                weak_aug=weak_aug,
                diffusion_aug=diffusion_aug,
                **kwargs,
            ),
            download=True,
        )
        test_data = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=False,
            transform=CifarBatchTransform(
                train_transform=False,
                batch_transform=False,
                n_transform=n_aug,
                strong_aug=strong_aug,
                strongest_aug=strongest_aug,
                weak_aug=weak_aug,
                diffusion_aug=diffusion_aug,
                **kwargs,
            ),
            download=True,
        )

    return train_data, memory_data, test_data


class StlBatchTransform:
    def __init__(self, n_transform, train_transform=True, batch_transform=True):
        if train_transform is True:
            self.transform = transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(
                        64,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(70, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27)),
                ]
            )
        self.n_transform = n_transform
        self.batch_transform = batch_transform

    def __call__(self, x):
        if self.batch_transform:
            C, H, W = TF.to_tensor(x).shape
            C_aug, H_aug, W_aug = self.transform(x).shape

            y = torch.zeros(self.n_transform, C_aug, H_aug, W_aug)
            for i in range(self.n_transform):
                y[i, :, :, :] = self.transform(x)
            return y
        else:
            return self.transform(x)


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class CifarBatchTransform:
    def __init__(
        self,
        n_transform,
        train_transform=True,
        batch_transform=True,
        strong_aug=False,
        diffusion_aug=False,
        weak_aug=False,
        strongest_aug=False,
        **kwargs,
    ):
        if train_transform:
            if strong_aug:
                lst_of_transform = [
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(0.5),
                    Solarization(0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            elif strongest_aug:
                lst_of_transform = [
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(0.75),
                    Solarization(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            elif weak_aug:
                lst_of_transform = [
                    transforms.RandomResizedCrop(32, scale=(0.3, 1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]                
            elif diffusion_aug:
                lst_of_transform = [
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            else:
                lst_of_transform = [
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]

            self.transform = transforms.Compose(lst_of_transform)
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            )
        self.n_transform = n_transform
        self.batch_transform = batch_transform
        self.diffusion_aug = diffusion_aug

    def __call__(self, x):
        if self.batch_transform:
            C, H, W = TF.to_tensor(x).shape
            C_aug, H_aug, W_aug = self.transform(x).shape

            y = torch.zeros(self.n_transform, C_aug, H_aug, W_aug)
            for i in range(self.n_transform):
                y[i, :, :, :] = self.transform(x)
            return y
        else:
            return self.transform(x)
