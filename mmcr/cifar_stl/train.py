# without this we lose GIGABYTES to memory fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torchvision
from tqdm import tqdm
import einops
import wandb

import torch.nn.functional as F

from mmcr.cifar_stl.data import get_datasets, CifarBatchTransform
from mmcr.cifar_stl.models import Model
from mmcr.cifar_stl.knn import test_one_epoch
from mmcr.cifar_stl.loss_mmcr import MMCR_Loss, BatchFIFOQueue, GradientPreconditioning
from mmcr.cifar_stl.analysis import calc_manifold_subspace_alignment, visualize_augmentations, visualize_feature_cov_decomp


import pdb

def vis_dist(key_name, prefix, vis_dict, loss_dict):
    foo = loss_dict[key_name]
    vis_dict[prefix + "_min"] = foo.min()
    vis_dict[prefix + "_max"] = foo.max()
    vis_dict[prefix + "_mean"] = foo.mean()
    vis_dict[prefix] = wandb.Histogram(foo)
    return vis_dict

def train(args):
    if args.wandb:
        wandb.init(config={
            "dataset":args.dataset,
            "n_aug": args.n_aug,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "lmbda": args.lmbda,
            "strong_aug":args.stronger_aug,
            "strongest_aug":args.strongest_aug,
            "diffusion_aug":args.diffusion_aug,
            "weak_aug":args.weak_aug,
            "diffusion_alpha":args.diff_alpha,
            "spectral_target":args.spectral_target,
            "spectral_topk":args.spectral_topk
        }, project="mmcr", entity="cmu-slots-group")


    torch.set_float32_matmul_precision('high')

    train_dataset, memory_dataset, test_dataset = get_datasets(
        dataset=args.dataset, n_aug=args.n_aug, strong_aug=args.stronger_aug, diffusion_aug=args.diffusion_aug, weak_aug=args.weak_aug, strongest_aug=args.strongest_aug
    )
    model = Model(projector_dims=[512, 128], dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True #, prefetch_factor=4, persistent_workers=True
    )
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=128, shuffle=True, num_workers=16
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=16
    )

    # test set with training transformations
    stats_dset = torchvision.datasets.CIFAR10(root="./datasets/", train=False, download=True, transform=CifarBatchTransform(train_transform=True, batch_transform=True, n_transform=10))
    stats_loader = torch.utils.data.DataLoader(stats_dset, batch_size=128, shuffle=False, num_workers=12)
    stats_data = next(iter(stats_loader))


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)

    sched = []
    if args.warmup_epoch > 0:
        warmup_iter = args.warmup_epoch * len(train_loader)
        sched.append(torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_iter))
    sched.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=args.final_lr))
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(sched)

    if args.wandb:
        wandb.watch(model, log_freq=10)

    total_loss, total_num, vis_dict = 0.0, 0, {}
    loss_function = MMCR_Loss(lmbda=args.lmbda, n_aug=args.n_aug, distributed=False, l2_spectral_norm=args.l2_spectral_norm, spectral_target=args.spectral_target, spectral_topk=args.spectral_topk, memory_bank=BatchFIFOQueue(args.mem_bank, args.batch_size) if args.mem_bank > 0 else None)
    preconditioner = GradientPreconditioning.apply

    model = model.cuda()
    model = torch.compile(model, mode="max-autotune")
    top_acc, total_steps = 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        train_bar = tqdm(train_loader)
        for step, data_tuple in enumerate(train_bar):
            optimizer.zero_grad(set_to_none=True)

            # forward pass
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            img_batch, labels = data_tuple
            img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W").cuda(non_blocking=True)
            feat, model_out = model(img_batch)
            if args.precond_alpha > 0:
                model_out = preconditioner(model_out, args.precond_alpha)
            loss, loss_dict = loss_function(model_out)
            
            # backward pass
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            # update the training bar
            total_num += data_tuple[0].size(0)
            total_loss += loss.item() * data_tuple[0].size(0)

            train_bar.set_description(
                "Train Epoch: [{}/{}] Loss: {:.3f}".format(
                    epoch, args.epochs, loss.item()
                )
            )
            total_steps += 1

            if total_steps % args.log_freq == 0:
                with torch.no_grad():
                    model.eval()
                    acc_1, acc_5 = test_one_epoch(model, memory_loader, test_loader)
                    if acc_1 > top_acc:
                        top_acc = acc_1
                    model.eval()

                    if args.wandb:
                        # check manifold subspace alignment 
                        vis_dict = calc_manifold_subspace_alignment(vis_dict, model, stats_data, False)
                        vis_dict = calc_manifold_subspace_alignment(vis_dict, model, stats_data, True)

                        # visualize augmentations
                        # img_batch = einops.rearrange(img_batch.detach().cpu(), "(B N) C H W -> B N C H W", B=args.batch_size)
                        # vis_dict = visualize_augmentations(vis_dict, img_batch)
                        assert not model.training

                        # track the e'val of the feature covariance matrix
                        model_out = F.normalize(model_out, dim=-1)
                        vis_dict = visualize_feature_cov_decomp(vis_dict, model_out, total_steps, prefix="out")

                        feat = F.normalize(feat, dim=-1)
                        vis_dict = visualize_feature_cov_decomp(vis_dict, feat, total_steps)

                        vis_dict["train_loss"] = total_loss / total_num
                        vis_dict["val_acc_1"] = acc_1
                        vis_dict["val_acc_5"] = acc_5
                        vis_dict["lr"] = scheduler.get_last_lr()[0]
                        vis_dict["precond_alpha"] = args.precond_alpha

                        wandb.log(vis_dict, step=total_steps)

                    assert not model.training
                    model.train()
                    
                    total_loss, total_num, vis_dict = 0.0, 0, {}

                    if epoch % args.save_freq == 0 or acc_1 == top_acc:
                        torch.save(
                            model.state_dict(),
                            f"{args.save_folder}/{args.dataset}_{args.n_aug}_{epoch}_acc_{acc_1:0.2f}.pth",
                        )

    if args.wandb:
        wandb.finish()
