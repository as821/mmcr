import torch
import torchvision
from tqdm import tqdm
import einops
import wandb

import torch.nn.functional as F

from mmcr.cifar_stl.data import get_datasets, CifarBatchTransform
from mmcr.cifar_stl.models import Model
from mmcr.cifar_stl.knn import test_one_epoch
from mmcr.cifar_stl.loss_mmcr import MMCR_Loss, BatchFIFOQueue
from mmcr.cifar_stl.analysis import calc_manifold_subspace_alignment, visualize_augmentations, loss_breakdown


import pdb

def calc_metrics(pred, target):
    with torch.no_grad():
        correct_mask = pred == target
        positive_correct_mask = correct_mask & pred
        total_correct = correct_mask.sum()
        true_positive = positive_correct_mask.sum()
        true_negative = total_correct - true_positive

        incorrect_mask = ~correct_mask
        positive_incorrect_mask = incorrect_mask & pred
        total_incorrect = incorrect_mask.sum()
        false_positive = positive_incorrect_mask.sum()
        false_negative = total_incorrect - false_positive

        accuracy = total_correct / pred.shape[0]
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        fpr = false_positive / (true_positive + false_positive)
        fnr = false_negative / (true_negative + false_negative)
        return accuracy, precision, recall, fpr, fnr

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

    def vis_dist(key_name, prefix, vis_dict, loss_dict):
        foo = loss_dict[key_name]
        vis_dict[prefix + "_min"] = foo.min()
        vis_dict[prefix + "_max"] = foo.max()
        vis_dict[prefix + "_mean"] = foo.mean()
        vis_dict[prefix] = wandb.Histogram(foo)
        return vis_dict

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


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_iter = args.warmup_epoch * len(train_loader)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_iter),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=args.final_lr)
    ])

    if args.wandb:
        wandb.watch(model, log_freq=10)

    total_loss, total_num, vis_dict = 0.0, 0, {}
    tot_pos, tot_neg, tot_inter, tot_intra = 0, 0, 0, 0

    target = torch.block_diag(*[torch.ones((args.n_aug, args.n_aug)) for _ in range(args.batch_size)]).cuda()

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
            img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W")
            _, out = model(img_batch)
            
            # calculate outer product of outputs projected to the unit circle (inner product of each pair of features), O(N^2)
            out = F.normalize(out, dim=-1)
            out = out @ out.T

            # loss = torch.linalg.norm(target - out)
            loss_mx = (target - out) ** 2
            loss = loss_mx.sum()

            # backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            pos_loss, neg_loss, inter_class, intra_class = loss_breakdown(loss_mx, target, labels, args.n_aug)

            # update the training bar
            total_num += data_tuple[0].size(0)
            total_loss += loss.item() * data_tuple[0].size(0)

            tot_pos += pos_loss / args.log_freq
            tot_neg += neg_loss / args.log_freq
            tot_inter += inter_class / args.log_freq
            tot_intra += intra_class / args.log_freq


            train_bar.set_description(
                "Train Epoch: [{}/{}] Loss: {:.1f}, pos: {:.1f}, neg: {:.1f} inter: {:.1f} intra: {:.1f}".format(
                    epoch, args.epochs, loss.item(), pos_loss.item(), neg_loss.item(), inter_class.item(), intra_class.item()
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
                        img_batch = einops.rearrange(img_batch.detach().cpu(), "(B N) C H W -> B N C H W", B=args.batch_size)
                        vis_dict = visualize_augmentations(vis_dict, img_batch)

                        assert not model.training
                        vis_dict["val_acc_1_out"], vis_dict["val_acc_5_out"] = test_one_epoch(model, memory_loader, test_loader, feat=False)
                        model.eval()

                        vis_dict["train_loss"] = total_loss / total_num
                        vis_dict["val_acc_1"] = acc_1
                        vis_dict["val_acc_5"] = acc_5
                        vis_dict["lr"] = scheduler.get_last_lr()[0]

                        vis_dict["pos_loss"] = pos_loss
                        vis_dict["neg_loss"] = neg_loss
                        vis_dict["inter_class_loss"] = inter_class
                        vis_dict["intra_class_loss"] = intra_class

                        wandb.log(vis_dict, step=total_steps)

                    assert not model.training
                    model.train()
                    
                    total_loss, total_num, vis_dict = 0.0, 0, {}
                    tot_pos, tot_neg, tot_inter, tot_intra = 0, 0, 0, 0

                    if epoch % args.save_freq == 0 or acc_1 == top_acc:
                        torch.save(
                            model.state_dict(),
                            f"{args.save_folder}/{args.dataset}_{args.n_aug}_{epoch}_acc_{acc_1:0.2f}.pth",
                        )

    if args.wandb:
        wandb.finish()
