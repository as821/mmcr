import torch
import torchvision
from tqdm import tqdm
import einops
import wandb

from mmcr.cifar_stl.data import get_datasets, CifarBatchTransform
from mmcr.cifar_stl.models import Model
from mmcr.cifar_stl.knn import test_one_epoch
from mmcr.cifar_stl.loss_mmcr import MMCR_Loss, BatchFIFOQueue
from mmcr.cifar_stl.analysis import calc_manifold_subspace_alignment, visualize_augmentations


def train(args):
    if args.wandb:
        wandb.init(config=args, project="linear_contrastive")

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
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True, drop_last=True, prefetch_factor=4, persistent_workers=True
    )
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=128, shuffle=True, num_workers=16
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=16
    )
   

    # test set with training transformations
    stats_dset = torchvision.datasets.CIFAR10(root="./datasets/", train=False, download=True, transform=CifarBatchTransform(train_transform=True, batch_transform=True, n_transform=100))
    stats_loader = torch.utils.data.DataLoader(stats_dset, batch_size=500, shuffle=False, num_workers=12)
    stats_data = next(iter(stats_loader))



    # TODO(as) define a model, define a loss function....

   
    model = Model(projector_dims=[512, 128], dataset=args.dataset)


    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.wandb:
        wandb.watch(model, log_freq=10)

    model = model.cuda(non_blocking=True)
    model = torch.compile(model, mode="max-autotune")
    top_acc = 0.0
    for epoch in range(args.epochs):
        total_loss, total_num, train_bar, vis_dict = 0.0, 0, tqdm(train_loader), {}
        for step, data_tuple in enumerate(train_bar):
            optimizer.zero_grad()

            # forward pass
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            img_batch, labels = data_tuple
            img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W").cuda(non_blocking=True)
            if args.diffusion_aug:
                rnd = (torch.randn_like(img_batch) * cifar_std) + cifar_mean
                # print(f"{img_batch.min()} {img_batch.max()} {rnd.min()} {rnd.max()}")
                # img_batch = (1 - args.diff_alpha) * img_batch + args.diff_alpha * rnd
                img_batch += args.diff_alpha * rnd
                # print(f"\t{img_batch.min()} {img_batch.max()}")

            features, out = model(img_batch)
            loss, loss_dict = loss_function(out.float())

            # backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # update the training bar
            total_num += data_tuple[0].size(0)
            total_loss += loss.item() * data_tuple[0].size(0)
            train_bar.set_description(
                "Train Epoch: [{}/{}] Loss: {:.4f}".format(
                    epoch, args.epochs, loss.item()
                )
            )

        if epoch % 1 == 0:
            with torch.no_grad():
                acc_1, acc_5 = test_one_epoch(
                    model,
                    memory_loader,
                    test_loader,
                )
                if acc_1 > top_acc:
                    top_acc = acc_1

                if args.wandb:
                    vis_dict["train_loss"] = total_loss / total_num
                    vis_dict["val_acc_1"] = acc_1
                    vis_dict["val_acc_5"] = acc_5
                    vis_dict["lr"] = scheduler.get_last_lr()[0]
                    wandb.log(vis_dict, step=epoch)


                if epoch % args.save_freq == 0 or acc_1 == top_acc:
                    torch.save(
                        model.state_dict(),
                        f"{args.save_folder}/{args.dataset}_{args.n_aug}_{epoch}_acc_{acc_1:0.2f}.pth",
                    )

    if args.wandb:
        wandb.finish()
