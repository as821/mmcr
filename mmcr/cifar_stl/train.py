import torch
import torchvision
from tqdm import tqdm
import einops
import wandb

from mmcr.cifar_stl.data import get_datasets, CifarBatchTransform
from mmcr.cifar_stl.models import Model
from mmcr.cifar_stl.knn import test_one_epoch
from mmcr.cifar_stl.loss_mmcr import MMCR_Loss
from mmcr.cifar_stl.analysis import calc_manifold_subspace_alignment


def train(
    dataset: str,
    n_aug: int,
    batch_size: int,
    lr: float,
    final_lr: float,
    epochs: int,
    lmbda: float,
    save_folder: str,
    save_freq: int,
    enable_wandb: bool, 
    weight_decay: float
):

    if enable_wandb:
        wandb.init(config={
            "dataset":dataset,
            "n_aug": n_aug,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "lmbda": lmbda,
        }, project="mmcr")

    torch.set_float32_matmul_precision('high')

    train_dataset, memory_dataset, test_dataset = get_datasets(
        dataset=dataset, n_aug=n_aug
    )
    model = Model(projector_dims=[512, 128], dataset=dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = MMCR_Loss(lmbda=lmbda, n_aug=n_aug, distributed=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=final_lr)

    if enable_wandb:
        wandb.watch(model, log_freq=10)

    model = model.cuda()
    model = torch.compile(model, mode="max-autotune")
    top_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, total_num, train_bar, vis_dict = 0.0, 0, tqdm(train_loader), {}
        for step, data_tuple in enumerate(train_bar):
            optimizer.zero_grad()

            # forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                img_batch, labels = data_tuple
                img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W")
                features, out = model(img_batch.cuda(non_blocking=True))
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
                    epoch, epochs, total_loss / total_num
                )
            )

        if epoch % 1 == 0:
            acc_1, acc_5 = test_one_epoch(
                model,
                memory_loader,
                test_loader,
            )
            if acc_1 > top_acc:
                top_acc = acc_1

            if enable_wandb:
                # check manifold subspace alignment 
                vis_dict = calc_manifold_subspace_alignment(vis_dict, model, stats_data)

                vis_dict["train_loss"] = total_loss / total_num
                vis_dict["val_acc_1"] = acc_1
                vis_dict["val_acc_5"] = acc_5
                vis_dict["lr"] = scheduler.get_last_lr()[0]
                wandb.log(vis_dict, step=epoch)


            if epoch % save_freq == 0 or acc_1 == top_acc:
                torch.save(
                    model.state_dict(),
                    f"{save_folder}/{dataset}_{n_aug}_{epoch}_acc_{acc_1:0.2f}.pth",
                )

    if enable_wandb:
        wandb.finish()
