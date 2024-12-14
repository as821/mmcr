import torch
import torchvision
from tqdm import tqdm
import einops
import wandb
import pdb

from mmcr.cifar_stl.data import get_datasets, CifarBatchTransform
from mmcr.cifar_stl.models import Model
from mmcr.cifar_stl.knn import test_one_epoch
from mmcr.cifar_stl.analysis import visualize_augmentations, calc_manifold_subspace_alignment
from mmcr.cifar_stl.augment import loss_function, log_model_jacobian, generate_aug_probs


def train(args):
    if args.wandb:
        wandb.init(config=args, project="mmcr", entity="cmu-slots-group")

    def vis_dist(key_name, prefix, vis_dict, loss_dict):
        foo = loss_dict[key_name]
        vis_dict[prefix + "_min"] = foo.min()
        vis_dict[prefix + "_max"] = foo.max()
        vis_dict[prefix + "_mean"] = foo.mean()
        vis_dict[prefix] = wandb.Histogram(foo)
        return vis_dict

    torch.set_float32_matmul_precision('high')

    # NOTE: force single "augmentation", actually just transforms to Tensor + normalizes
    args.n_aug = 1

    train_dataset, memory_dataset, test_dataset = get_datasets(
        dataset=args.dataset, n_aug=args.n_aug, strong_aug=args.stronger_aug, diffusion_aug=args.diffusion_aug, weak_aug=args.weak_aug, strongest_aug=args.strongest_aug
    )
    model = Model(projector_dims=[512, args.output_dim], dataset=args.dataset)

    n_workers = 16 if torch.cuda.is_available() else 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_workers #, pin_memory=True, drop_last=True, prefetch_factor=4, persistent_workers=True
    )
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=128, shuffle=True, num_workers=n_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=n_workers
    )

    # test set with training transformations
    stats_dset = torchvision.datasets.CIFAR10(root="./datasets/", train=False, download=True, transform=CifarBatchTransform(train_transform=True, batch_transform=True, n_transform=10))
    stats_loader = torch.utils.data.DataLoader(stats_dset, batch_size=128, shuffle=False, num_workers=12)
    stats_tuple = next(iter(stats_loader))
    stats_data = stats_tuple[0].flatten(0, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=args.final_lr)

    if args.wandb:
        wandb.watch(model, log_freq=10)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    aug_prob_map = generate_aug_probs([3, 32, 32], device)
    model = model.to(device, non_blocking=True)
    model = torch.compile(model, mode="max-autotune")
    top_acc = 0.0
    total_step = 0
    total_loss = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_num, train_bar, vis_dict = 0, tqdm(train_loader), {}
        for step, data_tuple in enumerate(train_bar):
            optimizer.zero_grad()

            # forward pass
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            img_batch, labels = data_tuple
            img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W").to(device, non_blocking=True)
            loss, loss_dict = loss_function(img_batch, model, aug_prob_map)

            # update the training bar
            total_num += data_tuple[0].size(0)
            total_loss += loss.item() * data_tuple[0].size(0)
            train_bar.set_description(
                "Train Epoch: [{}/{}] Loss: {:.6f}".format(
                    epoch, args.epochs, loss.item()
                )
            )

            # backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            if total_step % args.log_freq == 0:
                with torch.no_grad():
                    model.eval()
                    acc_1, acc_5 = test_one_epoch(
                        model,
                        memory_loader,
                        test_loader,
                    )
                    if acc_1 > top_acc:
                        top_acc = acc_1

                    if args.wandb:
                        # TODO(as): visualize mean aug
                        # visualize augmentations
                        # img_batch = einops.rearrange(img_batch.detach().cpu(), "(B N) C H W -> B N C H W", B=args.batch_size)
                        pair = torch.concat([img_batch.detach().cpu().unsqueeze(1), loss_dict["aug_ev"]], dim=1)
                        vis_dict = visualize_augmentations(vis_dict, pair)

                        # vis class-level clustering on feature + output levels
                        vis_dict = calc_manifold_subspace_alignment(vis_dict, model, stats_tuple, True, 512)
                        vis_dict = calc_manifold_subspace_alignment(vis_dict, model, stats_tuple, False, args.output_dim)
                        
                        # track norm of the model Jacobian (across augmentations of the test set) to detect collapse
                        vis_dict = log_model_jacobian(vis_dict, stats_data, model, device)

                        vis_dict["svd_loss"] = loss_dict["svd"]
                        vis_dict["tangent_loss"] = loss_dict["tangent"]
                        vis_dict["train_loss"] = total_loss / total_num
                        vis_dict["val_acc_1"] = acc_1
                        vis_dict["val_acc_5"] = acc_5
                        vis_dict["lr"] = scheduler.get_last_lr()[0]
                        wandb.log(vis_dict, step=total_step)
                    model.train()


                    if total_step % (args.log_freq * args.save_freq) == 0 or acc_1 == top_acc:
                        torch.save(
                            model.state_dict(),
                            f"{args.save_folder}/{args.dataset}_{args.n_aug}_{total_step}_acc_{acc_1:0.2f}.pth",
                        )
                total_loss = 0
            total_step += 1


        # TODO(as) implement more complete augmentations

        # TODO(as) try anti-collapse

        # TODO(as) does still sampling from the augmentation distribution help performance? (!!)

        # TODO(as) different options for removing batchnorms? (should we be using JAX instead?)


    if args.wandb:
        wandb.finish()
