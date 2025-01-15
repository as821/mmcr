import torch
import torchvision
from tqdm import tqdm
import einops
import wandb
import pdb
import time
import numpy as np

from mmcr.cifar_stl.data import get_datasets, CifarBatchTransform
from mmcr.cifar_stl.models import Model
from mmcr.cifar_stl.knn import test_one_epoch
from mmcr.cifar_stl.analysis import calc_manifold_subspace_alignment, batch_calc_aug_deviation, output_dim_stats, calc_aug_deviation_jacobian
from mmcr.cifar_stl.augment import loss_function, log_model_jacobian, generate_aug_probs, calc_aug_ev_var, calc_aug_var_decomp


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

    torch.backends.cuda.preferred_linalg_library(backend="cusolver")

    # NOTE: force single "augmentation", actually just transforms to Tensor + normalizes
    args.n_aug = 1

    train_dataset, memory_dataset, test_dataset = get_datasets(
        dataset=args.dataset, n_aug=args.n_aug, strong_aug=args.stronger_aug, diffusion_aug=args.diffusion_aug, weak_aug=args.weak_aug, strongest_aug=args.strongest_aug, aug_var_root=args.aug_var_root
    )
    model = Model(projector_dims=[512, args.output_dim], dataset=args.dataset)

    n_workers = 16 if torch.cuda.is_available() else 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_workers, drop_last=True, pin_memory=False #, prefetch_factor=4, persistent_workers=True
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=30),     # Linear warmup for 10 steps
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=args.final_lr)
    ])
    scaler = torch.amp.GradScaler(device)

    # TODO: debugging!! try to overfit on a single batch
    # data = next(iter(train_loader))
    
    if args.wandb:
        wandb.watch(model, log_freq=1)

    c, h, w = 3, 32, 32
    aug_prob_map = generate_aug_probs([c, h, w])
    model = model.to(dtype)
    model = model.to(device, non_blocking=True)
    # model = torch.compile(model, mode="max-autotune")
    
    intermediate_cpu = torch.zeros((args.batch_size, c, h * w, h * w), dtype=dtype).pin_memory()
    intermediate = torch.zeros((args.batch_size, c * h * w, c * h * w), dtype=dtype, device=device)

    top_acc = 0.0
    total_step = 0
    total_loss = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_num, train_bar, vis_dict = 0, tqdm(train_loader), {}
        for step, data_tuple in enumerate(train_bar):
            optimizer.zero_grad(set_to_none=True)

            # forward pass
            if args.aug_var_root != "":
                # load pre-computed augmentation variance decomposition
                # img_batch, labels, intermediate = data_tuple
                # intermediate = intermediate.to(device, non_blocking=True)

                img_batch, labels, indices = data_tuple
                for idx in range(indices.shape[0]):
                    intermediate_cpu[idx] = torch.load(args.aug_var_root + "/" + str(indices[idx].item()) + ".pt")
                img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W").to(device, non_blocking=True)
            else:
                img_batch, labels = data_tuple
                img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W").to(device, non_blocking=True)
                intermediate_cpu = calc_aug_var_decomp(img_batch, aug_prob_map)

            # TODO: this is stupid and unnecessary, remove this requirement

            # intermediate is currently per-channel but needs to be block diagonal instead
            for idx in range(c):
                start, end = idx * h * w, (idx + 1) * h * w    
                intermediate[:, start:end, start:end] = intermediate_cpu[:, idx]

            img_batch = img_batch.to(dtype)
            loss, loss_dict = loss_function(img_batch, model, intermediate)

            # update the training bar
            total_num += data_tuple[0].size(0)
            total_loss += loss.item() * data_tuple[0].size(0)
            train_bar.set_description(
                "Train Epoch: [{}/{}] Loss: {:.6f}".format(
                    epoch, args.epochs, loss.item()
                )
            )

            if dtype == torch.float32:
                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                loss = loss.float()
                scaler.scale(loss).backward()

                # AMP does not work properly with vmap(jacrev) code for some reason. Still want to scale losses and parameter data/grad need to be in fp32
                model = model.to(torch.float32)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # return to float16 after the update
                model = model.to(torch.float16)

            # with torch.no_grad():
            #     model.eval()
            #     mean_dist, mean_norm = calc_aug_deviation_jacobian(model, img_batch, aug_prob_map["rc"], device, jac_norm=False)
            #     mean_norm_norm = calc_aug_deviation_jacobian(model, img_batch, aug_prob_map["rc"], device, jac_norm=True)
            #     print(f"\n\n{mean_dist_norm} {mean_norm_norm}\n")
            #     mean_dist, orig_dist = batch_calc_aug_deviation(model, img_batch, aug_prob_map["rc"], device)
            #     print(f"\t{mean_dist} {orig_dist}")
            #     output_dim_stats(model, img_batch, device, {}, "foo")
            #     model.train()

            if total_step % args.log_freq == 0 and total_step != 0:
                with torch.no_grad():
                    model.eval()
                    model = model.float()
                    acc_1, acc_5 = test_one_epoch(model, memory_loader, test_loader, feat=False)
                    if acc_1 > top_acc:
                        top_acc = acc_1
                    model.eval()


                    if args.wandb:
                        model.eval()

                        # vis class-level clustering on feature + output levels
                        vis_dict = calc_manifold_subspace_alignment(vis_dict, model, stats_tuple, True, 512)
                        vis_dict = calc_manifold_subspace_alignment(vis_dict, model, stats_tuple, False, args.output_dim)
                        
                        # track norm of the model Jacobian (across augmentations of the test set) to detect collapse
                        vis_dict = log_model_jacobian(vis_dict, stats_data, model, device)

                        # log output dimension var/cov
                        # vis_dict = output_dim_stats(model, img_batch, device, vis_dict, "train_out")
                        # vis_dict = output_dim_stats(model, stats_data, device, vis_dict, "test_out")
                        # vis_dict = output_dim_stats(model, img_batch, device, vis_dict, "train_out_norm", normalize=True)
                        # vis_dict = output_dim_stats(model, stats_data, device, vis_dict, "test_out_norm", normalize=True)


                        # calculate augmentation embedding deviation from source image
                        vis_dict["test_mean_dist"], vis_dict["test_orig_dist"] = batch_calc_aug_deviation(model, stats_data, aug_prob_map["rc"], device)
                        vis_dict["train_mean_dist"], vis_dict["train_orig_dist"] = batch_calc_aug_deviation(model, img_batch, aug_prob_map["rc"], device)
                        
                        vis_dict["test_mean_dist_norm"], vis_dict["test_orig_dist_norm"] = batch_calc_aug_deviation(model, stats_data, aug_prob_map["rc"], device, normalize=True)
                        vis_dict["train_mean_dist_norm"], vis_dict["train_orig_dist_norm"] = batch_calc_aug_deviation(model, img_batch, aug_prob_map["rc"], device, normalize=True)

                        # calculate augmentation Jacobian embedding deviation (distance from original image embedding, norm of "Jac.norm() @ aug.T")
                        vis_dict["train_jac_aug_emb_norm"] = calc_aug_deviation_jacobian(model, img_batch, aug_prob_map["rc"], device, jac_norm=True, aug_norm=True)
                        vis_dict["test_jac_aug_emb_norm"] = calc_aug_deviation_jacobian(model, stats_data[:100], aug_prob_map["rc"], device, jac_norm=True, aug_norm=True)

                        assert not model.training
                        feat_acc_1, feat_acc_5 = test_one_epoch(model, memory_loader, test_loader, feat=True)
                        model.eval()

                        vis_dict["std_loss"] = loss_dict["std_loss"]
                        vis_dict["cov_loss"] = loss_dict["cov_loss"]
                        vis_dict["tangent_loss"] = loss_dict["tangent"]
                        vis_dict["train_jac_norm"] = loss_dict["jac_norm"]
                        vis_dict["jac_model_align_loss"] = loss_dict["jac_model_align_loss"]
                        # vis_dict["jac_aug_norm_loss"] = loss_dict["jac_aug_norm_loss"]
                        # vis_dict["jac_norm_loss"] = loss_dict["jac_norm_loss"]
                        vis_dict["train_loss"] = total_loss / total_num
                        vis_dict["val_acc_1"] = acc_1
                        vis_dict["val_acc_5"] = acc_5
                        vis_dict["val_acc_1_feat"] = feat_acc_1
                        vis_dict["val_acc_5_feat"] = feat_acc_5
                        vis_dict["lr"] = scheduler.get_last_lr()[0]
                        wandb.log(vis_dict, step=total_step)
                        
                        assert not model.training
                        model.train()


                    if total_step % (args.log_freq * args.save_freq) == 0 or acc_1 == top_acc:
                        torch.save(
                            model.state_dict(),
                            f"{args.save_folder}/{args.dataset}_{args.n_aug}_{total_step}_acc_{acc_1:0.2f}.pth",
                        )
                    # model = model.to(torch.float16)
                total_loss = 0
            total_step += 1


        # TODO(as) implement more complete augmentations

        # TODO(as) try anti-collapse

        # TODO(as) does still sampling from the augmentation distribution help performance? (!!)

        # TODO(as) different options for removing batchnorms? (should we be using JAX instead?)


    if args.wandb:
        wandb.finish()
