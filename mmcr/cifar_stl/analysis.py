
import torch 
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops

import pdb
import numpy as np


def calc_manifold_subspace_alignment(vis_dict, model, data_tuple, use_feat):
    prefix = "feat_" if use_feat else "out_"
    with torch.no_grad():        
        # 100 samples from the augmentation manfiolds of 500 images in the CIFAR-10
        data, target = data_tuple

        sz = 512 if use_feat else 128
        features = torch.zeros((data.shape[0], data.shape[1], sz), dtype=data.dtype, device="cuda")
        centroids = torch.zeros((data.shape[0], sz), dtype=data.dtype, device="cuda")
        aug_centroid_sim = torch.zeros((data.shape[0], data.shape[1]), device="cpu")
        for idx in range(data.shape[0]):
            feat, out = model(data[idx].cuda(non_blocking=True))
            if not use_feat:
                feat = out
                feat = F.normalize(feat, dim=-1)

            # calculate the centroid of this image manifold
            centroid = feat.mean(dim=0)
            centroids[idx] = centroid

            # cosine sim. of each augmentation to the centroid
            aug_centroid_sim[idx] = F.cosine_similarity(centroid.unsqueeze(0), feat, dim=1).cpu()

        # cosine similarity of centroid for images of the same class vs. different class
        centroid_sim = F.cosine_similarity(centroids.unsqueeze(0), centroids.unsqueeze(1), dim=2).cpu()
        same_class_mask = target.unsqueeze(0) == target.unsqueeze(1)

        same_class_sims = centroid_sim[same_class_mask].numpy().flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(same_class_sims, bins=50, edgecolor='black')
        plt.title('Intra-Class Cosine Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        vis_dict[prefix + "intra_class_centroid"] = same_class_sims.mean()
        vis_dict[prefix + "intra_class_centroid_dist"] = wandb.Image(plt)
        plt.close()

        other_class_sims = centroid_sim[~same_class_mask].numpy().flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(other_class_sims, bins=50, edgecolor='black')
        plt.title('Inter-Class Cosine Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        vis_dict[prefix + "inter_class_centroid"] = other_class_sims.mean()
        vis_dict[prefix + "inter_class_centroid_dist"] = wandb.Image(plt)
        plt.close()


        aug_centroid_sim = aug_centroid_sim.numpy().flatten()

        plt.figure(figsize=(10, 6))
        plt.hist(aug_centroid_sim, bins=50, edgecolor='black')
        plt.title('Augmentation-Centroid Cosine Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        vis_dict[prefix + "aug_centroid_sim"] = aug_centroid_sim.mean()
        vis_dict[prefix + "aug_centroid_sim_dist"] = wandb.Image(plt)
        plt.close()

    return vis_dict


def visualize_augmentations(vis_dict, tensor):
    N, B, C, H, W = tensor.shape
    
    # Create a figure with N rows and B columns
    fig, axes = plt.subplots(N, B, figsize=(20, 20))

    # denormalize images
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 1, 3, 1, 1)
    tensor = tensor * std + mean

    # Normalize to [0, 1] if not already
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    for n in range(N):
        for b in range(B):
            img = tensor[n, b]
            img = img.permute(1, 2, 0)  # RGB image
            axes[n, b].imshow(img)
            axes[n, b].axis('off')
    
    
    plt.tight_layout()
    vis_dict["augmentations"] = wandb.Image(plt)
    plt.close()
    return vis_dict

def loss_breakdown(loss_mx, labels):
    with torch.no_grad():
        batch_sz = labels.shape[0]
        loss_mx = einops.rearrange(loss_mx, "(A B) (C D) -> A C (B D)", A=batch_sz, C=batch_sz)
        loss_mx = loss_mx.sum(dim=-1)

        # breakdown loss by postive/negative samples
        pos_loss = loss_mx.diag().sum()
        loss_mx.fill_diagonal_(0)
        neg_loss = loss_mx.sum()

        # breakdown negative loss by inter/intra class
        intra_class_mask = labels.unsqueeze(-1) == labels.unsqueeze(0)
        intra_class = loss_mx[intra_class_mask].sum()
        inter_class = loss_mx[~intra_class_mask].sum()

    return pos_loss, neg_loss, inter_class, intra_class

def log_pos_neg_sample_embedding(args, vis_dict, out, labels):
    # Plot distribution of positive/negative sample embedding similarities
    # NOTE: can be very slow, run infrequently
    with torch.no_grad():
        out_mx = einops.rearrange(out, "(A B) (C D) -> A C B D", A=args.batch_size, C=args.batch_size)
        

        # positive embeddding similarities (remove self-similarity)
        pos_idx = torch.arange(out_mx.shape[0])
        pos = out_mx[pos_idx, pos_idx, :]
        mask = ~torch.eye(pos.shape[1], dtype=bool)
        pos_no_diag = pos.permute((1, 2, 0))[mask].flatten().cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(pos_no_diag, bins=50, edgecolor='black')
        plt.title('Positive Sample Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        vis_dict["pos_sim_hist"] = wandb.Image(plt)
        plt.close()

        # negative sample similarities
        mask = ~torch.eye(out_mx.shape[0], dtype=bool)
        neg = out_mx[mask]
        plt.figure(figsize=(10, 6))
        plt.hist(neg.cpu().numpy().flatten(), bins=50, edgecolor='black')
        plt.title('Negative Sample Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        vis_dict["neg_sim_hist"] = wandb.Image(plt)
        plt.close()

        # inter/intra class similarities
        intra_class_mask = labels.unsqueeze(-1) == labels.unsqueeze(0)
        intra = out_mx[intra_class_mask & mask]
        inter = out_mx[~intra_class_mask & mask]

        plt.figure(figsize=(10, 6))
        plt.hist(intra.cpu().numpy().flatten(), bins=50, edgecolor='black')
        plt.title('(Intra) Negative Sample Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        vis_dict["neg_intra_sim_hist"] = wandb.Image(plt)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(inter.cpu().numpy().flatten(), bins=50, edgecolor='black')
        plt.title('(Inter) Negative Sample Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        vis_dict["neg_inter_sim_hist"] = wandb.Image(plt)
        plt.close()

        return vis_dict





feat_cov_decomp_history = {}
def visualize_feature_cov_decomp(vis_dict, out, step, prefix="feature"):
    with torch.no_grad():
        assert step not in feat_cov_decomp_history
        feat_cov_decomp_history[step] = torch.linalg.eigvalsh(torch.cov(out.detach().T)).cpu() 

        # Create a line plot for each eigenvalue over all steps
        plt.figure(figsize=(10, 6))
        steps = sorted(feat_cov_decomp_history.keys())
        n_eigenvals = len(feat_cov_decomp_history[steps[0]])
        eigenvals = np.zeros((len(steps), n_eigenvals))
        for i, step in enumerate(steps):
            eigenvals[i] = feat_cov_decomp_history[step].numpy()
        
        # Plot each eigenvalue as a separate line
        for i in range(n_eigenvals):
            plt.plot(steps, eigenvals[:, i]) #, label=f'λ{i+1}')
        
        plt.xlabel('Step')
        plt.ylabel('Eigenvalue')
        plt.title('Feature Covariance Matrix Eigenvalues')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()

        vis_dict[prefix + "_cov_eval"] = wandb.Image(plt)
        plt.close()
        return vis_dict

