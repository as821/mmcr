
import torch 
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb

def calc_manifold_subspace_alignment(vis_dict, model, data_tuple, use_feat, out_dim=-1):
    prefix = "feat_" if use_feat else "out_"
    with torch.no_grad():        
        # 100 samples from the augmentation manfiolds of 500 images in the CIFAR-10
        data, target = data_tuple

        features = torch.zeros((data.shape[0], data.shape[1], out_dim), dtype=data.dtype, device="cuda")
        centroids = torch.zeros((data.shape[0], out_dim), dtype=data.dtype, device="cuda")
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


        # aug_centroid_sim = aug_centroid_sim.numpy().flatten()
        # plt.figure(figsize=(10, 6))
        # plt.hist(aug_centroid_sim, bins=50, edgecolor='black')
        # plt.title('Augmentation-Centroid Cosine Similarities')
        # plt.xlabel('Cosine Similarity')
        # plt.ylabel('Frequency')
        # vis_dict[prefix + "aug_centroid_sim"] = aug_centroid_sim.mean()
        # vis_dict[prefix + "aug_centroid_sim_dist"] = wandb.Image(plt)
        # plt.close()

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

def calc_aug_deviation(model, x, aug, device, naug=100):
    # Compare embedding of an image to that of its augmentations
    with torch.no_grad():
        model = model.to(device)
        orig_embed = model(x.unsqueeze(0))[1].squeeze()
        mean_embed, orig_dist = torch.zeros_like(orig_embed), 0
        
        augs = torch.zeros((naug, x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype, device=x.device)
        for idx in range(naug):
            augs[idx] = aug.generate_random_sample(x)
        embed = model(augs)[1]
        
        # distance of mean embedding from original, mean distance of an embedding from original
        orig_dist = torch.linalg.norm(orig_embed - embed, dim=1).mean()
        mean_embed_dist = torch.linalg.norm(orig_embed - embed.mean(dim=0))
        return mean_embed_dist, orig_dist

def batch_calc_aug_deviation(model, x, aug, device):
    mean_mean_embed_dist, mean_mean_orig_dist = 0, 0
    for idx in range(x.shape[0]):
        e, o = calc_aug_deviation(model, x[idx], aug, device)
        mean_mean_embed_dist += e
        mean_mean_orig_dist += o
    mean_mean_orig_dist /= x.shape[0]
    mean_mean_embed_dist /= x.shape[0]
    return mean_mean_embed_dist, mean_mean_orig_dist


def output_dim_stats(model, x, device):
    # Return covariance matrix for the output dimensions of the network
    embed = model(x.to(device))
    
    pdb.set_trace()         # TODO: check cov dimensions
    return torch.cov(embed)


