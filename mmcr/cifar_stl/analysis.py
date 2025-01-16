
import torch 
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops



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

def loss_breakdown(loss_mx, target, labels, n_aug):
    with torch.no_grad():
        batch_sz = labels.shape[0]
        tot_aug = batch_sz * n_aug
        loss_mx = einops.rearrange(loss_mx, "(A B) -> A B", A=tot_aug)
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




