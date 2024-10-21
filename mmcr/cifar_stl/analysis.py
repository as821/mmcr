
import torch 
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt



def calc_manifold_subspace_alignment(vis_dict, model, data_tuple):
    model.eval()
    with torch.no_grad():        
        # 100 samples from the augmentation manfiolds of 500 images in the CIFAR-10
        data, target = data_tuple
        features = torch.zeros((data.shape[0], data.shape[1], 512), dtype=data.dtype, device="cuda")
        centroids = torch.zeros((data.shape[0], 512), dtype=data.dtype, device="cuda")
        aug_centroid_sim = torch.zeros((data.shape[0], data.shape[1]), device="cpu")
        for idx in range(data.shape[0]):
            feat, out = model(data[idx].cuda(non_blocking=True))
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
        vis_dict["intra_class_centroid"] = same_class_sims.mean()
        vis_dict["intra_class_centroid_dist"] = wandb.Image(plt)
        plt.close()

        other_class_sims = centroid_sim[~same_class_mask].numpy().flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(other_class_sims, bins=50, edgecolor='black')
        plt.title('Inter-Class Cosine Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        vis_dict["inter_class_centroid"] = other_class_sims.mean()
        vis_dict["inter_class_centroid_dist"] = wandb.Image(plt)
        plt.close()


        aug_centroid_sim = aug_centroid_sim.numpy().flatten()

        print(f"{aug_centroid_sim.min()} {aug_centroid_sim.max()} {aug_centroid_sim.mean()}")

        plt.figure(figsize=(10, 6))
        plt.hist(aug_centroid_sim, bins=50, edgecolor='black')
        plt.title('Augmentation-Centroid Cosine Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        vis_dict["aug_centroid_sim"] = aug_centroid_sim.mean()
        vis_dict["aug_centroid_sim_dist"] = wandb.Image(plt)
        plt.close()

    return vis_dict




