
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


feat_cov_decomp_history = {}
def visualize_feature_cov_decomp(vis_dict, out, step, centered, prefix="feature", plot=True):
    pref = "centered_" if centered else "uncentered_"
    prefix = pref + prefix
    if prefix not in feat_cov_decomp_history:
        feat_cov_decomp_history[prefix] = {}
    with torch.no_grad():
        assert step not in feat_cov_decomp_history[prefix]

        if centered:
            cov = torch.cov(out.detach().T)
        else:
            cov = out.detach().T @ out.detach()
        feat_cov_decomp_history[prefix][step] = torch.linalg.eigvalsh(cov).cpu() 

        if plot:
            # Create a line plot for each eigenvalue over all steps
            plt.figure(figsize=(10, 6))
            steps = sorted(feat_cov_decomp_history[prefix].keys())
            n_eigenvals = len(feat_cov_decomp_history[prefix][steps[0]])
            eigenvals = np.zeros((len(steps), n_eigenvals))
            for i, step in enumerate(steps):
                eigenvals[i] = feat_cov_decomp_history[prefix][step].numpy()
            
            # Plot each eigenvalue as a separate line
            for i in range(n_eigenvals):
                plt.plot(steps, eigenvals[:, i]) #, label=f'λ{i+1}')
            
            plt.xlabel('Step')
            plt.ylabel('Eigenvalue')
            pref = "Centered " if centered else "Uncentered "
            plt.title(pref + 'Feature Covariance Matrix Eigenvalues')
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.yscale('log')
            plt.grid(True)
            plt.tight_layout()

            vis_dict[prefix + "_cov_eval"] = wandb.Image(plt)
            plt.close()
        return vis_dict



history = {}
def visualize_vector_time_series(vis_dict, out, step, prefix="", plot=True):
    if prefix not in history:
        history[prefix] = {}
    with torch.no_grad():
        assert step not in history[prefix]
        history[prefix][step] = out.detach().cpu() 

        if plot:
            # Create a line plot for each eigenvalue over all steps
            plt.figure(figsize=(10, 6))
            steps = sorted(history[prefix].keys())
            n_eigenvals = len(history[prefix][steps[0]])
            eigenvals = np.zeros((len(steps), n_eigenvals))
            for i, step in enumerate(steps):
                eigenvals[i] = history[prefix][step].numpy()
            
            # Plot each eigenvalue as a separate line
            for i in range(n_eigenvals):
                plt.plot(steps, eigenvals[:, i]) #, label=f'λ{i+1}')
            
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.title(prefix + ' Time Series')
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.yscale('log')
            plt.grid(True)
            plt.tight_layout()

            vis_dict[prefix + "_time_series"] = wandb.Image(plt)
            plt.close()
        return vis_dict


def visualize_centoid_sing_val_stats(vis_dict, step, centroids, sing_vals, plot=True):
    # plot number and stats of non-zero centroid covariance e'vals
    cov_matrix = centroids.T @ centroids
    evals = torch.linalg.eigvalsh(cov_matrix)
    nz_evals = evals[evals > 1e-4]
    vis_dict["centroid_eval_nnz"] = nz_evals.shape[0]
    vis_dict["centroid_eval_mean"] = nz_evals.mean()
    vis_dict["centroid_eval_var"] = nz_evals.var()

    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(cov_matrix, cmap='coolwarm', aspect='equal')
        plt.colorbar()
        plt.title('Centroid Covariance')
        vis_dict["centroid_cov_mx"] = wandb.Image(plt)
        plt.close()

    # plot centroid norms
    cnorms = torch.linalg.norm(centroids, dim=1)
    vis_dict = visualize_vector_time_series(vis_dict, cnorms, step, "centroid_norm", plot)

    # singular value stats
    vis_dict["sing_val_mean"] = sing_vals.mean()
    vis_dict["sing_val_mean"] = sing_vals.var()

    return vis_dict


cov_evec_decomp_history = {}
def visualize_cov_evec(vis_dict, out, step, centered, prefix="feature", plot=True, handle_inverted_evec=False):
    pref = "centered_" if centered else "uncentered_"
    prefix = pref + prefix
    if prefix not in cov_evec_decomp_history:
        cov_evec_decomp_history[prefix] = [{}, None]
    with torch.no_grad():
        assert step not in cov_evec_decomp_history[prefix][0]

        if centered:
            cov = torch.cov(out.detach().T)
        else:
            cov = out.detach().T @ out.detach()
        
        # NOTE: evec are sorted from smallest -> largest e'val   
        _, cur_evec = torch.linalg.eigh(cov)
        cur_evec = cur_evec.cpu()

        prev_evec = cov_evec_decomp_history[prefix][1]
        if prev_evec is not None:
            # evec are the columns --> rows of sim are the sim between a given e'vec and all evec of prev_evec
            sim = cur_evec.T @ prev_evec
            
            # NOTE: cosine similarity \in [-1, 1]. high negative cosin
            if handle_inverted_evec:
                sim = torch.abs(sim)

            # NOTE: probably a better way to do this, probably want unique assignments
            # cosine sim with closest prev evec (potentially have duplicates -> 2+ cur evec have the same "closest" prev evec)
            cov_evec_decomp_history[prefix][0][step] = torch.max(sim, dim=1)[0]

            # TODO: might be interesting to plot indices as well?

            if plot:
                # Create a line plot for each eigenvalue over all steps
                plt.figure(figsize=(10, 6))
                steps = sorted(cov_evec_decomp_history[prefix][0].keys())
                n_eigenvals = len(cov_evec_decomp_history[prefix][0][steps[0]])
                eigenvals = np.zeros((len(steps), n_eigenvals))
                for i, step in enumerate(steps):
                    eigenvals[i] = cov_evec_decomp_history[prefix][0][step].numpy()
                
                # Plot each eigenvector as a separate line
                for i in range(n_eigenvals):
                    plt.plot(steps, eigenvals[:, i]) #, label=f'λ{i+1}')
                
                plt.xlabel('Step')
                plt.ylabel("E'vec Cosine Sim. With Closest E'vec From Prior Step")
                pref = "Centered " if centered else "Uncentered "
                plt.title(pref + 'Feature Covariance Matrix Eigenvector Evolution')
                # plt.yscale('log')
                plt.grid(True)
                plt.tight_layout()
                vis_dict[prefix + "_cov_evec"] = wandb.Image(plt)
                plt.close()

                if prefix == "uncentered_out":
                    # plot top 20 e'val (sorted smallest -> largest by eval)
                    plt.figure(figsize=(10, 6))
                    for i in range(10):
                        plt.plot(steps, eigenvals[:, i], label=f'λ {i}')
                    plt.xlabel('Step')
                    plt.ylabel("Top E'vec Cosine Sim. With Closest E'vec From Prior Step")
                    pref = "Centered " if centered else "Uncentered "
                    plt.title(pref + 'Feature Covariance Matrix Eigenvector Evolution')
                    plt.grid(True)
                    plt.tight_layout()
                    vis_dict[prefix + "_small_cov_evec"] = wandb.Image(plt)
                    plt.close() 

                    # plot bottom 20 e'val (sorted smallest -> largest by eval)
                    plt.figure(figsize=(10, 6))
                    for i in range(1, 11, 1):
                        plt.plot(steps, eigenvals[:, -1 * i], label=f'λ -{i}')
                    plt.xlabel('Step')
                    plt.ylabel("Bottom E'vec Cosine Sim. With Closest E'vec From Prior Step")
                    pref = "Centered " if centered else "Uncentered "
                    plt.title(pref + 'Feature Covariance Matrix Eigenvector Evolution')
                    plt.grid(True)
                    plt.tight_layout()
                    vis_dict[prefix + "_large_cov_evec"] = wandb.Image(plt)
                    plt.close() 

        
        cov_evec_decomp_history[prefix][1] = cur_evec
        return vis_dict
