import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import random
from typing import Tuple
import math
import sys
import pdb


class BatchFIFOQueue():
    def __init__(self, n_batches, batch_sz, feature_dim=128):
        # Implement FIFO queue as a circular buffer
        self.buf = torch.zeros((n_batches * batch_sz, feature_dim), device="cuda")     # assumes constant batch size
        self.batch_sz = batch_sz
        self.n_batches = n_batches
        self.ptr = 0
        self.sz = 0         # queue is "warm" once this == n_batches

    def enqueue(self, batch):
        assert batch.shape[0] == self.batch_sz
        start = self.ptr * self.batch_sz
        end = (self.ptr + 1) * self.batch_sz
        # print(f"{self.buf.shape} {batch.shape} {self.buf[start : end, :].shape} {start} {end}")
        self.buf[start : end, :] = batch
        self.ptr = (self.ptr + 1) % self.n_batches
        if self.sz < self.n_batches:
            self.sz += 1
            if self.sz == self.n_batches:
                print(f"Batch queue is warm after {self.sz} batches.")

    def is_warm(self):
        return self.sz == self.n_batches


class GradientPreconditioning(torch.autograd.Function):
    """Custom autograd function for gradient preconditioning."""
    
    @staticmethod
    def forward(ctx, embeddings, alpha, thresh=-1, power=0, centered=False):
        """
        Forward pass stores embeddings for backward pass.
        Args:
            embeddings: Tensor of shape (2n, d) containing the embeddings
            alpha: Small positive constant for numerical stability
        """
        ctx.alpha = alpha
        ctx.save_for_backward(embeddings)
        ctx.power = power
        ctx.thresh = thresh
        ctx.centered = centered
        return embeddings

    @staticmethod
    def backward(ctx, grad_output):
        """
        Implements the gradient preconditioning:
        grad_new = grad_old @ (F^T F + alpha*I)^(-1)
        
        Args:
            grad_output: Original gradient of shape (2n, d)
        Returns:
            Preconditioned gradient
        """
        embeddings, = ctx.saved_tensors
        alpha = ctx.alpha
        power = ctx.power
        thresh = ctx.thresh
        
        if ctx.centered:
            cov_matrix = torch.cov(embeddings.T)
        else:
            cov_matrix = torch.mm(embeddings.t(), embeddings)
        
        # Add alpha * I for stability
        n_dim = cov_matrix.shape[0]
        cov_matrix.add_(alpha * torch.eye(n_dim, device=cov_matrix.device))

        if power > 0:
            # assert power <= 1, "Power >1 will increase the influence of e'val not reduce it"
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

            powered_eigenvalues = eigenvalues.pow(power)
            inv_powered_eigenvalues = 1.0 / powered_eigenvalues
            
            if thresh > 0:
                # apply a minimum scaling to all e'val (ex. set to 1 to only scale up e'vals and never scale down any of them)
                inv_powered_eigenvalues = inv_powered_eigenvalues.clamp(thresh)
            
            inv_cov = eigenvectors @ torch.diag(inv_powered_eigenvalues) @ eigenvectors.t()
            # print(f"({eigenvalues.max()} {eigenvalues.min()} {eigenvalues.mean()}) -> ({inv_powered_eigenvalues.max()} {inv_powered_eigenvalues.min()} {inv_powered_eigenvalues.mean()})")
        else:
            # Compute inverse
            inv_cov = torch.linalg.inv(cov_matrix)
            assert thresh < 0
        
            # e_val = torch.linalg.eigvalsh(cov_matrix)
            # print(f"{e_val.max()} {e_val.min()} {e_val.mean()}")

        # Apply preconditioning: grad_new = grad_old @ (F^T F + alpha*I)^(-1)
        preconditioned_grad = torch.mm(grad_output, inv_cov)
        
        return preconditioned_grad, None, None, None, None

class MMCR_Loss(nn.Module):
    def __init__(self, lmbda: float, n_aug: int, distributed: bool = False, memory_bank=None, l2_spectral_norm=False, spectral_target=False, spectral_topk=False, huber=False, huber_pow=0, sv_pow=0, centroid_dropout_prob=0, pca_dropout=0):
        super(MMCR_Loss, self).__init__()
        self.lmbda = lmbda
        self.n_aug = n_aug
        self.distributed = distributed
        self.first_time = True
        self.l2_spectral_norm = l2_spectral_norm
        self.spectral_target = spectral_target
        self.spectral_topk = spectral_topk
        self.huber = huber
        self.huber_pow = huber_pow
        self.sv_pow = sv_pow
        self.centroid_dropout_prob = centroid_dropout_prob
        self.pca_dropout = pca_dropout

        self.memory_bank = memory_bank

    def forward(self, z: Tensor, args) -> Tuple[Tensor, dict]:
        # print(f"{z.max()} {z.min()}")
        
        z = F.normalize(z, dim=-1)
        # z_local_ = einops.rearrange(z, "(B N) C -> B C N", N=self.n_aug)
        z_local = einops.rearrange(z, "B N C -> B C N")


        centroids = torch.mean(z_local, dim=-1)
        centroids_pre_condition = centroids.detach().cpu()

        global_sing_vals = torch.linalg.svdvals(centroids)
        
        global_nuc = global_sing_vals.sum()

        batch_size = z_local.shape[0]
        loss = -1 * global_nuc

        loss_dict = {
            "loss": loss.item(),
            "global_nuc": global_nuc.item(),
            "global_sing_vals" : global_sing_vals.detach().cpu(), 
            "centroid_post_conditioner" : centroids.detach().cpu(),
            "centroid_pre_conditioner" : centroids_pre_condition
        }
        return loss, loss_dict





class VICReg_Loss(nn.Module):
    def __init__(self, sim_coeff=25, std_coeff=25, cov_coeff=1):
        super(VICReg_Loss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    # def calc_neighbor_similarity(self, z):
        # calculate patch neighbor similarity 
        # n_patches = z.shape
        # n_patch_per_dim = int(math.sqrt(n_patches))
        # assert n_patch_per_dim ** 2 == n_patches
        # einops.rearrange(z, "A (B C) D -> A B C D", B=n_patch_per_dim)

        # treats all patches in the image as neighbors
        # patch_mean = z.mean(dim=1)
        # return F.mse_loss(z, patch_mean)

    def forward(self, z: Tensor, args) -> Tuple[Tensor, dict]:
        batch_sz, n_features = z.shape[0], z.shape[-1]
        # z = F.normalize(z, dim=-1)

        # sim_loss = self.calc_neighbor_similarity(z)
        patch_mean = z.mean(dim=1).unsqueeze(1).repeat(1, z.shape[1], 1)
        sim_loss = F.mse_loss(z, patch_mean)
        
        # calculate the variance losses using the 
        z = torch.flatten(z, start_dim=0, end_dim=1)
        z = z - z.mean(dim=0)
        std_z = torch.sqrt(z.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z))
        

        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
        cov_z = (z.T @ z) / (batch_sz - 1)
        cov_loss = off_diagonal(cov_z).pow_(2).sum().div(n_features)
        loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

        print(f"\t{loss}: {sim_loss} {std_loss} {cov_loss}")

        loss_dict = {
            "loss" : loss.item(),
            "sim_loss" : sim_loss.item(),
            "std_loss" : std_loss.item(),
            "cov_loss" : cov_loss.item()
        }
        return loss, loss_dict
