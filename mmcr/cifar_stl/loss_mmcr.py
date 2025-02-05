import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import random
from typing import Tuple

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
    def __init__(self, lmbda: float, n_aug: int, distributed: bool = False, memory_bank=None, l2_spectral_norm=False, spectral_target=False, spectral_topk=False):
        super(MMCR_Loss, self).__init__()
        self.lmbda = lmbda
        self.n_aug = n_aug
        self.distributed = distributed
        self.first_time = True
        self.l2_spectral_norm = l2_spectral_norm
        self.spectral_target = spectral_target
        self.spectral_topk = spectral_topk

        self.memory_bank = memory_bank

    def forward(self, z: Tensor, args) -> Tuple[Tensor, dict]:
        # print(f"{z.max()} {z.min()}")
        
        z = F.normalize(z, dim=-1)
        z_local_ = einops.rearrange(z, "(B N) C -> B C N", N=self.n_aug)

        # gather across devices into list
        if self.distributed:
            z_list = [
                torch.zeros_like(z_local_)
                for i in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(z_list, z_local_, async_op=False)
            z_list[torch.distributed.get_rank()] = z_local_

            # append all
            z_local = torch.cat(z_list)

        else:
            z_local = z_local_

        centroids = torch.mean(z_local, dim=-1)
        
        if args.precond_alpha > 0:
            centroids = GradientPreconditioning.apply(centroids, args.precond_alpha, args.precond_thresh, args.precond_pow, args.precond_center)

        # print(f"{z.max()} {z.min()} {centroids.max()} {centroids.min()}")
        # print(f"{torch.linalg.cond(centroids)}")

        if self.memory_bank is not None:
            curr_centroids = centroids.detach()
            if self.memory_bank.is_warm():
                centroids = torch.cat([self.memory_bank.buf.detach(), centroids])
            self.memory_bank.enqueue(curr_centroids)

        if self.lmbda != 0.0:
            local_nuc = torch.linalg.svdvals(z_local).sum()
        else:
            local_nuc = torch.tensor(0.0)
        global_sing_vals = torch.linalg.svdvals(centroids)
        
        if self.l2_spectral_norm:
            global_nuc = torch.linalg.vector_norm(global_sing_vals)
        elif self.spectral_target:
            # we want to minimize distance between singular values and 1, but further down this term is mult by -1. Counteract that here
            global_nuc = (global_sing_vals - 1).sum()
        elif self.spectral_topk:
            # maximize the value of the num. class largest values, minimize the rest
            sorted_values, _ = torch.sort(global_sing_vals, descending=True)
            global_nuc = sorted_values[:10].sum() - sorted_values[10:].sum()
        else:
            global_nuc = global_sing_vals.sum()

        batch_size = z_local.shape[0]
        loss = self.lmbda * local_nuc / batch_size - global_nuc

        loss_dict = {
            "loss": loss.item(),
            "local_nuc": local_nuc.item(),
            "global_nuc": global_nuc.item(),
            "global_sing_vals" : global_sing_vals.detach().cpu().numpy(), 
        }

        self.first_time = False

        return loss, loss_dict
