
import torch
import torchvision
import einops
import time
import math
from tqdm import tqdm

import torchvision.utils as vutils
import torch.nn.functional as F

import sys
import pdb


from  mmcr.cifar_stl.augmentation import RandomCrop, HorizFlip, Grayscale, ColorJitter


def generate_aug_probs(img_shape):
    with torch.no_grad():    
        zoom_factors = [1.25, 2, 3]
        return {
            "rc" : RandomCrop(img_shape, zoom_factors),
            # "horiz" : HorizFlip(0.5),
            # "gray" : Grayscale(0.2),
            # "jitter" : ColorJitter([0.4, 0.4, 0.2, 0.1], 0.8)
        }


def calc_aug_ev_var(x, prob_map):
    """

    TODO: these are the weak default aug, still need Gaussian blur, solarization, etc.

    transforms.RandomResizedCrop(32)        --> product of a bunch of "indep" variables
        - random vertical + horizontal shift
        - random shear (depending on aspect ratio)
        - zoom
    transforms.RandomHorizontalFlip(p=0.5)  --> easy convex comb
        - Bernoulli var. of flip application
    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8       --> easy convex comb
        - Bernoulli var. of jitter application (brightness, saturation, contrast, hue, ...) --> right now applies with prob. 1
    transforms.RandomGrayscale(p=0.2)         --> easy convex comb
        - Bernoulli var. of grayscale application
    """

    # TODO(as) apply horiz flip, grayscale, + jitter bernoulli augmentations

    ev, rrc_var = prob_map["rc"].calc_mean_var(x)
    var = rrc_var
    

    # EV needs to be in a format to go through the network
    ev = einops.rearrange(ev, "B (C H W) -> B C H W", C=x.shape[1], H=x.shape[2])

    # ev = ev[0]
    # x = x[0]
    # foo = (ev - ev.min()) / (ev.max() - ev.min())
    # vutils.save_image(foo, "ev.png")
    # vutils.save_image(x, "raw.png")
    # print("SAVED")
    # sys.exit()

    return ev, var





def calc_tangent_prop_loss(model, inp, var_decomp):
    # Calculate the mean Frobenius norm of the dot products of the scaled eigenvectors of the augmentation variance matrix with the Jacobian of the model at the given input

    # TODO: maybe functional_call helps?
    def _helper(x):
        # return F.normalize(model(x)[1].squeeze(), dim=-1)
        return model(x)[1].squeeze()

    def _loss_calc(x, var_decomp):
        J = torch.func.jacrev(_helper)(x).flatten(1, -1)
        assert len(J.shape) == 2
        norm = torch.linalg.norm(J)

        # Norm of the Jacobian in the direction of the augmentation variance (norm of the projection of the Jacobian onto each scaled aug variance e'vec)
        J_aug_norm = J @ (var_decomp / (torch.linalg.norm(var_decomp, dim=1).unsqueeze(0) + 1e-4))

        # NOTE: removes dependence of this loss on the Jacobian norm (removes the degenerate solution of minimizing the Jacobian norm). This makes the minimization of the anti-collapse loss work better
        J = F.normalize(J, dim=-1)
        
        return torch.linalg.matrix_norm(J @ var_decomp, ord="fro"), norm, J_aug_norm


    # TODO(as) sketchy... means running stats wont be updated
    model.eval()
    loss, jac_norm, jac_aug_norm = torch.func.vmap(_loss_calc)(inp.unsqueeze(1), var_decomp)

    # J = torch.zeros((inp.shape[0], 16, *inp.shape[1:]), device=inp.device, dtype=inp.dtype)
    # for idx in range(inp.shape[0]):
    #     J[idx] = torch.func.jacrev(helper)(inp[idx])

    model.train()    
    return loss.mean(), jac_norm.mean(), jac_aug_norm


def sampling_tangent_prop_loss(model, x, aug, naug=1000):
    # Sampling version of the tangent prop loss to act as a sanity check when debugging possible variance matrix/Jac bugs
    model.eval()

    def helper(x):
        return model(x)[1].squeeze()

    loss = torch.tensor(0., device=x.device, requires_grad=True)
    norm = torch.tensor(0., device=x.device)
    
    for idx in range(x.shape[0]):
        # calculate Jacobian at x[idx]
        inp = x[idx]
        jac = torch.func.jacrev(helper)(inp.unsqueeze(0)).flatten(1, -1)

        with torch.no_grad():
            norm += torch.linalg.norm(jac)
            augs = torch.zeros((naug, x.shape[1] * x.shape[2] * x.shape[3]), dtype=x.dtype, device=x.device)
            for jdx in range(naug):
                augs[jdx] = aug.generate_random_sample(inp).flatten()
        
        embed = F.normalize(jac, dim=-1) @ augs.T
        loss = loss + torch.linalg.norm(embed, dim=0).sum()

    loss = loss / (x.shape[0] * naug)
    norm /= x.shape[0]
    model.train()
    return loss, norm


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(model, batch):
    # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py#L202
    x = model(batch)[1]
    # x = x - x.mean(dim=0)
    batch_sz, num_features = x.shape[0], x.shape[1]
    
    std_x = torch.sqrt(x.var(dim=0) + 1e-8)
    std_loss = torch.mean(F.relu(1 - std_x))

    cov_x = (x.T @ x) / (batch_sz - 1)
    cov_x = off_diagonal(cov_x)
    cov_loss = cov_x.pow_(2).sum().div(num_features)
    
    print(f"\t{std_x.max()} {std_x.min()} ({cov_x.max()} {cov_x.min()}). {x.mean(dim=0).abs().max()}")
    # print(f"\t{x.max(dim=0).values.cpu().detach().numpy()} \n\t{x.min(dim=0).values.cpu().detach().numpy()}")
    
    return std_loss, cov_loss


def loss_function(img_batch, model, intermediate):
    """
    Calculate augmentation closed form TangentProp loss + modified MMCR anti-collapse objective
    """
    assert len(img_batch.shape) == 4


    # batch-level anti-collapse objective (MMCR) --> maximize singular values of normalized mean augmentations
    # out = model(img_batch)[1]
    # global_nuc = torch.linalg.svdvals(F.normalize(out.float(), dim=-1)).sum()     # TODO(as): using this as anti-collapse, do we want to be using L2 vs. L1 here?

    # NOTE: if we allow the update of BatchNorm running counts when calc vicreg_loss, loss diverges for some reason...
    # model.eval()

    std_loss, cov_loss = vicreg_loss(model, img_batch)
    
    # std_loss, cov_loss = torch.tensor(0), torch.tensor(0)
    # tangent_prop, mean_jac_norm = torch.tensor(0), torch.tensor(0)

    tangent_prop, mean_jac_norm, mean_jac_aug_norm = calc_tangent_prop_loss(model, img_batch, intermediate)
    cov_loss *= 0.1

    # pdb.set_trace()

    jac_aug_norm_loss = mean_jac_aug_norm.pow_(2).mean()

    loss = tangent_prop + std_loss + cov_loss

    print(f"{tangent_prop}, {jac_aug_norm_loss} ({mean_jac_aug_norm} {std_loss} {cov_loss}) -> {loss}")

    return loss, {"tangent":tangent_prop.item(), "std_loss":std_loss.item(), "cov_loss":cov_loss.item(), "jac_norm":mean_jac_norm.item(), "jac_aug_norm_loss":jac_aug_norm_loss.item()}


def log_model_jacobian(vis_dict, stats_data, model, device):
    jac_norm_sum = 0
    batch_sz = 16

    def helper(x):
        # F.normalize(model(x)[1].squeeze(), dim=-1)
        return model(x)[1].squeeze()

    for start in range(0, stats_data.shape[0], batch_sz):
        end = min(start + batch_sz, stats_data.shape[0])
        btch = stats_data[start : end].unsqueeze(1).to(device)
        jac = torch.func.vmap(torch.func.jacrev(helper))(btch).flatten(1, -1)
        jac_norm_sum += torch.linalg.norm(jac, dim=1).sum()

    # mean of per-sample Jacobian norms
    vis_dict["mean_jac_norm"] = jac_norm_sum / stats_data.shape[0]
    print(f"TEST AUG JAC NORM: {vis_dict["mean_jac_norm"]}\n")
    return vis_dict





def calc_aug_var_decomp(img_batch, aug_prob_map):
    with torch.no_grad():
        # calculate augmentation expected value and variance
        aug_ev, aug_var = calc_aug_ev_var(img_batch, aug_prob_map)

        S, U = torch.linalg.eigh(aug_var)
        U[S < 0, :] *= -1
        S = S.abs()
        S = torch.sqrt(S)
        
        intermediate = U * S.unsqueeze(-1)
        return intermediate