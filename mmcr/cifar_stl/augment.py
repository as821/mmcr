
import torch
import torchvision
import einops


import pdb




def bernoulli_aug(aug, orig, prob):
    ev = prob * aug + (1 - prob) * orig

    # covariance matrix is N x N (where N is the size of the flattened image)
    second_mom_diag = prob * (aug**2) + (1 - prob) * (orig**2)      # E[X^2]
    var_diag = second_mom_diag - (ev ** 2)                          # E[X^2] - E[X]^2
    
    # in the bernoulli case all pixels are independent of one another so covariance matrix is diagonal
    var = torch.diag(var_diag)
    return ev, var




def calc_aug_ev_var(x):
    """

    TODO: these are the weak default aug, still need Gaussian blur, solarization, etc.

    transforms.RandomResizedCrop(32)        --> product of a bunch of "indep" variables (not really, but keep naive assumption)
        - random vertical + horizontal shift
        - random shear (depending on aspect ratio)
        - zoom
    transforms.RandomHorizontalFlip(p=0.5)  --> easy convex comb
        - Bernoulli var. of flip application
    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8       --> easy convex comb
        - Bernoulli var. of jitter application (brightness, saturation, contrast, hue, ...)
    transforms.RandomGrayscale(p=0.2)         --> easy convex comb
        - Bernoulli var. of grayscale application
    """


    def random_resized_crop(x):
        pass

    def jitter(x, p, param=[0.4, 0.4, 0.2, 0.1]):
        pass

    def horiz_flip(x, p):
        flip = torchvision.transforms.functional.hflip(x)
        return bernoulli_aug(flip, x, p)

    def grayscale(x, p):
        gs = torchvision.transforms.Grayscale(3)(x)
        return bernoulli_aug(gs, x, p)


    # TODO(as): augmentations are jointly-distributed variables
    #   - EV: joint probability table 
    #   - Variance: E[X^2] - E[X]^2


    flip = torchvision.transforms.functional.hflip(x).flatten(1)
    gs = torchvision.transforms.Grayscale(3)(x).flatten(1)
    both = torchvision.transforms.functional.hflip(torchvision.transforms.Grayscale(3)(x)).flatten(1)
    none = x.flatten(1)
    flip_prob = 0.5
    gs_prob = 0.2

    # EV: E[X]
    ev = flip_prob * (1-gs_prob) * flip             # flip, no GS
    ev += (1 - flip_prob) * gs_prob * gs            # no flip, GS
    ev += (1 - flip_prob) * (1 - gs_prob) * none    # no flip, no GS
    ev += flip_prob * gs_prob * both                # flip, GS

    # 2nd moment: E[X^2]
    sec_mom = flip_prob * (1-gs_prob) * flip**2             # flip, no GS
    sec_mom += (1 - flip_prob) * gs_prob * gs**2            # no flip, GS
    sec_mom += (1 - flip_prob) * (1 - gs_prob) * none**2    # no flip, no GS
    sec_mom += flip_prob * gs_prob * both**2                # flip, GS

    # Variance: E[X^2] - E[X]^2
    variance = sec_mom - ev ** 2
    variance = torch.diag_embed(variance)
    return ev, variance



def calc_model_jac(model, inp):
    def helper(x):
        feature, out = model(x)
        return out.squeeze()

    # TODO(as) sketchy... means running stats wont be updated
    model.eval()            
    J = torch.func.vmap(torch.func.jacrev(helper))(inp)
    
    # J_aug_ev = torch.zeros((aug_ev.shape[0], 128, *aug_ev.shape[1:]), device=aug_ev.device, dtype=aug_ev.dtype)
    # for idx in range(aug_ev.shape[0]):
    #     J_aug_ev[idx] = torch.autograd.functional.jacobian(helper, aug_ev[idx], create_graph=True)
    model.train()
    return J




def loss_function(img_batch, model):
    """
    Calculate augmentation closed form TangentProp loss + modified MMCR anti-collapse objective
    """
    with torch.no_grad():
        # calculate augmentation expected value and variance
        aug_ev, aug_var = calc_aug_ev_var(img_batch)

        # calculate augmentation variance matrix (+ apply SVD) --> TODO(as): paper notation implies U.T == Vh (so eigh could be used). may have variance matrix shape wrong
        U, S, Vh = torch.linalg.svd(aug_var)
        # print(f"{torch.linalg.cond(aug_var.to(torch.float64)).item()}, {S.min()} {S.max()}")
        intermediate = torch.bmm(U, torch.sqrt(S).unsqueeze(-1))
        del U, S, Vh
    
        aug_ev = einops.rearrange(aug_ev, "B (C H W) -> B C H W", B=img_batch.shape[0], C=img_batch.shape[1], H=img_batch.shape[2]).unsqueeze(1)

    # calc. model Jacobian wrt EV aug
    J_aug_ev = calc_model_jac(model, aug_ev)
    J_aug_ev = J_aug_ev.flatten(2, -1)
    assert len(J_aug_ev.shape) == 3
    
    # calc TangentProp loss
    # intermediate = J_aug_ev @ evec @ torch.sqrt(eval)
    intermediate = torch.bmm(J_aug_ev, intermediate)[..., 0]
    
    # TODO(as): norm over batched matrices vs. mean per-example norm
    loss = torch.linalg.matrix_norm(intermediate, ord="fro") ** 2

    return loss





def log_model_jacobian(vis_dict, stats_data, model, device):
    jac_norm_sum = 0
    batch_sz = 16

    for start in range(0, stats_data.shape[0], batch_sz):
        end = min(start + batch_sz, stats_data.shape[0])
        btch = stats_data[start : end].unsqueeze(1).to(device)
        jac = calc_model_jac(model, btch)
        jac = jac.flatten(1, -1)

        jac_norm_sum += torch.linalg.norm(jac, dim=1).sum()

    # mean of per-sample Jacobian norms
    vis_dict["mean_jac_norm"] = jac_norm_sum / stats_data.shape[0]
    print(f"TEST AUG JAC NORM: {vis_dict["mean_jac_norm"]}\n")
    return vis_dict
