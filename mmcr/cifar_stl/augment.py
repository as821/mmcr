
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








def loss_function(img_batch, model):
    """
    Calculate augmentation closed form TangentProp loss + modified MMCR anti-collapse objective
    """
    with torch.no_grad():
        # calculate augmentation expected value and variance
        aug_ev, aug_var = calc_aug_ev_var(img_batch)

        # calculate augmentation variance matrix (+ apply SVD) --> TODO(as): paper notation implies U.T == Vh (so eigh could be used). may have variance matrix shape wrong
        eval, evec = torch.linalg.eigh(aug_var)

    # calc. model Jacobian wrt EV aug --> TODO(as): is jacrev differentiable??
    def helper(x):
        feature, out = model(x)
        return out.squeeze()
    
    model.eval()            # TODO(as) sketchy... means running stats wont be updated
    inp = einops.rearrange(aug_ev, "B (C H W) -> B C H W", B=img_batch.shape[0], C=img_batch.shape[1], H=img_batch.shape[2]).unsqueeze(1)
    J_aug_ev = torch.func.vmap(torch.func.jacrev(helper))(inp)
    model.train()
    
    J_aug_ev = J_aug_ev.flatten(2, -1)
    assert len(J_aug_ev.shape) == 3
    
    # calc TangentProp loss
    # intermediate = J_aug_ev @ evec @ torch.sqrt(eval)
    intermediate = torch.bmm(torch.bmm(J_aug_ev, evec), torch.sqrt(eval).unsqueeze(-1))[..., 0]
    loss = torch.linalg.matrix_norm(intermediate, ord="fro") ** 2

    return loss




