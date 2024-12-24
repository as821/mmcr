
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




def bernoulli_aug(aug, orig, prob):
    ev = prob * aug + (1 - prob) * orig

    # covariance matrix is N x N (where N is the size of the flattened image)
    second_mom_diag = prob * (aug**2) + (1 - prob) * (orig**2)      # E[X^2]
    var_diag = second_mom_diag - (ev ** 2)                          # E[X^2] - E[X]^2

    # in the bernoulli case all pixels are independent of one another so covariance matrix is diagonal
    var = torch.diag(var_diag.flatten())
    return ev, var





def resize_crop_operator(img_shape, zoom_factors):
    def _per_crop_prob_calc(M, zf, horiz_step, vert_step, h, w):

        # for u in range(h):
        #     for v in range(w):
        #         # (u, v) is (row, col) destination crop indexing. (x, y) is (row, col) indexing of the source pixel in the original image
        #         x = int((u + vert_step) / zf)
        #         y = int((v + horiz_step) / zf)
        #         m_source_idx = x * w + y

        #         m_idx = u * w + v
        #         local_M[m_idx, m_source_idx] += 1

        #         for u_prime in range(h):
        #             for v_prime in range(w):
        #                 x_prime = int((u_prime + vert_step) / zf)
        #                 y_prime = int((v_prime + horiz_step) / zf)
        #                 prime_idx = x_prime * w + y_prime
        #                 local_var[m_source_idx, prime_idx] += 1

        # return local_M, local_var

        # calculate the source pixel coordinates from: destination pixel coordinates, zoom amount, + horiz/vert translation
        local_M = torch.zeros_like(M)
        u_grid, v_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        x = ((u_grid + vert_step) / zf).int()
        y = ((v_grid + horiz_step) / zf).int()
        m_source_idx = (x * w + y).reshape(-1)
        m_dest_idx = (u_grid * w + v_grid).reshape(-1)
        
        # NOTE: can do assignment without accumulation since each m_dest_idx is unique. this implies that each (dest, source) pair is also unique
        local_M[m_dest_idx, m_source_idx] = 1

        # Outer product of counts with itself gives us the number of times each (m_source_idx, m_source_idx) pair appears
        # counts = torch.bincount(m_source_idx, minlength=var.shape[0])
        # local_var = torch.outer(counts, counts)

        return local_M

    
    # Discretized zoom (discretized crop sizes) allow the definition of a discrete uniform probability distribution over all crop size and horiz/vert translation pairs
    h, w = img_shape[-2], img_shape[-1]
    sz = h * w
    M = torch.zeros((sz, sz)) 

    cache = {}
    for u in range(h):
        for v in range(w):
            cache[(u, v)] = []

    # TODO(as) this is stupid. discretize the zoom. Then just define a uniform prob. dist over all possible crops (Cart. prod. of all sizes with all valid horiz/vert translations)
    # Then, iterate through all these crops and determine which source pixel each final pixel comes from (+ associate appropriate prob. mass to it)
    n_step = 0
    for zf in zoom_factors:
        # determine zoom size and then find all windows of original image size that will work as a crop
        zoom_h, zoom_w = int(zf * h), int(zf * w)
        n_horiz_step = zoom_w - w + 1
        n_vert_step = zoom_h - h + 1

        # iterate through all possible translations of this size of crop (assigning each uniform probability)
        for horiz_step in tqdm(range(n_horiz_step)):
            for vert_step in range(n_vert_step):
                n_step += 1
                # determine the source image pixel that corresponds to each output crop pixel and update M
                local_M = _per_crop_prob_calc(M, zf, horiz_step, vert_step, h, w)
                M += local_M

                # prep (u, v) -> (x, y) cache for variance calculation
                for u in range(w):
                    for v in range(h):
                        x = int((u + vert_step) / zf)
                        y = int((v + horiz_step) / zf)
                        cache[(u, v)].append((x, y))



    # normalize M values by the total number of possible (crop size, horiz/vert translation) combinations
    # NOTE: this weights all crop sizes equally so smaller crops will be more frequent than large ones (since theres more possible translations with larger zooms)
    M /= n_step

    # convert each cache entry into a tensor of flattened indices
    for k in cache:
        tens = torch.tensor(cache[k])
        out = tens[:, 0] * w + tens[:, 1]
        cache[k] = out

    # M is the same across all channels (and channels are independent of one another)
    return torch.block_diag(*[M for _ in range(img_shape[0])]), n_step, cache




def generate_aug_probs(img_shape, device):
    with torch.no_grad():    
        zoom_factors = [1.5, 2, 2.5, 3]
        rc_op, rc_nstep, rc_cache = resize_crop_operator(img_shape, zoom_factors)
        return {"resize_crop" : rc_op, "resize_nstep" : rc_nstep, "resize_cache" : rc_cache}



def simple_var(u, v, u_prime, v_prime):
    
    # TODO:
    #   - could pre-calculate which (x, y) map to (u, v) (for all thetas)
    
    v = 0
    # for x in range(h):
    #     for y in range(w):
    #         for x_prime in range(h):
    #             for y_prime in range(w):
                    
    #                 for zf in prob_map["zoom_factors"]:
    #                     for horiz_step in range():
    #                         for vert_step in range():
                                
    #                             # calculate transformed coordinates (u, v) that (x, y) in original image get mapped to under given theta
    #                             u_x_calc = None
    #                             v_y_calc = None
    #                             u_x_prime_calc = None
    #                             v_y_prime_calc = None

    #                             if u_x_calc == u and v_y_calc == v and u_x_prime_calc == u_prime and v_y_prime_calc == v_prime:
    #                                 v += img[x, y] * img[x_prime, y_prime] * (1 / num_steps)

    cache = {}

    for (x, y) in cache[(u, v)]:
        for (x_prime, y_prime) in cache[(u_prime, v_prime)]:
            v += img[x, y] * img[x_prime, y_prime] * (1 / num_steps)

    return v






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

    def horiz_vert_trans(batch):
        M = prob_map["horiz_vert_trans"]
        M = M.to(batch.device).to(batch.dtype)

        # Apply M to given image to calculate the augmentation mean and variance
        flat = batch.flatten(1).T
        EM = M @ flat
        
        # second moment is the convolution of the windows of possible translations for a pair of pixels, weighted by the probability of those translations
        with torch.no_grad():
            unif_range = 7
            patch_sz = unif_range * 2 + 1            
            inp_chan = 3
            conv = prob_map["horiz_vert_conv"]
            
            # for each pixel location in the image, we now have the set of all possible values it could take on for each possible horiz/vert translation
            patches = conv(batch).flatten(2)
            
            # take (probability-weighted) convolution of all pairs of pixels to get the second moment
            prob = torch.full((inp_chan, patch_sz * patch_sz), 1 / (patch_sz * patch_sz), device=patches.device)
            weighted_outer = torch.zeros((patches.shape[0], patches.shape[2], patches.shape[2], inp_chan), device=patches.device)
            for idx in range(patches.shape[0]):     # full outer product too memory intensive with larger batch sizes
                b = patches[idx].T
                b = einops.rearrange(b, "A (I C) -> A C I", C=inp_chan)

                # o = b.unsqueeze(0) * b.unsqueeze(1) 
                # o = o * prob
                # o = o.sum(dim=-1)
                weighted_outer[idx] = torch.einsum('chw,bhw,hw->cbh', b, b, prob)

            # reformat so each channel of image only interacts with entries for that channel (block_diag if it supported batching)
            shp = weighted_outer.shape
            second = torch.zeros((shp[0], shp[1] * shp[3], shp[2] * shp[3]), device=weighted_outer.device)
            for idx in range(shp[3]):
                second[:, idx * shp[1] : (idx + 1) * shp[1], idx * shp[2] : (idx + 1) * shp[2]] = weighted_outer[..., idx]

        EM_outer = (EM.unsqueeze(1) * EM.unsqueeze(0)).permute((2, 0, 1))
        var = second - EM_outer
        return EM.T, var


    def random_resized_crop(x):
        """
        Scale: [lower, upper] bound on the ratio of the original height/width of the crop prior to resizing (NOTE: RandomResizedCrop uses area + non-square crops)
        
        2) Select + perform horizontal/vertical translations 
        3) Perform zoom to achieve the selected crop size
        """
        


        # TODO(as) this is stupid. discretize the zoom. Then just define a uniform prob. dist over all possible crops (Cart. prod. of all sizes with all valid horiz/vert translations)
        # Then, iterate through all these crops and determine which source pixel each final pixel comes from (+ associate appropriate prob. mass to it)

        M = prob_map["resize_crop"].to(x.device).to(x.dtype)

        # mean image
        flat = x.flatten(1).T
        EM = M @ flat

        print("Entering cov calc...")

        # calculate augmentation variance
        cache, nstep = prob_map["resize_cache"], prob_map["resize_nstep"]
        second_mom = torch.zeros((x.shape[0], x.shape[1], M.shape[0], M.shape[1]), device=x.device)
        h, w = x.shape[-2], x.shape[-1]
        for idx in range(x.shape[0]):
            img = x[idx].flatten(1)

            for u in range(h):
                for v in range(w):
                    var_idx = u * w + v
                    uv_img = img[:, cache[(u, v)]]
                    for u_prime in range(h):
                        for v_prime in range(w):
                            var_prime_index = u_prime * w + v_prime
                            
                            # integrate over all ((x, y), (x', y')) pairs where t((x, y)) == (u, v) and t((x', y')) == (u', v')
                            prime_img = img[:, cache[(u_prime, v_prime)]]
                            second_mom[idx, :, var_idx, var_prime_index] = (prime_img.unsqueeze(2) * uv_img.unsqueeze(1)).sum(dim=(1, 2))

        second_mom /= nstep

        pdb.set_trace()

        return ev, var

    ev, rrc_var = random_resized_crop(x)
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





def calc_model_jac(model, inp):
    def helper(x):
        _, out = model(x)
        return out.squeeze()

    # TODO(as) sketchy... means running stats wont be updated
    model.eval()
    J = torch.func.vmap(torch.func.jacrev(helper))(inp)
    model.train()

    # J_aug_ev = torch.zeros((aug_ev.shape[0], 128, *aug_ev.shape[1:]), device=aug_ev.device, dtype=aug_ev.dtype)
    # for idx in range(aug_ev.shape[0]):
    #     J_aug_ev[idx] = torch.autograd.functional.jacobian(helper, aug_ev[idx], create_graph=True)
    return J




def loss_function(img_batch, model, aug_prob_map):
    """
    Calculate augmentation closed form TangentProp loss + modified MMCR anti-collapse objective
    """
    assert len(img_batch.shape) == 4
    with torch.no_grad():
        # calculate augmentation expected value and variance
        aug_ev, aug_var = calc_aug_ev_var(img_batch, aug_prob_map)

        # U, S, Vh = torch.linalg.svd(aug_var)
        # del Vh

        S, U = torch.linalg.eigh(aug_var)
        
        # diff = (aug_var[0] - aug_var[0].T).abs().max()
        # print(f"{S.min()} {S.max()} {diff}")
        # TODO(as) odd this is needed, maybe ill-conditioned?? float64 makes no difference
        
        U[S < 0, :] *= -1
        S = S.abs()

        S = torch.sqrt(S)
        intermediate = U * S.unsqueeze(-1)
        del U, S, aug_var

    # batch-level anti-collapse objective (MMCR) --> maximize singular values of normalized mean augmentations
    out = model(aug_ev)[1]
    out = F.normalize(out, dim=-1)
    global_sing_vals = torch.linalg.svdvals(out)
    global_nuc = global_sing_vals.sum()     # TODO(as): using this as anti-collapse, do we want to be using L2 vs. L1 here?

    # calc. model Jacobian wrt EV aug
    aug_ev = aug_ev.unsqueeze(1)
    J_aug_ev = calc_model_jac(model, aug_ev)
    J_aug_ev = J_aug_ev.flatten(2, -1)
    assert len(J_aug_ev.shape) == 3
    
    # scaled e'vec are the columns of the matrix
    intermediate = intermediate.permute(0, 2, 1)
    
    # minimize the cosine similarity (not the dot product) of the Jacobian and the covarinace e'vec
    # J_aug_ev = F.normalize(J_aug_ev, dim=-1)
    # intermediate = F.normalize(intermediate, dim=1)        # e'vec are the rows now, normalize them

    # calc TangentProp loss
    res = torch.bmm(J_aug_ev, intermediate)
    
    # TODO(as): unclear if mean of norm of cosine similarity is the best loss
    tangent_prop = torch.linalg.matrix_norm(res, ord="fro")
    tangent_prop = tangent_prop.mean()

    loss = tangent_prop - global_nuc
    print(f"{global_nuc} {tangent_prop} -> {loss}")

    return loss, {"tangent":tangent_prop.item(), "svd":global_nuc.item(), "aug_ev":aug_ev.detach().to("cpu", non_blocking=True)}





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
