
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
    cache_tensor = torch.zeros((h * w, n_step), dtype=torch.int32)
    for k in cache:
        tens = torch.tensor(cache[k])
        assert tens.shape[0] == n_step
        cache_tensor[k[0] * w + k[1]] = tens[:, 0] * w + tens[:, 1]

    # M is the same across all channels (and channels are independent of one another)
    return torch.block_diag(*[M for _ in range(img_shape[0])]), n_step, cache_tensor




def generate_aug_probs(img_shape, device):
    with torch.no_grad():    
        zoom_factors = [1.25, 1.75]
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
        
        # calculate augmentation variance
        cache, nstep = prob_map["resize_cache"], prob_map["resize_nstep"]
        
        # for all ((u, v), (u', v')): integrate over all ((x, y), (x', y')) pairs where t((x, y)) == (u, v) and t((x', y')) == (u', v')
        # sum over all elements in the outer product of x[..., cache] with itself along its final dimension
        x = x.flatten(2)
        second_mom = torch.einsum('bchw,bcdk->bchd', x[..., cache], x[..., cache])

        # scale by prob. of each possible augmentation
        second_mom /= nstep

        # reformat so each channel of image only interacts with entries for that channel (block_diag if it supported batching)
        shp = second_mom.shape
        second = torch.zeros((shp[0], shp[1] * shp[2], shp[1] * shp[3]), device=second_mom.device)
        for idx in range(shp[1]):
            second[:, idx * shp[2] : (idx + 1) * shp[2], idx * shp[3] : (idx + 1) * shp[3]] = second_mom[:, idx, ...]
        del second_mom
        EM_outer = (EM.unsqueeze(1) * EM.unsqueeze(0)).permute((2, 0, 1))
        
        var = second - EM_outer
        return EM.T, var

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

    # J = torch.zeros((inp.shape[0], 16, *inp.shape[1:]), device=inp.device, dtype=inp.dtype)
    # for idx in range(inp.shape[0]):
    #     J[idx] = torch.func.jacrev(helper)(inp[idx])

    model.train()
    return J


def calc_tangent_prop_loss(model, inp, var_decomp):
    # Calculate the mean Frobenius norm of the dot products of the scaled eigenvectors of the augmentation variance matrix with the Jacobian of the model at the given input

    def _helper(x):
        return model(x)[1].squeeze()

    def _loss_calc(x, var_decomp):
        J = torch.func.jacrev(_helper)(x).flatten(1, -1)
        assert len(J.shape) == 2
        return torch.linalg.matrix_norm(J @ var_decomp, ord="fro")


    # TODO(as) sketchy... means running stats wont be updated
    model.eval()
    loss = torch.func.vmap(_loss_calc)(inp.unsqueeze(1), var_decomp)

    # J = torch.zeros((inp.shape[0], 16, *inp.shape[1:]), device=inp.device, dtype=inp.dtype)
    # for idx in range(inp.shape[0]):
    #     J[idx] = torch.func.jacrev(helper)(inp[idx])


    model.train()    
    return loss.mean()





def loss_function(img_batch, model, aug_prob_map, intermediate):
    """
    Calculate augmentation closed form TangentProp loss + modified MMCR anti-collapse objective
    """
    assert len(img_batch.shape) == 4


    # batch-level anti-collapse objective (MMCR) --> maximize singular values of normalized mean augmentations
    # out = model(img_batch)[1]
    # global_nuc = torch.linalg.svdvals(F.normalize(out.float(), dim=-1)).sum()     # TODO(as): using this as anti-collapse, do we want to be using L2 vs. L1 here?
    global_nuc = torch.tensor([0])

    tangent_prop = calc_tangent_prop_loss(model, img_batch, intermediate)

    # pdb.set_trace()
    # global_nuc *= 0.1

    loss = tangent_prop #- global_nuc
    print(f"{global_nuc} {tangent_prop} -> {loss}")

    return loss, {"tangent":tangent_prop.item(), "svd":global_nuc.item()}





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
