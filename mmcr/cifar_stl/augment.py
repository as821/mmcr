
import torch
import torchvision
import einops
import time
import math

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




def generate_aug_probs(img_shape, device):
    def horiz_vert_trans_operator(img_shape, device):
        unif_range = 7
        
        sz = img_shape[-2] * img_shape[-1]
        M = torch.zeros((sz, sz))        
        for x in range(img_shape[-2]):
            for y in range(img_shape[-1]):
                m_idx = x * img_shape[-2] + y
                
                # NOTE: assume uniform dist over some max translation for both horizontal and vertical for initial simplicity
                for x_off in range(-1 * unif_range, unif_range):
                    if x_off + x < 0 or x_off + x >= img_shape[-1]:
                        continue
                    for y_off in range(-1 * unif_range, unif_range):
                        if y_off + y < 0 or y_off + y >= img_shape[-2]:
                            continue
                        m_off_idx = (x + x_off) * img_shape[-2] + (y + y_off)
                        
                        # horiz and vert translation each happen with prob 1 / (2 * range + 1). Prob of both happening is that squared
                        M[m_idx, m_off_idx] = (1 / (unif_range * 2 + 1)) ** 2
        
        # a "1" for each location in the image. this filter extracts the (patch_sz, patch_sz) patch centered at each pixel in the image (with zero padding)
        patch_sz = unif_range * 2 + 1
        inp_chan = 3
        conv = torch.nn.Conv2d(inp_chan, patch_sz * patch_sz * inp_chan, patch_sz, stride=1, padding=unif_range, bias=False).to(device)
        conv.weight.data.fill_(0)
        for idx in range(patch_sz):
            for jdx in range(patch_sz):
                for cdx in range(inp_chan):
                    conv.weight.data[idx * patch_sz * inp_chan + jdx * inp_chan + cdx, cdx, idx, jdx] = 1

        # M is the same across all channels (and channels are independent of one another)
        return torch.block_diag(*[M for _ in range(img_shape[0])]), conv

    def zoom_operator(img_shape, unif_range=[3, 3]):
        # NOTE: only support uniform scaling of both axes for now
        # for each other pixel in the image, what is the probability that its value impacts the value of the current pixel (under the zoom distribution)
        sz = img_shape[-2] * img_shape[-1]
        scale_sz = math.ceil(sz * unif_range[1])
        M = torch.zeros((scale_sz, sz))        
        # for u in range(img_shape[-2]):
        #     for v in range(img_shape[-1]):
        #         m_idx = u * img_shape[-2] + v
        #         for x in range(u, img_shape[-2]):
        #             for y in range(v, img_shape[-1]):
        #                 # TODO: div by zero
        #                 x_scale = u / x
        #                 y_scale = v / y


        #                 # TODO: how much prob mass??


        # TODO(as) what about the bounds checks??? all these scaled indices assume the image size changes when in reality it will be cropped down...

        # TODO(as) --> only keep those destination indices that will be in the center of the image 
        #       M will stay the current size, but we need to adjust indexing appropriately
        # VISUALIZE current, then likely need to center indexing on middle of image rather than top left


        # NOTE: to handle the issues detailed above, will start with source rather than destination pixel
        range_sz = unif_range[1] - unif_range[0]
        for u in range(img_shape[-2]):
            for v in range(img_shape[-1]):
                m_idx = u * img_shape[-2] + v

                # for a given source (x, y) location determine the bounds of the places it could end up given the current scaling parameters
                x_mn_scale, x_mx_scale = int(u * unif_range[0]), int(u * unif_range[1])
                y_mn_scale, y_mx_scale = int(v * unif_range[0]), int(v * unif_range[1])
                
                print(f"\n({u}, {v}): ({x_mn_scale}, {x_mx_scale}) ({y_mn_scale} {y_mx_scale}) ... ({unif_range[0]}, {unif_range[1]})")

                if x_mn_scale == x_mx_scale and y_mn_scale == y_mx_scale:
                    # all probability mass for this source pixel goes to a single destination
                    m_dest_idx = x_mn_scale * img_shape[-2] + y_mn_scale
                    M[m_dest_idx, m_idx] = 1
                    continue

                if x_mn_scale == x_mx_scale:
                    # prob mass + dest determined entirely by y
                    prev = unif_range[0]
                    for y_idx in range(y_mn_scale, y_mx_scale+1):
                        # find point at which Y switches to the next value, all probability mass from last switch to the next one goes to y_idx
                        if y_idx < y_mx_scale:
                            scale = (y_idx + 1) / v
                        else:
                            scale = unif_range[1]       # final index, all the rest of the prob. mass must go here
                        
                        print(f"\t(y-only, {y_idx}): {scale} {(scale - prev) / range_sz}")
                        assert scale >= prev
                        assert scale <= unif_range[1]
                        m_dest_idx = x_mn_scale * img_shape[-2] + y_idx
                        M[m_dest_idx, m_idx] = (scale - prev) / range_sz        # prob from uniform scaling dist that (u, v) ends up at this pixel value
                        prev = scale
                    continue

                if y_mn_scale == y_mx_scale:
                    # prob mass + dest determined entirely by x
                    prev = unif_range[0]
                    for x_idx in range(x_mn_scale, x_mx_scale + 1):
                        if x_idx < y_mx_scale:
                            scale = (x_idx + 1) / u
                        else:
                            scale = unif_range[1]

                        print(f"\t(x-only, {x_idx}): {scale} {(scale - prev) / range_sz}")

                        assert scale >= prev
                        assert scale <= unif_range[1]
                        m_dest_idx = x_idx * img_shape[-2] + y_mn_scale
                        M[m_dest_idx, m_idx] = (scale - prev) / range_sz
                        prev = scale
                    continue



                # since the distribution over scaling factors is uniform and we now have the bounds for the max/min scaling that can be applied
                # we can determine the probability that this pixel will contribute to each of the destination
                prev = unif_range[0]
                for x_val in range(x_mn_scale, x_mx_scale + 1):
                    # determine scale value (prob mass)
                    if x_val < x_mx_scale:
                        scale = (x_val + 1) / u
                    else:
                        scale = unif_range[1]
                
                    print(f"\t{x_val}: {scale} {(scale - prev) / range_sz}")
            
                    assert scale >= prev
                    assert scale <= unif_range[1]

                    # both axes scaled jointly. determine the y values that are traversed during this x scaling
                    y_base_idx = int(v * prev)
                    y_top_idx = int(v * scale)
                    y_prev = prev
                    for y_val in range(y_base_idx, y_top_idx + 1):
                        if y_val < y_top_idx:
                            y_scale = (y_val + 1) / v
                        else:
                            y_scale = scale
                        
                        print(f"\t\t({x_val}, {y_val}): {y_scale} {(y_scale - y_prev) / range_sz}")
                        
                        assert y_scale >= y_prev
                        assert y_scale <= scale
                        m_dest_idx = x_val * img_shape[-2] + y_val
                        M[m_dest_idx, m_idx] = (y_scale - y_prev) / range_sz
                        y_prev = y_scale
                    prev = scale

        # TODO(as): still need to consider scaling of rows... these are given as source pixel probabilities but the destination pixel rows need to sum to 1 (do they really?)


        # M is the same across all channels (and channels are independent of one another)
        return torch.block_diag(*[M for _ in range(img_shape[0])])

    zoom_factor = 1
    img_shape = (img_shape[0], int(img_shape[1] * zoom_factor), int(img_shape[2] * zoom_factor))

    trans_op, conv = horiz_vert_trans_operator(img_shape, device)
    return {"horiz_vert_trans" : trans_op, "horiz_vert_conv":conv, "zoom":zoom_operator(img_shape)}



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


    def zoom(batch):
        # TODO(as) generate M(theta), then apply. zoom is just scaling: y' = \alpha * y
        # Only consider positive scaling (cropping). May have to do this in conjunction with horiz/vert translation for this to make sense
    
        M = prob_map["horiz_vert_trans"]
        M = M.to(batch.device).to(batch.dtype)
        
        flat = batch.flatten(1).T
        EM = M @ flat

        return EM.T, None



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
        
        # TODO(as) get expected augmentation + variance for zoom 
        # ev, zoom_var = zoom(x)


        # NOTE: use deterministic zoom for now (effectively a fixed-size crop rather than a random sized one)
        zoom_factor = 1
        b, c, h, w = x.shape
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)
        x = torchvision.transforms.functional.resize(x, (new_h, new_w), antialias=True)

        # get expected augmentation + variance for horizontal + vertical translations
        ev, trans_var = horiz_vert_trans(x)
        var = trans_var

        # center crop back to original size
        crop_h = int((new_h - h) / 2)
        crop_w = int((new_w - w) / 2)        
        ev = einops.rearrange(ev, "B (C H W) -> B C H W", H=new_h, W=new_w)
        ev = ev[..., crop_h : crop_h + h, crop_w : crop_w + w]
        ev = einops.rearrange(ev, "B C H W -> B (C H W)")

        var = einops.rearrange(var, "B (C H W) (X Y Z) -> B C H W X Y Z", H=new_h, W=new_w, Y=new_h, Z=new_w)
        var = var[..., crop_h : crop_h + h, crop_w : crop_w + w, :, crop_h : crop_h + h, crop_w : crop_w + w]
        var = einops.rearrange(var, "B C H W X Y Z -> B (C H W) (X Y Z)")

        return ev, var

    ev, rrc_var = random_resized_crop(x)
    var = rrc_var
    
    # EV needs to be in a format to go through the network
    ev = einops.rearrange(ev, "B (C H W) -> B C H W", C=x.shape[1], H=x.shape[2])

    ev = ev[0]
    x = x[0]


    foo = (ev - ev.min()) / (ev.max() - ev.min())
    vutils.save_image(foo, "ev.png")
    vutils.save_image(x, "raw.png")
    print("SAVED")
    sys.exit()

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
