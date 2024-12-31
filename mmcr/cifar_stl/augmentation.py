import torch
import torchvision
from tqdm import tqdm
import einops

import random

import pdb


class RandomCrop():
    def __init__(self, img_shape, zoom_factors):
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
        self.op = torch.block_diag(*[M for _ in range(img_shape[0])])
        self.nstep = n_step
        self.cache = cache_tensor
        
        self.zf = zoom_factors
        self.img_shape = img_shape
        self.resize = torchvision.transforms.Resize((self.img_shape[-2], self.img_shape[-1]), interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT, max_size=None, antialias=False)


    def calc_mean_var(self, x):
        """
        Scale: [lower, upper] bound on the ratio of the original height/width of the crop prior to resizing (NOTE: RandomResizedCrop uses area + non-square crops)
        
        2) Select + perform horizontal/vertical translations 
        3) Perform zoom to achieve the selected crop size
        """

        # TODO(as) this is stupid. discretize the zoom. Then just define a uniform prob. dist over all possible crops (Cart. prod. of all sizes with all valid horiz/vert translations)
        # Then, iterate through all these crops and determine which source pixel each final pixel comes from (+ associate appropriate prob. mass to it)

        M = self.op.to(x.device).to(x.dtype)

        # mean image
        flat = x.flatten(1).T
        EM = (M @ flat).T
        
        # calculate augmentation variance
        
        # for all ((u, v), (u', v')): integrate over all ((x, y), (x', y')) pairs where t((x, y)) == (u, v) and t((x', y')) == (u', v')
        # sum over all elements in the outer product of x[..., cache] with itself along its final dimension
        x = x.flatten(2)
        slc = x[..., self.cache]
        second_mom = torch.einsum('bchw,bcdk->bchd', slc, slc)

        # scale by prob. of each possible augmentation
        second_mom /= self.nstep

        # var = second moment - EM^2
        # assumes all channels are independent
        EM_tmp = einops.rearrange(EM, "a (b c) -> a b c", b=second_mom.shape[1])
        second_mom -= (EM_tmp.unsqueeze(-2) * EM_tmp.unsqueeze(-1))

        return EM, second_mom


    def generate_random_sample(self, x):
        assert len(x.shape) == 3

        # Randomly sample a zoom factor, then randomly sample a vertical/horizontal translation. Resize and return.
        rand_zf = random.choice(self.zf)
        h, w = self.img_shape[-2], self.img_shape[-1]
        zoom_h, zoom_w = int(rand_zf * h), int(rand_zf * w)
        n_horiz_step = zoom_w - w + 1
        n_vert_step = zoom_h - h + 1
        rand_horiz = random.randrange(n_horiz_step)
        rand_vert = random.randrange(n_vert_step)

        # get crop
        vert = int(rand_vert / rand_zf)
        vert_end = int((rand_vert + h) / rand_zf)
        horiz = int(rand_horiz / rand_zf)
        horiz_end = int((rand_horiz + w) / rand_zf)
        crop = x[:, vert : vert_end, horiz : horiz_end]
        out = self.resize(crop)

        assert list(out.shape) == self.img_shape
        return out



class BernoulliAug():
    def __init__(self, prob):
        self.prob = prob

    def _augment(self, x):
        return None

    def calc_mean_var(self, x, prob):
        aug = self._augment(x)
        ev = prob * aug + (1 - prob) * x

        # covariance matrix is N x N (where N is the size of the flattened image)
        second_mom_diag = prob * (aug ** 2) + (1 - prob) * (x ** 2)         # E[X^2]
        var_diag = second_mom_diag - (ev ** 2)                              # E[X^2] - E[X]^2

        # in the bernoulli case all pixels are independent of one another so covariance matrix is diagonal
        var = torch.diag(var_diag.flatten())
        return ev, var
    
    def generate_random_sample(self, x):
        pass
            
class HorizFlip(BernoulliAug):
    def __init__(self, prob):
        super().__init__(prob)

    def _augment(self, x):
        return torchvision.transforms.functional.hflip(x)

class Grayscale(BernoulliAug):
    def __init__(self, prob):
        super().__init__(prob)
        self.gs = torchvision.transforms.Grayscale(3)

    def _augment(self, x):
        return self.gs(x)

class ColorJitter():
    def __init__(self, param, prob):
        # ColorJitter samples uniformly at random from range for each. Specify size 0 range to get deterministic behavior
        mx = [(j, j) for j in [i + 1 for i in param[:-1]]] + [(param[-1], param[-1])]
        mn = [(j, j) for j in [max(0, 1 - i) for i in param[:-1]]] + [(-1 * param[-1], -1 * param[-1])]
        
        self.mx_jitter = torchvision.transforms.ColorJitter(*mx)
        self.mn_jitter = torchvision.transforms.ColorJitter(*mn)
        
        self.param = param
        self.prob = prob

    def calc_mean_var(self, x):
        
        # TODO(as) this is wrong:
        #   - need Bernoulli over entire transform
        #   - need EV to be mean jitter value (0.5 * (mx - mn) + mn)
        #   - variance calc. is wrong. this is a joint over uniform distributions, not one big uniform dist
        
        # variance of the uniform distribution ((b - a)^2) / 12 where b and a are the limits (per-pixel in our case)
        x_mx_jitter = self.mx_jitter(x)
        x_mn_jitter = self.mn_jitter(x)
        var_diag = ((x_mx_jitter - x_mn_jitter) ** 2) / 12

        # NOTE: "mean" image is the average over all the jittered images... not necessarily an un-perturbed image
        
        return x, torch.diag(var_diag.flatten())

    def generate_random_sample(self, x):
        pass

