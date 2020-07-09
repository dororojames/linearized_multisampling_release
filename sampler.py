import torch
import torch.nn.functional as F


def det3x3(mat):
    """calculate the determinant of a 3x3 matrix, support batch."""
    M = mat.reshape(-1, 3, 3).permute(1, 2, 0)

    detM = M[0, 0] * (M[1, 1] * M[2, 2] - M[2, 1] * M[1, 2]) \
        - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0]) \
        + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    return detM.reshape(*mat.size()[:-2]) if mat.dim() > 2 else detM.contiguous()


def inv3x3(mat):
    """calculate the inverse of a 3x3 matrix, support batch."""
    M = mat.reshape(-1, 3, 3).permute(1, 2, 0)

    # Divide the matrix with it's maximum element
    max_vals = M.flatten(0, 1).max(0)[0]
    M = M / max_vals

    adjM = torch.empty_like(M)
    adjM[0, 0] = M[1, 1] * M[2, 2] - M[2, 1] * M[1, 2]
    adjM[0, 1] = M[0, 2] * M[2, 1] - M[0, 1] * M[2, 2]
    adjM[0, 2] = M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]
    adjM[1, 0] = M[1, 2] * M[2, 0] - M[1, 0] * M[2, 2]
    adjM[1, 1] = M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]
    adjM[1, 2] = M[1, 0] * M[0, 2] - M[0, 0] * M[1, 2]
    adjM[2, 0] = M[1, 0] * M[2, 1] - M[2, 0] * M[1, 1]
    adjM[2, 1] = M[2, 0] * M[0, 1] - M[0, 0] * M[2, 1]
    adjM[2, 2] = M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1]

    # Divide the maximum value once more
    invM = adjM / (det3x3(M.permute(2, 0, 1)) * max_vals)
    return invM.permute(2, 0, 1).reshape_as(mat)


def cat_grid_z(grid, fill_value: int = 1):
    """concat z axis of grid at last dim , return shape (B, H, W, 3)"""
    return torch.cat([grid, torch.full_like(grid[..., 0:1], fill_value)], dim=-1)


class LinearizedMutilSample():
    num_grid = 8
    noise_strength = 0.5
    need_push_away = True
    fixed_bias = False

    @classmethod
    def hyperparameters(cls):
        return {'num_grid': cls.num_grid, 'noise_strength': cls.noise_strength,
                'need_push_away': cls.need_push_away, 'fixed_bias': cls.fixed_bias}

    @classmethod
    def set_hyperparameters(cls, **kwargs):
        selfparams = cls.hyperparameters()
        for key, value in kwargs.items():
            if selfparams[key] != value:
                setattr(cls, key, value)
                print('Set Linearized Mutil Sample hyperparam:`%s` to %s' %
                      (key, value))

    @classmethod
    def create_auxiliary_grid(cls, grid, inputsize):
        grid = grid.unsqueeze(1).repeat(1, cls.num_grid, 1, 1, 1)

        WH = grid.new_tensor([[grid.size(-2), grid.size(-3)]])
        grid_noise = torch.randn_like(grid[:, 1:]) / WH * cls.noise_strength
        grid[:, 1:] += grid_noise

        if cls.need_push_away:
            input_h, input_w = inputsize[-2:]
            least_offset = grid.new_tensor([2.0 / input_w, 2.0 / input_h])
            noise = torch.randn_like(grid[:, 1:]) * least_offset
            grid[:, 1:] += noise

        return grid

    @classmethod
    def warp_input(cls, input, auxiliary_grid, padding_mode='zeros', align_corners=False):
        assert input.dim() == 4
        assert auxiliary_grid.dim() == 5

        B, num_grid, H, W = auxiliary_grid.size()[:4]
        inputs = input.unsqueeze(1).repeat(1, num_grid, 1, 1, 1).flatten(0, 1)
        grids = auxiliary_grid.flatten(0, 1).detach()
        warped_input = F.grid_sample(inputs, grids, 'bilinear',
                                     padding_mode, align_corners)
        return warped_input.reshape(B, num_grid, -1, H, W)

    @classmethod
    def linearized_fitting(cls, warped_input, auxiliary_grid):
        assert warped_input.size(1) > 1, 'num of grid should be larger than 1'
        assert warped_input.dim() == 5, 'shape should be: B x Grid x C x H x W'
        assert auxiliary_grid.dim() == 5, 'shape should be: B x Grid x H x W x XY'
        assert warped_input.size(1) == auxiliary_grid.size(1)

        center_image = warped_input[:, 0]
        other_image = warped_input[:, 1:]
        center_grid = auxiliary_grid[:, 0]
        other_grid = auxiliary_grid[:, 1:]

        delta_intensity = other_image - center_image.unsqueeze(1)
        delta_grid = other_grid - center_grid.unsqueeze(1)

        # concat z and reshape to [B, H, W, XY1, Grid-1]
        xT = cat_grid_z(delta_grid).permute(0, 2, 3, 4, 1)
        # calculate dI/dX, euqation(7) in paper
        xTx = xT @ xT.transpose(3, 4)  # [B, H, W, XY1, XY1]
        xTx_inv_xT = inv3x3(xTx) @ xT  # [B, H, W, XY1, Grid-1]

        # prevent manifestation from out-of-bound samples mentioned in section 6.1 of paper
        dW, dH = delta_grid.abs().chunk(2, dim=-1)
        delta_mask = ((dW <= 1.0) * (dH <= 1.0)).permute(0, 2, 3, 4, 1)
        xTx_inv_xT = xTx_inv_xT * delta_mask

        # [B, Grid-1, C, H, W] reshape to [B, H, W, Grid-1, C]
        delta_intensity = delta_intensity.permute(0, 3, 4, 1, 2)
        # gradient_intensity shape: [B, H, W, XY1, C]
        gradient_intensity = xTx_inv_xT @ delta_intensity

        # stop gradient shape: [B, H, W, C, XY1]
        gradient_intensity = gradient_intensity.detach().transpose(3, 4)

        # center_grid shape: [B, H, W, XY1]
        grid_xyz_stop = cat_grid_z(center_grid.detach(), int(cls.fixed_bias))
        gradient_grid = cat_grid_z(center_grid) - grid_xyz_stop

        # map to linearized, equation(2) in paper
        return center_image + (gradient_intensity @ gradient_grid.unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2)

    @classmethod
    def apply(cls, input, grid, padding_mode='zeros', align_corners=False):
        assert input.size(0) == grid.size(0)
        auxiliary_grid = cls.create_auxiliary_grid(grid, input.size())
        warped_input = cls.warp_input(
            input, auxiliary_grid, padding_mode, align_corners)
        linearized_input = cls.linearized_fitting(warped_input, auxiliary_grid)
        return linearized_input


def linearized_grid_sample(input, grid, padding_mode='zeros', align_corners=False,
                           num_grid=8, noise_strength=.5, need_push_away=True, fixed_bias=False):
    """Linearized multi-sampling

    Args:
        input (tensor): (B, C, H, W)
        grid (tensor): (B, H, W, 2)
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        num_grid (int, optional): multisampling. Defaults to 8.
        noise_strength (float, optional): auxiliary noise. Defaults to 0.5.
        need_push_away (bool, optional): pushaway grid. Defaults to True.
        fixed_bias (bool, optional): Defaults to False.

    Returns:
        tensor: linearized sampled input

    Reference:
        paper: https://arxiv.org/abs/1901.07124
        github: https://github.com/vcg-uvic/linearized_multisampling_release
    """
    LinearizedMutilSample.set_hyperparameters(
        num_grid=num_grid, noise_strength=noise_strength, need_push_away=need_push_away, fixed_bias=fixed_bias)
    return LinearizedMutilSample.apply(input, grid, padding_mode, align_corners)


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    original function prototype:
    torch.nn.functional.grid_sample(
        input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    copy from pytorch 1.3.0 source code
    add linearized_grid_sample
    """
    if mode == 'linearized':
        return LinearizedMutilSample.apply(input, grid, padding_mode, align_corners)
    else:
        return F.grid_sample(input, grid, mode, padding_mode, align_corners)


class GeneratorCache(object):
    def __init__(self, func):
        self.func = func
        self._cache = {}

    def __call__(self, size, *args, **kwargs):
        assert isinstance(size, (list, tuple))
        if isinstance(size, list):
            size = tuple(size)
        key = (size, args, tuple(kwargs.items()))
        if key not in self._cache:
            self._cache[key] = self.func(size, *args, **kwargs)
        return self._cache[key]


@GeneratorCache
def meshgrid(size, align_corners=True):
    # type: (List[int]) -> Tensor
    """return meshgrid (B, H, W, 2) of input size(width first, range (-1, -1)~(1, 1))"""
    coords = torch.meshgrid(*[torch.linspace(-1, 1, s)*(1 if align_corners else (s-1)/s)
                              if s > 1 else torch.zeros(1) for s in size[2:]])
    return torch.stack(coords[::-1], dim=-1).repeat(*([size[0]] + [1]*(len(size)-1)))


class Sampler():
    '''a differentiable image sampler which works with theta'''

    def __init__(self, sampling_mode='linearized', padding_mode='border', align_corners=True):
        self.sampling_mode = sampling_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def warp_image(self, input, theta, out_shape=None):
        if input.dim() < 4:
            input = input.unsqueeze(0)
        if theta.dim() < 3:
            theta = theta.unsqueeze(0)
        assert input.size(0) == theta.size(
            0), 'batch size of inputs do not match the batch size of theta'
        if out_shape is None:
            out_shape = input.size()[-2:]
        out_shape = (input.size(0), 1, out_shape[-2], out_shape[-1])
        # create grid for interpolation (in frame coordinates)
        grid = F.affine_grid(theta, out_shape, self.align_corners)
        # sample warped input
        return grid_sample(input, grid, self.sampling_mode, self.padding_mode, self.align_corners)
