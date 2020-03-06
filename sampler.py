import torch
import torch.nn.functional as F


def mat_3x3_inv(mat):
    '''
    calculate the inverse of a 3x3 matrix, support batch.
    :param mat: torch.Tensor -- [input matrix, shape: (B, 3, 3)]
    :return: mat_inv: torch.Tensor -- [inversed matrix shape: (B, 3, 3)]
    '''
    if mat.dim() < 3:
        mat = mat.unsqueeze(0)
    assert mat.size(1) == mat.size(2) == 3

    # Divide the matrix with it's maximum element
    max_vals = mat.max(1, keepdim=True)[0].max(2, keepdim=True)[0]
    mat = mat / max_vals

    inv_det = 1.0 / mat_3x3_det(mat)

    mat_inv = torch.zeros_like(mat)
    mat_inv[:, 0, 0] = (mat[:, 1, 1] * mat[:, 2, 2] -
                        mat[:, 2, 1] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 0, 1] = (mat[:, 0, 2] * mat[:, 2, 1] -
                        mat[:, 0, 1] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 0, 2] = (mat[:, 0, 1] * mat[:, 1, 2] -
                        mat[:, 0, 2] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 1, 0] = (mat[:, 1, 2] * mat[:, 2, 0] -
                        mat[:, 1, 0] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 1, 1] = (mat[:, 0, 0] * mat[:, 2, 2] -
                        mat[:, 0, 2] * mat[:, 2, 0]) * inv_det
    mat_inv[:, 1, 2] = (mat[:, 1, 0] * mat[:, 0, 2] -
                        mat[:, 0, 0] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 2, 0] = (mat[:, 1, 0] * mat[:, 2, 1] -
                        mat[:, 2, 0] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 2, 1] = (mat[:, 2, 0] * mat[:, 0, 1] -
                        mat[:, 0, 0] * mat[:, 2, 1]) * inv_det
    mat_inv[:, 2, 2] = (mat[:, 0, 0] * mat[:, 1, 1] -
                        mat[:, 1, 0] * mat[:, 0, 1]) * inv_det

    # Divide the maximum value once more
    mat_inv = mat_inv / max_vals
    return mat_inv


def mat_3x3_det(mat):
    '''
    calculate the determinant of a 3x3 matrix, support batch.
    '''
    if mat.dim() < 3:
        mat = mat.unsqueeze(0)
    assert mat.size(1) == mat.size(2) == 3

    det = mat[:, 0, 0] * (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) \
        - mat[:, 0, 1] * (mat[:, 1, 0] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 0]) \
        + mat[:, 0, 2] * (mat[:, 1, 0] * mat[:, 2, 1] -
                          mat[:, 1, 1] * mat[:, 2, 0])
    return det


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
        for key, item in kwargs.items():
            if selfparams[key] != item:
                setattr(cls, key, item)
                print('Set Linearized Mutil Sample hyperparam:`%s` to %s' %
                      (key, item))

    @classmethod
    def create_auxiliary_grid(cls, grid, inputsize):
        grid = grid.unsqueeze(1).repeat(1, cls.num_grid, 1, 1, 1)

        WH = grid.new_tensor([[grid.size(-2), grid.size(-3)]])
        grid_noise = torch.randn_like(grid[:, 1:]) / WH * cls.noise_strength
        grid[:, 1:] += grid_noise

        if cls.need_push_away:
            input_H, input_W = inputsize[-2:]
            least_offset = grid.new_tensor([2.0 / input_W, 2.0 / input_H])
            noise = torch.randn_like(grid[:, 1:]) * least_offset
            grid[:, 1:] += noise

        return grid

    @classmethod
    def warp_input(cls, input, auxiliary_grid, padding_mode='zeros'):
        assert input.dim() == 4
        assert auxiliary_grid.dim() == 5

        B, num_grid, H, W = auxiliary_grid.size()[:4]
        inputs = input.unsqueeze(1).repeat(1, num_grid, 1, 1, 1).flatten(0, 1)
        grids = auxiliary_grid.flatten(0, 1).detach()
        warped_input = F.grid_sample(inputs, grids, mode='bilinear',
                                     padding_mode=padding_mode, align_corners=True)
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
        x = cat_grid_z(delta_grid).permute(0, 2, 3, 4, 1)
        # calculate dI/dX, euqation(7) in paper
        xTx = x.matmul(x.transpose(3, 4))  # [B, H, W, XY1, XY1]
        xTx_inv = mat_3x3_inv(xTx.view(-1, 3, 3)).view_as(xTx)
        xTx_inv_xT = xTx_inv.matmul(x)  # [B, H, W, XY1, Grid-1]

        # prevent manifestation from out-of-bound samples mentioned in section 6.1 of paper
        dW, dH = delta_grid.abs().chunk(2, dim=-1)
        delta_mask = ((dW <= 1.0) * (dH <= 1.0)).permute(0, 2, 3, 4, 1)
        xTx_inv_xT = xTx_inv_xT * delta_mask

        # [B, Grid-1, C, H, W] reshape to [B, H, W, Grid-1, C]
        delta_intensity = delta_intensity.permute(0, 3, 4, 1, 2)
        # gradient_intensity shape: [B, H, W, XY1, C]
        gradient_intensity = xTx_inv_xT.matmul(delta_intensity)

        # stop gradient shape: [B, H, W, C, XY1]
        gradient_intensity = gradient_intensity.detach().transpose(3, 4)

        # center_grid shape: [B, H, W, XY1]
        grid_xyz_stop = cat_grid_z(center_grid.detach(), int(cls.fixed_bias))
        gradient_grid = cat_grid_z(center_grid) - grid_xyz_stop

        # map to linearized, equation(2) in paper
        return center_image + gradient_intensity.matmul(gradient_grid.unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2)

    @classmethod
    def apply(cls, input, grid, padding_mode='zeros'):
        assert input.size(0) == grid.size(0)
        auxiliary_grid = cls.create_auxiliary_grid(grid, input.size())
        warped_input = cls.warp_input(input, auxiliary_grid, padding_mode)
        linearized_input = cls.linearized_fitting(warped_input, auxiliary_grid)
        return linearized_input


def linearized_grid_sample(input, grid, padding_mode='zeros',
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
    return LinearizedMutilSample.apply(input, grid, padding_mode)


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    original function prototype:
    torch.nn.functional.grid_sample(
        input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    copy from pytorch 1.3.0 source code
    add linearized_grid_sample
    """
    if mode == 'linearized':
        warped_img = LinearizedMutilSample.apply(
            input, grid, padding_mode=padding_mode)
    else:
        warped_img = F.grid_sample(
            input, grid, mode, padding_mode=padding_mode, align_corners=align_corners)

    return warped_img


def meshgrid(size):
    # type: (List[int]) -> Tensor
    """return meshgrid (B, H, W, 2) of input size(width first, range (-1, -1)~(1, 1))"""
    coordh, coordw = torch.meshgrid(torch.linspace(-1, 1, size[-2]),
                                    torch.linspace(-1, 1, size[-1]))
    return torch.stack([coordw, coordh], dim=2).repeat(size[0], 1, 1, 1)


def homography_grid(matrix, size):
    # type: (Tensor, List[int]) -> Tensor
    grid = cat_grid_z(meshgrid(size).to(matrix.device))  # B, H, W, 3
    homography = grid.flatten(1, 2).bmm(matrix.transpose(1, 2)).view_as(grid)
    grid, ZwarpHom = homography.split([2, 1], dim=-1)
    return grid / ZwarpHom.add(1e-8)


def transform_to_grid(matrix, size):
    if matrix.size(1) == 2:
        return F.affine_grid(matrix, size, align_corners=True)
    else:
        return homography_grid(matrix, size)


class Sampler():
    '''a differentiable image sampler which works with theta'''

    def __init__(self, sampling_mode='linearized', padding_mode='border'):
        self.sampling_mode = sampling_mode
        self.padding_mode = padding_mode

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
        grid = transform_to_grid(theta, out_shape)
        # sample warped input
        return grid_sample(input, grid, self.sampling_mode, self.padding_mode)
