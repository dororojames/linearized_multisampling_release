import torch
import torch.nn.functional as F


######### Utils to minimize dependencies #########
# Move utils to another file if you want

def has_nan(x) -> bool:
    '''
    check whether x contains nan.
    :param x: torch or numpy var.
    :return: single bool, True -> x containing nan, False -> ok.
    '''
    return torch.isnan(x).any()


def set_nan_to_zero(input, name='input'):
    if has_nan(input):
        print('nan val in %s! set to zeros' % name)
        nidx = torch.isnan(input)
        return input.masked_fill(nidx, 0)
    return input


######### Linearized multi-sampling #########
def linearized_grid_sample(input, grid, num_grid=8, noise_strength=0.5, need_push_away=True, fixed_bias=False, **kwargs):
    """Linearized multi-sampling

    Args:
        input (tensor): (B, C, H, W)
        grid (tensor): (B, 3, 2)
        num_grid (int, optional): multisampling. Defaults to 8.
        noise_strength (float, optional): auxiliary noise. Defaults to 0.5.
        need_push_away (bool, optional): pushaway grid. Defaults to True.
        fixed_bias (bool, optional): Defaults to False.

    Returns:
        tensor: linearized sampled input
    """
    def create_auxiliary_grid(grid, inputWH):
        grid = grid.unsqueeze(1).repeat(1, num_grid, 1, 1, 1)

        WH = grid.new_tensor([[grid.size(2), grid.size(2)]])
        grid_noise = torch.randn_like(grid[:, 1:]) / WH * noise_strength
        grid[:, 1:] = grid[:, 1:] + grid_noise
        if need_push_away:
            least_offset = grid.new_tensor([2.0/inputWH[-1], 2.0/inputWH[-2]])
            noise = torch.randn_like(grid[:, 1:]) * least_offset
            grid[:, 1:] = grid[:, 1:] + noise
        return grid

    def warp_input(input, grid):
        assert input.dim() == 4
        assert grid.dim() == 5
        assert input.size(0) == grid.size(0)

        B, num_grid, H, W = grid.size()[:4]
        input = input.repeat_interleave(num_grid, 0)
        grid = grid.flatten(0, 1).detach()
        warped_input = F.grid_sample(input, grid, mode='bilinear', **kwargs)
        return warped_input.reshape(B, num_grid, -1, H, W)

    def cat_grid_z(grid, fill_value=1):
        return torch.cat([grid, torch.full_like(grid[..., 0:1], fill_value)], dim=-1)

    def linearized_fitting(input, grid):
        assert input.dim() == 5, 'shape should be: B x Grid x C x H x W'
        assert grid.dim() == 5, 'shape should be: B x Grid x H x W x XY'
        assert input.size(0) == grid.size(0)
        assert input.size(1) == grid.size(1)
        assert input.size(1) > 1, 'num of grid should be larger than 1'

        center_image = input[:, 0]
        other_image = input[:, 1:]
        center_grid = grid[:, 0]
        other_grid = grid[:, 1:]

        delta_intensity = other_image - center_image.unsqueeze(1)
        delta_grid = other_grid - center_grid.unsqueeze(1)

        delta_mask = (delta_grid[..., 0:1] >= -1.0) * (delta_grid[..., 0:1] <= 1.0) * (
            delta_grid[..., 1:2] >= -1.0) * (delta_grid[..., 1:2] <= 1.0)
        delta_grid = cat_grid_z(delta_grid) * delta_mask.float()

        # reshape to [B, H, W, Grid-1, XY1]
        delta_grid = delta_grid.permute(0, 2, 3, 4, 1)
        # calculate dI/dX, euqation(7) in paper
        xTx = delta_grid.matmul(delta_grid.transpose(3, 4))
        # take inverse
        xTx_inv = xTx.view(-1, 3, 3).inverse().view(xTx.shape)
        xTx_inv_xT = xTx_inv.matmul(delta_grid)  # [B, H, W, XY1, Grid-1]

        # [B, Grid-1, C, H, W] reshape to [B, H, W, Grid-1, C]
        delta_intensity = delta_intensity.permute(0, 3, 4, 1, 2)
        # gradient_intensity shape: [B, H, W, XY1, C]
        gradient_intensity = xTx_inv_xT.matmul(delta_intensity)

        gradient_intensity = set_nan_to_zero(
            gradient_intensity, 'gradient_intensity')

        # stop gradient shape: [B, C, H, W, XY1]
        gradient_intensity = gradient_intensity.permute(0, 4, 1, 2, 3).detach()

        # center_grid shape: [B, H, W, XY1]
        grid_xyz_stop = cat_grid_z(center_grid, int(fixed_bias))
        gradient_grid = cat_grid_z(center_grid) - grid_xyz_stop.detach()

        # map to linearized, equation(2) in paper
        return center_image + gradient_intensity.mul(gradient_grid.unsqueeze(1)).sum(-1)

    auxiliary_grid = create_auxiliary_grid(grid, input.size()[-2:])
    warped_input = warp_input(input, auxiliary_grid)
    out = linearized_fitting(warped_input, auxiliary_grid)
    return out


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    '''
    original function prototype:
    torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')
    copy from pytorch 1.3.0 source code
    add linearized_grid_sample
    '''
    if mode == 'linearized':
        warped_img = linearized_grid_sample(
            input, grid, padding_mode=padding_mode, align_corners=True)
    else:
        warped_img = F.grid_sample(
            input, grid, mode, padding_mode=padding_mode, align_corners=align_corners)

    warped_img = set_nan_to_zero(warped_img, 'warped input')

    return warped_img


class DifferentiableImageSampler():
    '''a differentiable image sampler which works with theta'''

    def __init__(self, sampling_mode, padding_mode):
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
        grid = F.affine_grid(theta, out_shape, align_corners=True)
        # sample warped input
        return grid_sample(input, grid, self.sampling_mode, self.padding_mode)