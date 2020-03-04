import torch
import torch.nn.functional as F


def cat_grid_z(grid, fill_value: int = 1):
    """concat z axis of grid at last dim , return shape (B, H, W, 3)"""
    return torch.cat([grid, torch.full_like(grid[..., 0:1], fill_value)], dim=-1)


def linearized_grid_sample(input, grid, num_grid=8, noise_strength=0.5, need_push_away=True, fixed_bias=False, **kwargs):
    """Linearized multi-sampling

    Args:
        input (tensor): (B, C, H, W)
        grid (tensor): (B, H, W, 2)
        num_grid (int, optional): multisampling. Defaults to 8.
        noise_strength (float, optional): auxiliary noise. Defaults to 0.5.
        need_push_away (bool, optional): pushaway grid. Defaults to True.
        fixed_bias (bool, optional): Defaults to False.
        others : as torch.nn.functional.grid_sample()

    Returns:
        tensor: linearized sampled input

    Reference:
        paper: https://arxiv.org/abs/1901.07124
        github: https://github.com/vcg-uvic/linearized_multisampling_release
    """
    def create_auxiliary_grid(grid):
        grid = grid.unsqueeze(1)
        other_grid = grid.repeat(1, num_grid-1, 1, 1, 1)

        WH = grid.new_tensor([grid.size(-2), grid.size(-3)])
        other_grid += torch.randn_like(other_grid) / WH * noise_strength

        if need_push_away:
            inputH, inputW = input.size()[-2:]
            least_offset = grid.new_tensor([2.0/inputW, 2.0/inputH])
            other_grid += torch.randn_like(other_grid) * least_offset

        return torch.cat([grid, other_grid], dim=1)

    def warp_input(input, auxiliary_grid):
        assert input.dim() == 4
        assert auxiliary_grid.dim() == 5

        B, num_grid, H, W = auxiliary_grid.size()[:4]
        inputs = input.unsqueeze(1).repeat(1, num_grid, 1, 1, 1).flatten(0, 1)
        grids = auxiliary_grid.flatten(0, 1).detach()
        warped_input = F.grid_sample(inputs, grids, mode='bilinear', **kwargs)
        return warped_input.reshape(B, num_grid, -1, H, W)

    def linearized_fitting(warped_input, auxiliary_grid):
        assert auxiliary_grid.size(
            1) > 1, 'num of grid should be larger than 1'
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
        xTx_inv = xTx.view(-1, 3, 3).inverse().view_as(xTx)
        xTx_inv_xT = xTx_inv.matmul(x)  # [B, H, W, XY1, Grid-1]

        # prevent manifestation from out-of-bound samples mentioned in section 6.1 of paper
        dW, dH = delta_grid.abs().chunk(2, dim=-1)
        delta_mask = ((dW <= 1.0) * (dH <= 1.0)).permute(0, 2, 3, 4, 1)
        xTx_inv_xT = xTx_inv_xT * delta_mask

        # [B, Grid-1, C, H, W] reshape to [B, H, W, Grid-1, C]
        delta_intensity = delta_intensity.permute(0, 3, 4, 1, 2)
        # gradient_intensity shape: [B, H, W, XY1, C]
        gradient_intensity = xTx_inv_xT.matmul(delta_intensity)

        # stop gradient shape: [B, C, H, W, XY1]
        gradient_intensity = gradient_intensity.permute(0, 4, 1, 2, 3).detach()

        # center_grid shape: [B, H, W, XY1]
        grid_xyz_stop = cat_grid_z(center_grid.detach(), int(fixed_bias))
        gradient_grid = cat_grid_z(center_grid) - grid_xyz_stop

        # map to linearized, equation(2) in paper
        return center_image + gradient_intensity.mul(gradient_grid.unsqueeze(1)).sum(-1)

    assert input.size(0) == grid.size(0)
    auxiliary_grid = create_auxiliary_grid(grid)
    warped_input = warp_input(input, auxiliary_grid)
    linearized_input = linearized_fitting(warped_input, auxiliary_grid)
    return linearized_input


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    original function prototype:
    torch.nn.functional.grid_sample(
        input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    copy from pytorch 1.3.0 source code
    add linearized_grid_sample
    """
    if mode == 'linearized':
        warped_img = linearized_grid_sample(
            input, grid, padding_mode=padding_mode, align_corners=align_corners)
    else:
        warped_img = F.grid_sample(
            input, grid, mode, padding_mode=padding_mode, align_corners=align_corners)

    return warped_img


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
        grid = F.affine_grid(theta, out_shape, align_corners=True)
        # sample warped input
        return grid_sample(input, grid, self.sampling_mode, self.padding_mode)
