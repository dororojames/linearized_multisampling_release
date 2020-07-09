import matplotlib.pyplot as plt
import torch
from torch.nn.functional import affine_grid, l1_loss, mse_loss

import utils
from perturbation_helper import vec2mat
from sampler import grid_sample, meshgrid


class GradientVisualizer(object):
    """GradientVisualizer

    Args:
        input (torch.Tensor): image (1, 3, H, W)
        target_mo (torch.Tensor): target motion (1, 2). Default: (0, 0)
        out_shape (pair of int): target shape. Default: input.size()
        grid_size (int): gradient grid samlpe size, will sample (N x N) grids. Default: 10
        criterion (str): loss funciton in 'l1' | 'l2' | 'mse'. Default: 'mse'
        optimizer (str): optimizer in 'SGD' | 'Adam'. Default: 'SGD'
        lr (float): learning rate for optimizer. Default: 1e-2
    """

    def __init__(self, input, target_mo=None, out_shape=None,
                 grid_size=10, criterion='mse', optimizer='SGD', lr=1e-2):
        size = input.size() if out_shape is None else [1, 1] + list(out_shape)
        target_mo = torch.zeros(1, 2) if target_mo is None else target_mo
        target_grid = affine_grid(vec2mat(target_mo), size)
        target = grid_sample(input, target_grid, align_corners=True)

        assert input.dim() == target.dim() == 4
        self.input = input
        self.target_mo = target_mo
        self.target = target
        self.out_shape = target.size()[-2:]
        self.grid = meshgrid((1, 1, grid_size, grid_size)).view(-1, 1, 2)

        if criterion == 'l1':
            self.criterion = l1_loss
        elif criterion in ['mse', 'l2']:
            self.criterion = mse_loss
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam
        self.lr = lr

    def get_updated_motion(self, motion, sampler):
        motion = motion.clone().detach().requires_grad_()
        optimizer = self.optimizer([motion], self.lr)

        optimizer.zero_grad()
        warped_image = sampler.warp_image(
            self.input, vec2mat(motion), self.out_shape)
        loss = self.criterion(warped_image, self.target)
        loss.backward()
        optimizer.step()
        return motion.detach()

    def get_gradient_grid(self, sampler):
        gradient_grid = []
        for motion in self.grid:
            updated_motion = self.get_updated_motion(motion, sampler)
            gradient = updated_motion - motion
            gradient_grid.append(dict(motion=motion, gradient=gradient))
        return gradient_grid

    def draw_gradient_grid(self, sampler, filename=None):
        """draw_gradient_grid"""
        gradient_grid = self.get_gradient_grid(sampler)

        fig, ax = plt.subplots()
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.imshow(utils.torch_img_to_np_img(
            self.input[0]), extent=[-1, 1, -1, 1])

        total_angle = 0
        target_mo = self.target_mo[0].cpu().numpy()
        for gradient in gradient_grid:
            base_loc = gradient['motion'][0].neg().cpu().numpy()
            gradient = gradient['gradient'][0].neg().cpu().numpy()
            gradient_dir = utils.unit_vector(gradient)
            gt_dir = utils.unit_vector(target_mo - base_loc)
            angle = utils.angle_between(gradient_dir, gt_dir)

            total_angle += angle
            try:
                cur_color = utils.angle_to_color(angle)
            except ValueError:
                cur_color = [0., 0., 0.]

            ax.arrow(*base_loc, *(gradient_dir/10),
                     head_width=0.05, head_length=0.1, color=cur_color)
        # plt.show()
        if filename is None:
            filename = sampler.sampling_mode
        print(filename, 'mean error angle:', total_angle/len(gradient_grid))
        plt.savefig('./%s.png' % filename)
        plt.close(fig)
