import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import utils
from perturbation_helper import mat2vec, vec2mat
from sampler import transform_to_grid


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
        target_grid = transform_to_grid(vec2mat(target_mo), size)
        target = F.grid_sample(
            input, target_grid, mode='bilinear', align_corners=True)

        assert input.dim() == target.dim() == 4
        self.input = input
        self.target_mo = target_mo
        self.target = target
        self.out_shape = list(target.size()[-2:])
        self.grid_size = grid_size

        if criterion == 'l1':
            self.criterion = F.l1_loss
        elif criterion in ['mse', 'l2']:
            self.criterion = F.mse_loss
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam
        self.lr = lr

    def create_translation_grid(self):
        y_steps, x_steps = torch.meshgrid(torch.linspace(-1, 1, steps=self.grid_size),
                                          torch.linspace(-1, 1, steps=self.grid_size))
        return torch.stack([x_steps, y_steps], dim=-1).view(-1, 1, 2)

    def get_updated_motion(self, motion, sampler):
        motion = motion.clone().detach().requires_grad_(True)
        optimizer = self.optimizer([motion], self.lr)

        optimizer.zero_grad()
        warped_image = sampler.warp_image(
            self.input, vec2mat(motion), self.out_shape)
        loss = self.criterion(warped_image, self.target)
        loss.backward()
        optimizer.step()
        return motion

    def get_gradient_grid(self, sampler):
        gradient_grid = []
        for motion in self.create_translation_grid():
            updated_motion = self.get_updated_motion(motion, sampler)
            gradient = updated_motion - motion
            gradient_grid.append(dict(motion=motion, gradient=gradient))
        return gradient_grid

    def draw_gradient_grid(self, sampler, filename=None):
        """draw_gradient_grid"""
        gradient_grid = self.get_gradient_grid(sampler)

        _, ax = plt.subplots()
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.imshow(utils.torch_img_to_np_img(
            self.input[0]), extent=[-1, 1, -1, 1])

        total_angle = 0
        target_mo = self.target_mo[0].data.cpu().numpy()
        for gradient in gradient_grid:
            base_loc = gradient['motion'][0].data.neg().cpu().numpy()
            gradient = gradient['gradient'][0].data.neg().cpu().numpy()
            gradient_dir = utils.unit_vector(gradient)
            gt_dir = utils.unit_vector(target_mo - base_loc)
            angle = utils.angle_between(gradient_dir, gt_dir)

            total_angle += angle
            try:
                cur_color = utils.angle_to_color(angle)
            except ValueError:
                cur_color = [0., 0., 0.]
            gradient_dir /= 10
            ax.arrow(base_loc[0], base_loc[1], gradient_dir[0], gradient_dir[1],
                     head_width=0.05, head_length=0.1, color=cur_color)
        # plt.show()
        if filename is None:
            filename = sampler.sampling_mode
        print(filename, angle/len(gradient_grid))
        plt.savefig('./%s.png' % filename)
        plt.close('all')
