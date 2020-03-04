import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import perturbation_helper
import utils


class GradientVisualizer(object):
    """GradientVisualizer"""

    def __init__(self, input, target, grid_size=10, criterion='mse', optimizer='SGD', lr=1e-2):
        assert input.dim() == target.dim() == 4
        self.input = input
        self.target = target
        self.out_shape = list(target.size()[-2:])
        self.grid_size = grid_size

        if criterion == 'l1':
            self.criterion = F.l1_loss
        elif criterion in ['mse', 'l2']:
            self.criterion = F.mse_loss
        else:
            raise ValueError('unknown optimization criterion: %s' % criterion)
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
        theta = perturbation_helper.vec2mat_for_translation(motion)
        warped_image = sampler.warp_image(self.input, theta, self.out_shape)
        loss = self.criterion(warped_image, self.target)
        loss.backward()
        optimizer.step()
        return motion

    def get_gradient_vec(self, motion, sampler):
        updated_motion = self.get_updated_motion(motion, sampler)
        gradient_vec = updated_motion - motion
        return gradient_vec

    def get_gradient_grid(self, sampler):
        gradient_grid = []
        for motion in self.create_translation_grid():
            gradient = self.get_gradient_vec(motion, sampler)
            gradient_grid.append(dict(motion=motion, gradient=gradient))
        return gradient_grid

    def draw_gradient_grid(self, sampler, filename=None):
        """draw_gradient_grid"""
        gradient_grid = self.get_gradient_grid(sampler)

        _, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(utils.torch_img_to_np_img(
            self.input[0]), extent=[-1, 1, -1, 1])

        for gradient in gradient_grid:
            ori_point = np.zeros([2], dtype=np.float32)
            base_loc = 0 - gradient['motion'][0].data.cpu().numpy()
            gradient_dir = gradient['gradient'][0].data.cpu().numpy()
            gradient_dir = 0 - utils.unit_vector(gradient_dir)
            gt_dir = utils.unit_vector(ori_point - base_loc)

            angle = utils.angle_between(gradient_dir, gt_dir)
            try:
                cur_color = utils.angle_to_color(angle)
            except ValueError:
                cur_color = [0., 0., 0.]
            gradient_dir = gradient_dir / 10
            ax.arrow(base_loc[0], base_loc[1], gradient_dir[0], gradient_dir[1],
                     head_width=0.05, head_length=0.1, color=cur_color)
        # plt.show()
        if filename is None:
            filename = sampler.sampling_mode
        plt.savefig('./%s.png' % filename)
        plt.close('all')
