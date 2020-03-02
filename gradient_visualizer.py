import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import perturbation_helper
import utils


class GradientVisualizer(object):
    """GradientVisualizer"""

    def __init__(self, grid_size=10, criterion='mse', lr=1e-2, out_shape=[16, 16]):
        self.grid_size = grid_size
        self.criterion = criterion
        self.lr = lr
        self.out_shape = out_shape

    def create_translation_grid(self, resolution=None):
        if resolution is None:
            resolution = self.grid_size
        y_steps, x_steps = torch.meshgrid(torch.linspace(-1, 1, steps=resolution),
                                          torch.linspace(-1, 1, steps=resolution))
        return torch.stack([x_steps, y_steps], dim=-1).view(-1, 1, 2)

    def build_criterion(self):
        if self.criterion == 'l1loss':
            criterion = torch.nn.L1Loss()
        elif self.criterion == 'mse':
            criterion = torch.nn.MSELoss()
        else:
            raise ValueError('unknown optimization criterion: {0}'.format(
                self.criterion))
        return criterion

    def build_gd_optimizer(self, params):
        return torch.optim.SGD([params], lr=self.lr)

    def get_next_translation_vec(self, ori_image, translation_vec, image_warper):
        translation_vec = translation_vec.clone().detach().requires_grad_(True)
        translation_mat = perturbation_helper.vec2mat_for_translation(
            translation_vec)
        criterion = self.build_criterion()
        optimizer = self.build_gd_optimizer(params=translation_vec)

        target = F.interpolate(
            ori_image, size=self.out_shape, mode='bilinear', align_corners=True).detach()
        warped_image = image_warper.warp_image(
            ori_image, translation_mat, self.out_shape)

        loss = criterion(warped_image, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return translation_vec

    def get_gradient_over_translation_vec(self, ori_image, translation_vec, image_warper):
        next_translation_vec = self.get_next_translation_vec(
            ori_image, translation_vec, image_warper)
        moving_dir = next_translation_vec - translation_vec
        return moving_dir

    def get_gradient_grid(self, ori_image, image_warper):
        gradient_grid = []
        translation_grid = self.create_translation_grid()
        for translation_vec in translation_grid:
            cur_gradient = self.get_gradient_over_translation_vec(
                ori_image, translation_vec, image_warper)
            gradient_pack = {
                'translation_vec': translation_vec, 'gradient': cur_gradient}
            gradient_grid.append(gradient_pack)
        return gradient_grid

    def draw_gradient_grid(self, ori_image, image_warper):
        gradient_grid = self.get_gradient_grid(
            ori_image.unsqueeze(0), image_warper)

        fig, ax = plt.subplots()
        ax.axis('off')
        ori_image_show = utils.torch_img_to_np_img(ori_image)
        ax.imshow(ori_image_show, extent=[-1, 1, -1, 1])

        for gradient in gradient_grid:
            ori_point = np.zeros([2], dtype=np.float32)
            base_loc = 0 - (gradient['translation_vec'])[0].data.cpu().numpy()
            gradient_dir = (gradient['gradient'])[0].data.cpu().numpy()
            gradient_dir = 0 - utils.unit_vector(gradient_dir)
            gt_dir = ori_point - base_loc
            gt_dir = utils.unit_vector(gt_dir)

            angle = utils.angle_between(gradient_dir, gt_dir)
            try:
                cur_color = utils.angle_to_color(angle)
            except ValueError:
                cur_color = [0., 0., 0.]
            gradient_dir = gradient_dir / 10
            ax.arrow(base_loc[0], base_loc[1], gradient_dir[0], gradient_dir[1],
                     head_width=0.05, head_length=0.1, color=cur_color)
        # plt.show()
        plt.savefig('./%s.png' % image_warper.sampling_mode)
        plt.close('all')
