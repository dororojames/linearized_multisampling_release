import imageio
import matplotlib.pyplot as plt
import torch

import gradient_visualizer
import sampler
import sampler_o
import utils


def torchseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


image_path = './imgs/cute.jpg'
cute_cat = imageio.imread(image_path)
cute_cat = cute_cat / 255.0
# plt.imshow(cute_cat)
# plt.show()
# convert np image to torch image
cute_cat = utils.np_img_to_torch_img(cute_cat).float()
print(cute_cat.shape)

trans_mat = torch.tensor([[[0.6705,  0.4691, -0.1369],
                           [-0.4691,  0.6705, -0.0432]]], device='cpu')
out_shape = [128, 128]

bilinear_sampler = sampler.DifferentiableImageSampler(
    'bilinear', 'zeros')
bilinear_transformed_image = bilinear_sampler.warp_image(
    cute_cat, trans_mat, out_shape=out_shape)
print(bilinear_transformed_image.shape)
# convert torch image to np
# bilinear_transformed_image = utils.torch_img_to_np_img(bilinear_transformed_image)
# plt.imshow(bilinear_transformed_image[0])
# plt.show()

torchseed(66)
linearized_sampler = sampler.DifferentiableImageSampler(
    'linearized', 'zeros')
linearized_transformed_image = linearized_sampler.warp_image(
    cute_cat, trans_mat, out_shape=out_shape)
print(linearized_transformed_image.shape)
# convert torch image to np
# linearized_transformed_image = utils.torch_img_to_np_img(linearized_transformed_image)
# plt.imshow(linearized_transformed_image[0])
# plt.show()


class FakeOptions():
    pass


opt = FakeOptions()
opt.padding_mode = 'zeros'
opt.grid_size = 10
opt.optim_criterion = 'mse'
opt.optim_lr = 1e-2
opt.out_shape = [16, 16]

gradient_visualizer_instance = gradient_visualizer.GradientVisualizer(opt)
gradient_visualizer_instance.draw_gradient_grid(
    cute_cat[None], bilinear_sampler)
gradient_visualizer_instance.draw_gradient_grid(
    cute_cat[None], linearized_sampler)