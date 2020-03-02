import imageio
import matplotlib.pyplot as plt
import torch

from gradient_visualizer import GradientVisualizer
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


gradient_visualizer = GradientVisualizer()
gradient_visualizer.draw_gradient_grid(cute_cat, bilinear_sampler)
gradient_visualizer.draw_gradient_grid(cute_cat, linearized_sampler)
