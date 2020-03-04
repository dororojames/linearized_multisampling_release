import torch
from torchvision.utils import save_image

import perturbation_helper
import sampler
import sampler_o
import utils
from gradient_visualizer import GradientVisualizer


def torchseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    image_path = './imgs/cute.jpg'
    cute_cat = utils.loadimage(image_path).unsqueeze(0)
    print(cute_cat.shape)

    trans_mat = torch.FloatTensor([[[0.6705,  0.4691, -0.1369],
                                    [-0.4691,  0.6705, -0.0432]]])
    out_shape = [128, 128]

    bilinear_sampler = sampler.Sampler('bilinear', 'zeros')
    bilinear_tarnsformed = bilinear_sampler.warp_image(
        cute_cat, trans_mat, out_shape=out_shape)
    save_image(bilinear_tarnsformed, 'bilinear_transformed.png')
    # utils.showimg(bilinear_tarnsformed)

    # torchseed(666)
    # linearized_sampler_o = sampler_o.Sampler(
    #     'linearized', 'zeros')
    # linearized_tarnsformed_o = linearized_sampler_o.warp_image(
    #     cute_cat, trans_mat, out_shape=out_shape)
    # save_image(linearized_tarnsformed_o, 'linearized_transformed_ori.png')
    # # utils.showimg(linearized_tarnsformed_o)

    torchseed(666)
    linearized_sampler = sampler.Sampler('linearized', 'zeros')
    linearized_tarnsformed_t = linearized_sampler.warp_image(
        cute_cat, trans_mat, out_shape=out_shape)
    save_image(linearized_tarnsformed_t, 'linearized_tarnsformed.png')
    # utils.showimg(linearized_tarnsformed_t)

    # print(linearized_tarnsformed_o.equal(linearized_tarnsformed_t))

    # target_mo = torch.rand(1, 2)*2 - 1
    # target_mat = perturbation_helper.vec2mat_for_translation(target_mo)
    target_mat = perturbation_helper.gen_identity_mat(1)
    target = bilinear_sampler.warp_image(cute_cat, target_mat, [16, 16])

    visualizer = GradientVisualizer(input=cute_cat, target=target)
    visualizer.draw_gradient_grid(bilinear_sampler)
    # torchseed(666)
    # visualizer.draw_gradient_grid(linearized_sampler_o, 'linearized_ori')
    torchseed(666)
    visualizer.draw_gradient_grid(linearized_sampler)
