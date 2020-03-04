import colorsys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets.folder import default_loader as imgloader
from torchvision.transforms.functional import to_tensor


def embed_breakpoint(terminate=True):
    embedding = ('import IPython\n'
                 'import matplotlib.pyplot as plt\n'
                 'IPython.embed()\n'
                 )
    if terminate:
        embedding += (
            'assert 0, \'force termination\'\n'
        )

    return embedding


def torchseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loadimage(path):
    return to_tensor(imgloader(path))


def torch_img_to_np_img(torch_img):
    '''convert a torch image to matplotlib-able numpy image
    torch use Channels x Height x Width
    numpy use Height x Width x Channels
    Arguments:
        torch_img {[type]} -- [description]
    '''
    assert isinstance(
        torch_img, torch.Tensor), 'cannot process data type: {0}'.format(type(torch_img))
    if torch_img.dim() == 4 and (torch_img.size(1) == 3 or torch_img.size(1) == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
    if torch_img.dim() == 3 and (torch_img.size(0) == 3 or torch_img.size(0) == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
    elif torch_img.dim() == 2:
        return torch_img.detach().cpu().numpy()
    else:
        raise ValueError('cannot process this image')


def showimg(torch_img):
    """convert torch image to np"""
    if torch_img.dim() == 4:
        torch_img = torch_img[0]
    img = torch_img_to_np_img(torch_img.clamp(min=0, max=1))
    plt.imshow(img)
    plt.show()


def np_img_to_torch_img(np_img):
    """convert a numpy image to torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width

    Arguments:
        np_img {[type]} -- [description]
    """
    assert isinstance(
        np_img, np.ndarray), 'cannot process data type: {0}'.format(type(np_img))
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return torch.from_numpy(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return torch.from_numpy(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return torch.from_numpy(np_img)
    else:
        raise ValueError(
            'cannot process this image with shape: {0}'.format(np_img.shape))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_to_color(angle):
    red_hue, _, _ = colorsys.rgb_to_hsv(1, 0, 0)
    green_hue, _, _ = colorsys.rgb_to_hsv(0, 1, 0)
    cur_hue = np.interp(angle, (0, np.pi), (green_hue, red_hue))
    cur_color = colorsys.hsv_to_rgb(cur_hue, 1, 1)
    return cur_color
