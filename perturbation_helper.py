"""
function for generation perturbation
modified from: https://github.com/chenhsuanlin/inverse-compositional-STN
"""
import numpy as np
import torch
from torch import stack


def gen_perturbation_vec(opt, num_pert: int):
    """generate homography perturbation

    Arguments:
        opt      -- [user defined options]
        num_pert -- [generate how many perturbations]
    Returns:
        transformation matrix, shape is (B, warp_dim)
    """
    assert opt.need_pert, 'please enable perturbation'
    if opt.warp_type == 'translation':
        perturbation_vec = gen_pert_for_translation(opt, num_pert)
    elif opt.warp_type == 'trans+rot':
        perturbation_vec = gen_pert_for_trans_rot(opt, num_pert)
    elif opt.warp_type == 'similarity':
        perturbation_vec = gen_pert_for_similarity(opt, num_pert)
    else:
        raise ValueError('unknown warping method')
    return perturbation_vec


def gen_perturbation_mat(opt, num_pert: int):
    """generate homography perturbation matrix

    Arguments:
        opt      -- [user defined options]
        num_pert -- [generate how many perturbations]
    Returns:
        transformation matrix, shape is (B, 3, 3)
    """
    perturbation_vec = gen_perturbation_vec(opt, num_pert)
    perturbation_mat = vec2mat(perturbation_vec)
    return perturbation_mat


def gen_identity_mat(num_ident: int):
    """
    :param num_ident: number of 2x3 identity matrix
    :return: identity matrix, shape is (B, 2, 3)
    """
    identity = torch.eye(2, 3).repeat(num_ident, 1, 1)
    return identity


def gen_random_rotation(opt, num_pert: int):
    rad = float(opt.rotation_range) / 180.0 * np.pi
    if opt.pert_distribution == 'normal':
        theta = torch.randn(num_pert, 1).clamp_(min=-2, max=2)
    elif opt.pert_distribution == 'uniform':
        theta = torch.rand(num_pert, 1)*2 - 1
    else:
        raise NotImplementedError('unknown sampling distribution')
    return theta * rad


def gen_random_translation(opt, num_pert):
    if opt.pert_distribution == 'normal':
        tx = torch.randn(num_pert, 1).clamp_(min=-2, max=2)
    elif opt.pert_distribution == 'uniform':
        tx = torch.rand(num_pert, 1)*2 - 1
    else:
        raise NotImplementedError('unknown sampling distribution')
    return tx * opt.translation_range


def gen_random_scaling(opt, num_pert):
    if opt.pert_distribution == 'normal':
        sx = torch.randn(num_pert, 1).clamp_(min=-2, max=2)
    elif opt.pert_distribution == 'uniform':
        sx = torch.rand(num_pert, 1)*2 - 1
    else:
        raise NotImplementedError('unknown sampling distribution')
    return sx * opt.scaling_range


def gen_pert_for_translation(opt, num_pert):
    tx = gen_random_translation(opt, num_pert)
    ty = gen_random_translation(opt, num_pert)
    # make it a torch vector
    perturbation_vec = stack([tx, ty], dim=-1)
    return perturbation_vec


def gen_pert_for_trans_rot(opt, num_pert):
    theta = gen_random_rotation(opt, num_pert)
    tx = gen_random_translation(opt, num_pert)
    ty = gen_random_translation(opt, num_pert)
    # make it a torch vector
    perturbation_vec = stack([theta, tx, ty], dim=-1)
    return perturbation_vec


def gen_pert_for_similarity(opt, num_pert):
    theta = gen_random_rotation(opt, num_pert)
    s = gen_random_scaling(opt, num_pert)
    tx = gen_random_translation(opt, num_pert)
    ty = gen_random_translation(opt, num_pert)
    # make it a torch vector
    perturbation_vec = stack([theta, s, tx, ty], dim=-1)
    return perturbation_vec


def vec2mat(vec):
    """covert a transformation vector to transformation matrix,

    Args:
        vec -- [transformation vector: , shape: (B, n)], where n is the number of warping parameters

    Returns:
        mat -- [transformation matrix, shape: (B, 2, 3)]
    """
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    assert vec.dim() == 2
    B = vec.size(0)
    O = vec.new_zeros(B)
    I = vec.new_ones(B)

    if vec.size(1) == 2:  # "translation"
        tx, ty = torch.unbind(vec, dim=1)
        transformation_mat = stack([stack([I, O, tx], dim=-1),
                                    stack([O, I, ty], dim=-1)], dim=1)
    elif vec.size(1) == 3:  # trans_rot
        theta, tx, ty = vec.unbind(dim=1)
        cos, sin = torch.cos(theta), torch.sin(theta)
        transformation_mat = stack([stack([cos, -sin, tx], dim=-1),
                                    stack([sin,  cos, ty], dim=-1)], dim=1)
    elif vec.size(1) == 4:  # "similarity"
        pc, ps, tx, ty = torch.unbind(vec, dim=1)
        transformation_mat = stack([stack([pc, -ps, tx], dim=-1),
                                    stack([ps,  pc, ty], dim=-1)], dim=1)
    elif vec.size(1) == 6:  # "affine"
        p1, p2, tx, p4, p5, ty = torch.unbind(vec, dim=1)
        transformation_mat = stack([stack([p1, p2, tx], dim=-1),
                                    stack([p4, p5, ty], dim=-1)], dim=1)
    elif vec.size(1) == 8:  # "homography"
        vec = torch.cat([vec, O.unsqueeze(0)], dim=-1)
        transformation_mat = vec.view(-1, 3, 3)
    else:
        raise NotImplementedError('unknown warping method')
    return transformation_mat


def mat2vec(mat, warpType):
    """convert warp matrix to parameters"""
    row0, row1, row2 = mat.unbind(dim=1)
    e00, e01, e02 = row0.unbind(dim=1)
    e10, e11, e12 = row1.unbind(dim=1)
    e20, e21,   _ = row2.unbind(dim=1)
    if warpType == "translation":
        p = stack([e02, e12], dim=1)
    elif warpType == "similarity":
        p = stack([e00, e10, e02, e12], dim=1)
    elif warpType == "affine":
        p = stack([e00, e01, e02, e10, e11, e12], dim=1)
    elif warpType == "homography":
        p = stack([e00, e01, e02, e10, e11, e12, e20, e21], dim=1)
    else:
        raise NotImplementedError('unknown warping method')
    return p


def compose(p, dp, warpType):
    """compute composition of warp parameters"""
    pMtrx = vec2mat(p)
    dpMtrx = vec2mat(dp)
    pMtrxNew = dpMtrx.bmm(pMtrx)
    pMtrxNew = pMtrxNew / pMtrxNew[:, 2:3, 2:3]
    pNew = mat2vec(pMtrxNew, warpType)
    return pNew


def inverse(p, warpType):
    """compute inverse of warp parameters"""
    pMtrx = vec2mat(p)
    pInvMtrx = pMtrx.inverse()
    pInv = mat2vec(pInvMtrx, warpType)
    return pInv
