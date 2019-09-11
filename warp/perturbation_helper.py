"""
function for generation perturbation
modified from: https://github.com/chenhsuanlin/inverse-compositional-STN
"""
import numpy as np
import torch

from utils import utils


def gen_perturbation_vec(opt, num_pert: int):
    """generate homography perturbation

    Arguments:
        opt      -- [user defined options]
        num_pert -- [generate how many perturbations]
    Returns:
        transformation matrix, shape is (B, warp_dim)
    """
    # TODO: remove np, use torch
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
    perturbation_mat = vec2mat(opt, perturbation_vec)
    return perturbation_mat


def gen_identity_mat(num_ident: int):
    """
    :param num_ident: number of 3x3 identity matrix
    :return: identity matrix, shape is (B, 3, 3)
    """
    identity = torch.eye(3)
    identity = identity.repeat(num_ident, 1, 1)
    return identity


def gen_random_rotation(opt, num_pert: int):
    rad = float(opt.rotation_range) / 180.0 * np.pi
    if opt.pert_distribution == 'normal':
        theta = np.clip(np.random.normal(size=(num_pert,)) * rad, -2.0 * rad, 2.0 * rad)
    elif opt.pert_distribution == 'uniform':
        theta = np.random.uniform(low=-1, high=1, size=(num_pert,)) * rad
    else:
        raise NotImplementedError('unknown sampling distribution')
    return theta


def gen_random_translation(opt, num_pert):
    if opt.pert_distribution == 'normal':
        dx = np.clip(np.random.normal(size=(num_pert,)) * opt.translation_range,
                     -2.0 * opt.translation_range,
                     2.0 * opt.translation_range)
    elif opt.pert_distribution == 'uniform':
        dx = np.random.uniform(low=-1, high=1, size=(num_pert,)) * opt.translation_range
    else:
        raise NotImplementedError('unknown sampling distribution')
    return dx


def gen_random_scaling(opt, num_pert):
    if opt.pert_distribution == 'normal':
        sx = np.clip(np.random.normal(size=(num_pert,)) * opt.scaling_range,
                     -2.0 * opt.scaling_range,
                     2.0 * opt.scaling_range)
    elif opt.pert_distribution == 'uniform':
        sx = np.random.uniform(low=-1, high=1, size=(num_pert,)) * opt.scaling_range
    else:
        raise NotImplementedError('unknown sampling distribution')
    return sx


def gen_pert_for_translation(opt, num_pert):
    dx = gen_random_translation(opt, num_pert)
    dy = gen_random_translation(opt, num_pert)
    # make it a torch vector
    perturbation_vec = utils.to_torch(np.stack([dx, dy], axis=-1).astype(np.float32))
    return perturbation_vec


def gen_pert_for_trans_rot(opt, num_pert):
    theta = gen_random_rotation(opt, num_pert)
    dx = gen_random_translation(opt, num_pert)
    dy = gen_random_translation(opt, num_pert)
    # make it a torch vector
    perturbation_vec = utils.to_torch(np.stack([theta, dx, dy], axis=-1).astype(np.float32))
    return perturbation_vec


def gen_pert_for_similarity(opt, num_pert):
    theta = gen_random_rotation(opt, num_pert)
    s = gen_random_scaling(opt, num_pert)
    dx = gen_random_translation(opt, num_pert)
    dy = gen_random_translation(opt, num_pert)
    # make it a torch vector
    perturbation_vec = utils.to_torch(np.stack([theta, s, dx, dy], axis=-1).astype(np.float32))
    return perturbation_vec


def vec2mat(opt, vec):
    """covert a transformation vector to transformation matrix,

    Arguments:

        vec -- [transformation vector: , shape: (B, n)], where n is the number of warping parameters
    Returns:
        mat -- [transformation matrix, shape: (B, 3, 3)]
    """
    assert isinstance(vec, torch.Tensor), 'cannot process data type: {0}'.format(type(vec))
    if len(vec.shape) == 1:
        vec = vec[None]
    assert len(vec.shape) == 2
    if opt.warp_type == 'translation':
        transformation_mat = vec2mat_for_translation(vec)
    elif opt.warp_type == 'trans+rot':
        transformation_mat = vec2mat_for_trans_rot(vec)
    elif opt.warp_type == 'similarity':
        transformation_mat = vec2mat_for_similarity(vec)
    else:
        raise NotImplementedError('unknown warping method')
    return transformation_mat


def vec2mat_for_translation(vec):
    assert vec.shape[1] == 2
    _len = vec.shape[0]
    O = torch.zeros([_len], dtype=torch.float32, device=vec.device)
    I = torch.ones([_len], dtype=torch.float32, device=vec.device)

    dx, dy = torch.unbind(vec, dim=1)
    transformation_mat = torch.stack([torch.stack([I, O, dx], dim=-1),
                                      torch.stack([O, I, dy], dim=-1),
                                      torch.stack([O, O, I], dim=-1)], dim=1)
    return transformation_mat


def vec2mat_for_trans_rot(vec):
    assert vec.shape[1] == 3
    _len = vec.shape[0]
    O = torch.zeros([_len], dtype=torch.float32, device=vec.device)
    I = torch.ones([_len], dtype=torch.float32, device=vec.device)

    p1, p2, p3 = torch.unbind(vec, dim=1)
    theta = p1
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    dx = p2
    dy = p3
    R = torch.stack([torch.stack([cos, -sin, O], dim=-1),
                     torch.stack([sin, cos, O], dim=-1),
                     torch.stack([O, O, I], dim=-1)], dim=1)
    S = torch.stack([torch.stack([I, O, O], dim=-1),
                     torch.stack([O, I, O], dim=-1),
                     torch.stack([O, O, I], dim=-1)], dim=1)
    T = torch.stack([torch.stack([I, O, dx], dim=-1),
                     torch.stack([O, I, dy], dim=-1),
                     torch.stack([O, O, I], dim=-1)], dim=1)
    transformation_mat = torch.bmm(R, torch.bmm(S, T))

    return transformation_mat


def vec2mat_for_similarity(vec):
    assert vec.shape[1] == 4
    _len = vec.shape[0]
    O = torch.zeros([_len], dtype=torch.float32, device=vec.device)
    I = torch.ones([_len], dtype=torch.float32, device=vec.device)

    p1, p2, p3, p4 = torch.unbind(vec, dim=1)
    theta = p1
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    s = 2.0 ** (p2)
    dx = p3
    dy = p4
    R = torch.stack([torch.stack([cos, -sin, O], dim=-1),
                     torch.stack([sin, cos, O], dim=-1),
                     torch.stack([O, O, I], dim=-1)], dim=1)
    S = torch.stack([torch.stack([s, O, O], dim=-1),
                     torch.stack([O, s, O], dim=-1),
                     torch.stack([O, O, I], dim=-1)], dim=1)
    T = torch.stack([torch.stack([I, O, dx], dim=-1),
                     torch.stack([O, I, dy], dim=-1),
                     torch.stack([O, O, I], dim=-1)], dim=1)
    transformation_mat = torch.bmm(R, torch.bmm(S, T))

    return transformation_mat
