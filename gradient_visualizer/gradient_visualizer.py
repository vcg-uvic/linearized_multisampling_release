import colorsys

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import utils
from warp import perturbation_helper, sampling_helper


class GradientVisualizer(object):
    def __init__(self, opt):
        self.opt = opt
        self.warper = sampling_helper.DifferentiableImageSampler('bilinear', 'zeros')

    def build_criterion(self):
        if self.opt.optim_criterion == 'l1loss':
            criterion = torch.nn.L1Loss()
        elif self.opt.optim_criterion == 'mse':
            criterion = torch.nn.MSELoss()
        else:
            raise ValueError('unknown optimization criterion: {0}'.format(self.opt.optim_criterion))
        return criterion

    def build_gd_optimizer(self, params):
        optim_list = [{"params": params, "lr": self.opt.optim_lr}]
        optimizer = torch.optim.SGD(optim_list)
        return optimizer

    def create_translation_grid(self, resolution=None):
        if resolution is None:
            resolution = self.opt.grid_size
        results = []
        x_steps = torch.linspace(-1, 1, steps=resolution)
        y_steps = torch.linspace(-1, 1, steps=resolution)
        for x in x_steps:
            for y in y_steps:
                translation_vec = torch.stack([x, y], dim=0)[None]
                results.append(translation_vec)
        return results

    def get_next_translation_vec(self, data_pack, image_warper):
        translation_vec = data_pack['translation_vec'].clone().detach().requires_grad_(True)
        translation_mat = perturbation_helper.vec2mat_for_translation(translation_vec)
        orig_image = data_pack['original_image']
        criterion = self.build_criterion()
        optimizer = self.build_gd_optimizer(params=translation_vec)
        ident_mat = perturbation_helper.gen_identity_mat(1)
        down_sampled_orig_image = self.warper.warp_image(orig_image, ident_mat, self.opt.out_shape).detach()
        warped_image = image_warper.warp_image(orig_image, translation_mat, self.opt.out_shape)
        loss = criterion(warped_image, down_sampled_orig_image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return translation_vec

    def get_gradient_over_translation_vec(self, data_pack, image_warper):
        translation_vec = data_pack['translation_vec']
        next_translation_vec = self.get_next_translation_vec(data_pack, image_warper)
        moving_dir = next_translation_vec - translation_vec
        return moving_dir

    def get_gradient_grid(self, orig_image, image_warper):
        gradient_grid = []
        translation_grid = self.create_translation_grid()
        for translation_vec in translation_grid:
            data_pack = {}
            data_pack['translation_vec'] = translation_vec
            data_pack['original_image'] = orig_image
            cur_gradient = self.get_gradient_over_translation_vec(data_pack, image_warper)
            gradient_pack = {'translation_vec': translation_vec, 'gradient': cur_gradient}
            gradient_grid.append(gradient_pack)
        return gradient_grid

    def draw_gradient_grid(self, orig_image, image_warper):
        gradient_grid = self.get_gradient_grid(orig_image, image_warper)

        _, ax = plt.subplots()
        ax.axis('equal')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        orig_image_show = utils.torch_img_to_np_img(orig_image)[0]
        ax.imshow(orig_image_show, extent=[-1, 1, -1, 1])

        for gradient in gradient_grid:
            orig_point = np.zeros([2], dtype=np.float32)
            base_loc = 0 - (gradient['translation_vec'])[0].data.cpu().numpy()
            gradient_dir = (gradient['gradient'])[0].data.cpu().numpy()
            gradient_dir = 0 - utils.unit_vector(gradient_dir)
            gt_dir = orig_point - base_loc
            gt_dir = utils.unit_vector(gt_dir)

            angle = utils.angle_between(gradient_dir, gt_dir)
            try:
                cur_color = self.angle_to_color(angle)
            except ValueError:
                cur_color = [0., 0., 0.]
            gradient_dir = gradient_dir / 10
            ax.arrow(base_loc[0], base_loc[1], gradient_dir[0], gradient_dir[1], head_width=0.05, head_length=0.1, color=cur_color)
        plt.show()

    def angle_to_color(self, angle):
        red_hue, _, _ = colorsys.rgb_to_hsv(1, 0, 0)
        green_hue, _, _ = colorsys.rgb_to_hsv(0, 1, 0)
        cur_hue = np.interp(angle, (0, np.pi), (green_hue, red_hue))
        cur_color = colorsys.hsv_to_rgb(cur_hue, 1, 1)
        return cur_color
