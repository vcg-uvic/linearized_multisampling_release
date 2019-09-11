import torch
from warp import linearized


class DifferentiableImageSampler():
    '''
    a differentiable image sampler which works with homography
    '''

    def __init__(self, sampling_mode, padding_mode):
        self.sampling_mode = sampling_mode
        self.padding_mode = padding_mode
        self.flatten_xy_cache = {}

    def warp_image(self, image, homography, out_shape=None):
        assert isinstance(image, torch.Tensor), 'cannot process data type: {0}'.format(type(image))
        assert isinstance(homography, torch.Tensor), 'cannot process data type: {0}'.format(type(homography))

        if out_shape is None:
            out_shape = image.shape[-2:]
        if len(image.shape) < 4:
            image = image[None]
        if len(homography.shape) < 3:
            homography = homography[None]
        assert image.shape[0] == homography.shape[0], 'batch size of images do not match the batch size of homographies'
        # create grid for interpolation (in frame coordinates)
        x, y = self.create_flatten_xy(x_steps=out_shape[-1], y_steps=out_shape[-2], device=homography.device)
        grid = self.flatten_xy_to_warped_grid(x, y, homography, out_shape)
        # sample warped image
        warped_img = linearized.grid_sample(image, grid, mode=self.sampling_mode, padding_mode=self.padding_mode)

        if linearized.has_nan(warped_img):
            print('nan value in warped image! set to zeros')
            warped_img[linearized.is_nan(warped_img)] = 0

        return warped_img

    def create_flatten_xy(self, x_steps: int, y_steps: int, device):
        if (x_steps, y_steps) in self.flatten_xy_cache:
            x, y = self.flatten_xy_cache[(x_steps, y_steps)]
            return x.clone(), y.clone()
        y, x = torch.meshgrid([
            torch.linspace(-1.0, 1.0, steps=y_steps, device=device),
            torch.linspace(-1.0, 1.0, steps=x_steps, device=device)
        ])
        x, y = x.flatten(), y.flatten()
        self.flatten_xy_cache[(x_steps, y_steps)] = x, y
        return x.clone(), y.clone()

    def flatten_xy_to_warped_grid(self, x, y, homography, out_shape):
        batch_size = homography.shape[0]
        # append ones for homogeneous coordinates
        xy = torch.stack([x, y, torch.ones_like(x)])
        xy = xy.repeat([batch_size, 1, 1])  # shape: (B, 3, N)

        xy_warped = torch.matmul(homography, xy)
        xy_warped, z_warped = xy_warped.split(2, dim=1)
        xy_warped = xy_warped / (z_warped + 1e-8)
        x_warped, y_warped = torch.unbind(xy_warped, dim=1)
        # build grid
        grid = torch.stack([
            x_warped.view(batch_size, *out_shape[-2:]),
            y_warped.view(batch_size, *out_shape[-2:])
        ], dim=-1)
        return grid
