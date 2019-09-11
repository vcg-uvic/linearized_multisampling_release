'''
Linearized multi-sampling core part.
All methods are encapsuled in class LinearizedMutilSampler.
Hyperparameters are stored as static variables.
Main sampling method entrance is linearized_grid_sample.
'''

import torch


######### Utils to minimize dependencies #########
# Move utils to another file if you want
def print_notification(content_list, notification_type='NOTIFICATION'):
    print('---------------------- {0} ----------------------'.format(notification_type))
    print()
    for content in content_list:
        print(content)
    print()
    print('----------------------------------------------------')


def is_nan(x):
    '''
    get mask of nan values.
    :param x: torch or numpy var.
    :return: a N-D array of bool. True -> nan, False -> ok.
    '''
    return x != x


def has_nan(x) -> bool:
    '''
    check whether x contains nan.
    :param x: torch or numpy var.
    :return: single bool, True -> x containing nan, False -> ok.
    '''
    return is_nan(x).any()


def mat_3x3_inv(mat):
    '''
    calculate the inverse of a 3x3 matrix, support batch.
    :param mat: torch.Tensor -- [input matrix, shape: (B, 3, 3)]
    :return: mat_inv: torch.Tensor -- [inversed matrix shape: (B, 3, 3)]
    '''
    if len(mat.shape) < 3:
        mat = mat[None]
    assert mat.shape[1:] == (3, 3)

    # Divide the matrix with it's maximum element
    max_vals = mat.max(1)[0].max(1)[0].view((-1, 1, 1))
    mat = mat / max_vals

    det = mat_3x3_det(mat)
    inv_det = 1.0 / det

    mat_inv = torch.zeros(mat.shape, device=mat.device)
    mat_inv[:, 0, 0] = (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 0, 1] = (mat[:, 0, 2] * mat[:, 2, 1] - mat[:, 0, 1] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 0, 2] = (mat[:, 0, 1] * mat[:, 1, 2] - mat[:, 0, 2] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 1, 0] = (mat[:, 1, 2] * mat[:, 2, 0] - mat[:, 1, 0] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 1, 1] = (mat[:, 0, 0] * mat[:, 2, 2] - mat[:, 0, 2] * mat[:, 2, 0]) * inv_det
    mat_inv[:, 1, 2] = (mat[:, 1, 0] * mat[:, 0, 2] - mat[:, 0, 0] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 2, 0] = (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 2, 0] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 2, 1] = (mat[:, 2, 0] * mat[:, 0, 1] - mat[:, 0, 0] * mat[:, 2, 1]) * inv_det
    mat_inv[:, 2, 2] = (mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 1, 0] * mat[:, 0, 1]) * inv_det

    # Divide the maximum value once more
    mat_inv = mat_inv / max_vals
    return mat_inv


def mat_3x3_det(mat):
    '''
    calculate the determinant of a 3x3 matrix, support batch.
    '''
    if len(mat.shape) < 3:
        mat = mat[None]
    assert mat.shape[1:] == (3, 3)

    det = mat[:, 0, 0] * (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) \
        - mat[:, 0, 1] * (mat[:, 1, 0] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 0]) \
        + mat[:, 0, 2] * (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 1, 1] * mat[:, 2, 0])
    return det


######### Linearized multi-sampling #########
def grid_sample(input, grid, mode='bilinear', padding_mode='zeros'):
    '''
    original function prototype:
    torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')
    copy from pytorch 1.2.0 source code
    '''
    if mode == 'linearized':
        return LinearizedMutilSampler.linearized_grid_sample(input, grid, padding_mode)
    else:
        return torch.nn.functional.grid_sample(input, grid, mode, padding_mode)


class LinearizedMutilSampler():

    num_grid = 8
    NUM_XY = 2
    noise_strength = 0.5
    need_push_away = True
    fixed_bias = False
    is_hyperparameters_set = False

    @classmethod
    def set_hyperparameters(cls, opt):
        cls.num_grid = opt.num_grid
        cls.noise_strength = opt.noise_strength
        cls.need_push_away = opt.need_push_away
        cls.fixed_bias = opt.fixed_bias
        if cls.is_hyperparameters_set:
            raise RuntimeError('Trying to reset the hyperparamter for linearized multi sampler, currently not allowed')
        else:
            cls.is_hyperparameters_set = True
        notification = []
        notification.append('Hyperparameters are set')
        notification.append('num_grid: {0}'.format(cls.num_grid))
        notification.append('noise_strength: {0}'.format(cls.noise_strength))
        notification.append('need_push_away: {0}'.format(cls.need_push_away))
        notification.append('fixed_bias: {0}'.format(cls.fixed_bias))
        print_notification(notification)

    @classmethod
    def linearized_grid_sample(cls, input, grid, padding_mode):
        # assert cls.is_hyperparameters_set, 'linearized sampler hyperparameters are not set'
        assert isinstance(input, torch.Tensor), 'cannot process data type: {0}'.format(type(input))
        assert isinstance(grid, torch.Tensor), 'cannot process data type: {0}'.format(type(grid))

        batch_size, source_channels, source_height, source_width = input.shape
        least_offset = torch.tensor([2.0 / source_width, 2.0 / source_height], device=grid.device)
        auxiliary_grid = cls.create_auxiliary_grid(grid, least_offset)
        warped_input = cls.warp_input_with_auxiliary_grid(input, auxiliary_grid, padding_mode)
        out = cls.linearized_fitting(warped_input, auxiliary_grid)
        return out

    @classmethod
    def linearized_fitting(cls, input, grid):
        def defensive_assert(input, grid):
            assert len(input.shape) == 5, 'shape should be: B x Grid x C x H x W'
            assert len(grid.shape) == 5, 'shape should be: B x Grid x H x W x XY'
            assert input.shape[0] == grid.shape[0]
            assert input.shape[1] == grid.shape[1]
            assert input.shape[1] > 1, 'num of grid should be larger than 1'

        def get_center_and_other(input, grid):
            center_image = input[:, 0:1]
            center_grid = grid[:, 0:1]
            other_image = input[:, 1:]
            otehr_grid = grid[:, 1:]
            result = {'center_image': center_image,
                      'center_grid': center_grid,
                      'other_image': other_image,
                      'other_grid': otehr_grid,
                      }
            return result

        defensive_assert(input, grid)
        extracted_dict = get_center_and_other(input, grid)
        delta_vals = cls.get_delta_vals(extracted_dict)
        center_image = extracted_dict['center_image']
        center_grid = extracted_dict['center_grid']
        delta_intensity = delta_vals['delta_intensity']
        delta_grid = delta_vals['delta_grid']
        # reshape to [B, H, W, Grid-1, XY1]
        delta_grid = delta_grid.permute(0, 2, 3, 1, 4)
        # reshape to [B, H, W, Grid-1, C]
        delta_intensity = delta_intensity.permute(0, 3, 4, 1, 2)
        # calculate dI/dX, euqation(7) in paper
        xTx = torch.matmul(torch.transpose(delta_grid, 3, 4), delta_grid)

        # take inverse
        xTx_inv = mat_3x3_inv(xTx.view(-1, 3, 3))
        xTx_inv = xTx_inv.view(xTx.shape)
        xTx_inv_xT = torch.matmul(xTx_inv, torch.transpose(delta_grid, 3, 4))
        # gradient_intensity shape: [B, H, W, XY1, C]
        gradient_intensity = torch.matmul(xTx_inv_xT, delta_intensity)

        if has_nan(gradient_intensity):
            print('nan val in gradient_intensity')
            nan_idx = is_nan(gradient_intensity)
            gradient_intensity[nan_idx] = torch.zeros(gradient_intensity[nan_idx].shape,
                                                      device=gradient_intensity.device).detach()

        # stop gradient
        gradient_intensity_stop = gradient_intensity.detach()
        center_grid_stop = center_grid.detach()

        # center_grid shape: [B, H, W, XY1, 1]
        center_grid_xyz = torch.cat([center_grid, torch.ones(center_grid[..., 0:1].shape, device=center_grid.device)],
                                    dim=4).permute(0, 2, 3, 4, 1)
        if cls.fixed_bias:
            center_grid_xyz_stop = torch.cat([center_grid_stop, torch.ones(center_grid_stop[..., 0:1].shape, device=center_grid_stop.device)],
                                             dim=4).permute(0, 2, 3, 4, 1)
        else:
            center_grid_xyz_stop = torch.cat([center_grid_stop, torch.zeros(center_grid_stop[..., 0:1].shape, device=center_grid_stop.device)],
                                             dim=4).permute(0, 2, 3, 4, 1)

        # map to linearized, equation(2) in paper
        image_linearized = torch.matmul(gradient_intensity_stop.transpose(3, 4), (center_grid_xyz - center_grid_xyz_stop))[..., 0].permute(0, 3, 1, 2) + center_image[:, 0]
        return image_linearized

    @staticmethod
    def get_delta_vals(data_dict):
        def defensive_assert(center_image, other_image):
            assert len(center_image.shape) == 5, 'shape should be: B x Grid x C x H x W'
            assert len(other_image.shape) == 5, 'shape should be: B x Grid x C x H x W'
            assert center_image.shape[0] == other_image.shape[0]
            assert center_image.shape[1] == 1, 'num of center_image per single sample should be 1'
            assert other_image.shape[1] >= 1, ('num of other_image per single sample should be larger'
                                               ' or equal than 1, got shape {0} for {1}'.format(other_image.shape,
                                                                                                'other_image'))

        center_image = data_dict['center_image']
        center_grid = data_dict['center_grid']
        other_image = data_dict['other_image']
        other_grid = data_dict['other_grid']
        defensive_assert(center_image, other_image)

        batch_size = other_image.shape[0]
        num_other_image = other_image.shape[1]
        center_image_batch = center_image.repeat([1, num_other_image, 1, 1, 1])
        center_grid_batch = center_grid.repeat([1, num_other_image, 1, 1, 1])
        delta_intensity = other_image - center_image_batch
        delta_grid = other_grid - center_grid_batch
        delta_mask = (delta_grid[..., 0:1] >= -1.0) * (delta_grid[..., 0:1] <= 1.0) * (delta_grid[..., 1:2] >= -1.0) * (delta_grid[..., 1:2] <= 1.0)
        delta_mask = delta_mask.float()
        delta_grid = torch.cat([delta_grid, torch.ones(delta_grid[..., 0:1].shape, device=delta_grid.device)], dim=4)
        delta_grid *= delta_mask
        delta_vals = {'delta_intensity': delta_intensity,
                      'delta_grid': delta_grid,
                      }
        return delta_vals

    @staticmethod
    def warp_input_with_auxiliary_grid(input, grid, padding_mode):
        assert len(input.shape) == 4
        assert len(grid.shape) == 5
        assert input.shape[0] == grid.shape[0]

        batch_size, num_grid, height, width, num_xy = grid.shape
        grid = grid.reshape([-1, height, width, num_xy])
        grid = grid.detach()
        input = input.repeat_interleave(num_grid, 0)
        warped_input = torch.nn.functional.grid_sample(input, grid, mode='bilinear',
                                                       padding_mode=padding_mode)
        warped_input = warped_input.reshape(batch_size, num_grid, -1, height, width)
        return warped_input

    @classmethod
    def create_auxiliary_grid(cls, grid, least_offset):
        batch_size, height, width, num_xy = grid.shape
        grid = grid.repeat(1, cls.num_grid, 1, 1).reshape(batch_size, cls.num_grid, height, width, num_xy)
        grid = cls.add_noise_to_grid(grid, least_offset)
        return grid

    @classmethod
    def add_noise_to_grid(cls, grid, least_offset):
        grid_shape = grid.shape
        assert len(grid_shape) == 5
        batch_size, num_grid, height, width, num_xy = grid_shape
        assert num_xy == cls.NUM_XY
        assert num_grid == cls.num_grid

        grid_noise = torch.randn([batch_size, cls.num_grid - 1, height, width, num_xy], device=grid.device) / torch.tensor([[width, height]], dtype=torch.float32, device=grid.device) * cls.noise_strength
        grid[:, 1:] += grid_noise
        if cls.need_push_away:
            grid = cls.push_away_samples(grid, least_offset)
        return grid

    @classmethod
    def push_away_samples(cls, grid, least_offset):
        grid_shape = grid.shape
        assert len(grid_shape) == 5
        batch_size, num_grid, height, width, num_xy = grid_shape
        assert num_xy == cls.NUM_XY
        assert num_grid == cls.num_grid
        assert cls.need_push_away

        noise = torch.randn(grid[:, 1:].shape, device=grid.device)
        noise = noise * least_offset
        grid[:, 1:] = grid[:, 1:] + noise
        return grid
