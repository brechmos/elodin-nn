import numpy as np


def gray2rgb(data):
    """
    Convert 2D data set to 3D gray scale

    :param data:
    :return:
    """
    data_out = np.zeros((data.shape[0], data.shape[1], 3))
    data_out[:, :, 0] = data
    data_out[:, :, 1] = data
    data_out[:, :, 2] = data

    return data_out
