import numpy as np
from bson import ObjectId


def convert_objectid(dct):
    return {k: v if not isinstance(v, ObjectId) else str(v) for k, v in dct.items()}


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


def rgb2plot(data):
    """
    Convert the input data to RGB. This is basically clipping and cropping the intensity range for display

    :param data:
    :return:
    """

    mindata, maxdata = np.percentile(data[np.isfinite(data)], (0.01, 99.0))
    return np.clip((data - mindata) / (maxdata - mindata) * 255, 0, 255).astype(np.uint8)
