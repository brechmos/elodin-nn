import os
import numpy as np

import imageio

from transfer_learning.fingerprint import FingerprintResnet

def load_jpg(filename):
    return np.array(imageio.imread(filename))

def test_load_data():
    data = load_jpg('data/j8za09050_drz_small.jpg')
    cmp = np.array([[255, 252, 246, 255], [255, 241, 255, 241], [255, 255, 246, 13], [248, 255, 255, 0]])
    assert np.allclose(data[10:14, 10:14], cmp, atol=1)
