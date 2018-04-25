import os
import numpy as np

import imageio

from transfer_learning.fingerprint import FingerprintResnet

def load_jpg(filename):
    return np.array(imageio.imread(filename))

def test_calculate_fingerprint():

    data = load_jpg('data/j8za09050_drz_small.jpg')
    fresnet = FingerprintResnet()

    predictions = fresnet.calculate(data[:224,:224])

    assert predictions[0][:2] == ('n03223299', 'doormat')
    assert np.allclose(predictions[0][2], 0.40468791, atol=0.001)

    assert predictions[1][:2] == ('n04589890', 'window_screen')
    assert np.allclose(predictions[1][2], 0.14722134, atol=0.001)

    assert predictions[2][:2] == ('n03857828', 'oscilloscope')
    assert np.allclose(predictions[2][2], 0.051630393, atol=0.001)
