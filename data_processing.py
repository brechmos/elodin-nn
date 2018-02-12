import uuid
import glob
import time
import numpy as np

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("DataProcessing")
log.setLevel(logging.INFO)


class DataProcessing:
    def __init__(self):
        pass

import scipy.ndimage
class ZoomData(DataProcessing):

    def __init__(self, zoom_level=1):
        self._zoom_level = zoom_level

    def __repr__(self):
        return 'ZoomData (level {})'.format(self._zoom_level)

    def process(self, input_data):

        if len(input_data.shape) == 2:
            return scipy.ndimage.zoom(input_data, self._zoom_level)
        else:
            output_data = None
            for ii in range(input_data.shape[2]):
                od = scipy.ndimage.zoom(input_data[:, :, ii], self._zoom_level)
                if output_data is None:
                    output_data = np.zeros((od.shape[0], od.shape[1], 3))
                output_data[:, :, ii] = od
            return output_data

import scipy.ndimage.filters
class MedianFilterData(DataProcessing):
    def __init__(self, kernel_size):
        self._kernel_size = kernel_size

    def __repr__(self):
        return 'MedianFilterData (kernel size {})'.format(self._kernel_size)

    def process(self, input_data):
        return scipy.ndimage.filters.median_filter(input_data, size=self._kernel_size)
