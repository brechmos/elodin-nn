import numpy as np
import imageio
from astropy.io import fits

import logging
logging.basicConfig(format='%(levelname)s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("Data")
log.setLevel(logging.DEBUG)

class Data:
    def __init__(self, filename, *args, **kwargs):
        self._filename = filename
        self._input_data = None

        log.info('Filename is {}'.format(self._filename))

        self.load_data()

        log.info('Data is shape {}'.format(self._input_data.shape))

    def __repr__(self):
        return self._filename

    @property
    def shape(self):
        return self._input_data.shape

    @property
    def filename(self):
        return self._filename

    def data(self):
        return self._input_data

    def __getitem__(self, index):
        if self._input_data is None:
            self.load_data()

        return self._input_data[index]

    def load_data(self):
        if self._filename.endswith('tiff'):
            self._input_data = np.array(imageio.imread(self._filename))
        elif 'fits' in self._filename:
            self._input_data = fits.open(self._filename)[1].data


class BasicData(Data):
    def __init__(self, filename, *args, **kwargs):
        super(BasicData, self).__init__(filename, *args, **kwargs)


class MedianFilterData(Data):

    def __init__(self, filename, *args, **kwargs):
        super(MedianFilterData, self).__init__(filename)

        from scipy.ndimage.filters import median_filter
        log.info('Calculating the median filter on the input data')
        self._input_data = median_filter(self._input_data, (3, 3, 1))

    def __repr__(self):
        return '{}: Median filter 3x3'.format(self._filename)


class ArbitraryFilterData(Data):

    def __init__(self, filename, *args, **kwargs):
        super(MedianFilterData, self).__init__(filename, *args, **kwargs)

        from scipy.ndimage.filters import median_filter
        self._input_data = median_filter(self._input_data, (3, 3, 1))

    def __repr__(self):
        return '{}: Arbitrary filter'.format(self._filename)
