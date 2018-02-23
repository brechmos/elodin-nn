import uuid
import itertools
import numpy as np

from skimage import measure
from skimage import filters
import skimage.transform

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("Cutouts")
log.setLevel(logging.INFO)


class Cutouts:
    def __init__(self):
        # list of data processing elements
        self._uuid = str(uuid.uuid4())

        # number of pixels around the image to include in the display
        self._image_display_margin = 50

    def create_cutouts(self, data):
        raise NotImplemented('Must create cutouts function')

    def save(self):
        raise NotImplemented('Must create cutouts function')

    @staticmethod
    def load(parameters):
        for class_ in Cutouts.__subclasses__():
            if class_.__name__ == parameters['cutout_type']:

                # If the class name matches, instantiate, pass in the parameters
                # and return the newly created class.
                tt = class_()
                tt.load(parameters)
                return tt


class BasicCutouts:
    def __init__(self, output_size, step_size):

        # list of data processing elements
        self._output_size = output_size
        self._step_size = step_size
        self._uuid = str(uuid.uuid4())

    def create_cutouts(self, data):
        """
        Calculate the fingerprints for each subsection of the image in each file.

        :param data:
        :return:
        """

        N = self._output_size // 2

        # Determine the centers to use for the fingerprint calculation
        nrows, ncols = data.shape[:2]
        rows = range(N, nrows-N, self._step_size)
        cols = range(N, ncols-N, self._step_size)

        for ii, (row, col) in enumerate(itertools.product(rows, cols)):
            td = data[row-N:row+N, col-N:col+N]
            yield row-N, row+N, col-N, col+N, td

    def save(self):
        return {
            'cutout_type': self.__class__.__name__,
            'output_size': self._output_size,
            'step_size': self._step_size,
            'uuid': self._uuid
        }

    def _load(self, parameters):
        self._step_size = parameters['step_size']
        self._output_size = parameters['output_size']
        self._uuid = parameters['uuid']


class BlobCutouts:
    def __init__(self, output_size, mean_threshold=2.0, gaussian_smoothing_sigma=10, label_padding=80):

        # list of data processing elements
        self._output_size = output_size
        self._mean_threshold = mean_threshold
        self._gaussian_smoothing_sigma = gaussian_smoothing_sigma
        self._label_padding = label_padding
        self._uuid = str(uuid.uuid4())

    def create_cutouts(self, data):
        """

        :param data: Input data
        :param mean_threshold: Threshold applied to mean in order to clip background
        :param gaussian_smoothing_sigma: How much smoothing should be applied to the image
        :param label_padding: Padding around the labeled cutout.
        :return:
        """

        # Find the blobs
        gray_image = np.dot(data[:, :, :3], [0.299, 0.587, 0.114])
        im = filters.gaussian_filter(gray_image, sigma=self._gaussian_smoothing_sigma)
        blobs = im > self._mean_threshold * im.mean()
        blobs_labels = measure.label(blobs, background=0)

        # Loop over the blobs
        for ii in range(1, blobs_labels.max()):
            b = blobs_labels == ii

            # Find the extent of the image
            rows, cols = np.nonzero(b)
            min_row, max_row, mean_row = int(rows.min()), int(rows.max()), int(rows.mean())
            min_col, max_col, mean_col = int(cols.min()), int(cols.max()), int(cols.mean())

            log.debug('label mins and maxs {} {}   {} {}'.format(min_row, max_row, min_col, max_col))

            # Pad the blob a bit
            min_row = max(min_row - self._label_padding, 0)
            max_row = min(max_row + self._label_padding, data.shape[0])
            min_col = max(min_col - self._label_padding, 0)
            max_col = min(max_col + self._label_padding, data.shape[1])

            log.debug('label mins and maxs with padding {} {}   {} {}'.format(min_row, max_row, min_col, max_col))

            # Correct so it is square
            rr = max_row - min_row
            cc = max_col - min_col
            if rr > cc:
                min_col = min_col - (rr - cc) // 2
                max_col = max_col - (rr - cc) // 2
            else:
                min_row = min_row - (cc - rr) // 2
                max_row = max_row - (cc - rr) // 2

            # Skip for now
            if min_row < 0 or min_col < 0 or max_row >= data.shape[0] or max_col >= data.shape[1]:
                continue

            log.debug('Cutting out {} {}   {} {}'.format(min_row, max_row, min_col, max_col))

            # Resize the image and return
            td = skimage.transform.resize(data[min_row:max_row, min_col:max_col],
                                          [self._output_size, self._output_size])

            yield min_row, max_row, min_col, max_col, td

    def save(self):
        return {
            'cutout_type': self.__class__.__name__,
            'output_size': self._output_size,
            'mean_threshold': self._mean_threshold,
            'gaussian_smoothing_sigma': self._gaussian_smoothing_sigma,
            'label_padding': self._label_padding,
            'uuid': self._uuid
        }

    def _load(self, parameters):
        self._output_size = parameters['output_size']
        self._gaussian_smoothing_sigma = parameters['gaussian_smoothing_sigma']
        self._label_padding = parameters['label_padding']
        self._mean_threshold = parameters['mean_threshold']
        self._uuid = parameters['uuid']