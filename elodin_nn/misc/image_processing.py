import numpy as np
import scipy.ndimage.interpolation
from skimage import exposure
from skimage.transform import resize as skimage_resize
from ..tl_logging import get_logger

from ..misc import image_processing

import logging
log = get_logger('image processing', level=logging.INFO)


class ImageProcessing(object):

    _processing_name = ''

    @staticmethod
    def load(parameters):
        """
        Load an ImageProcessing instance based on the parameters.
        The parameters must come from the save() method based on one of the 
        subclasses defined below.

        Parameters
        -----------
        parameters : dict
            The parameter dictionary that defines the type and parameters
            of an ImageProcessing instance..

        Raises
        ------
        TypeError
            In case ``data`` or ``meta`` don't meet the restrictions.

        """

        #
        # Find the correct sub class based on the processing_type
        # in the parameter dict.
        #

        for cls in ImageProcessing.__subclasses__():
            if cls._processing_name == parameters['processing_type']:

                # Create an instance of the ImageProcessing subclass with parameters passed
                ip = cls(**parameters['parameters'])

                # Return the ImageProcessing instance.
                return ip

    def __str__(self):
        return self._processing_name

    def save(self):
        """
        Save the instance information of the subclass as a dict
        along with parameters.

        Raises
        ------
        TypeError
            In case ``data`` or ``meta`` don't meet the restrictions.
        """

        # self._processing_name comes in as a class variable of the subclass
        # self._parameters is defined in the __init__ of the subclass
        return {
            'processing_type': self._processing_name,
            'parameters': self._parameters
        }


class GrayScale(ImageProcessing):
    """
    Convert the image to gray scale (3D -> 2D).
    """
    _processing_name = 'gray_scale'

    def __init__(self):
        """
        No parameters so set internal to empty.
        """
        self._parameters = {}

    def process(self, numpy_data):
        """
        Convert to grayscale.
        """
        log.debug('grayscale the data')
        if len(numpy_data.shape) == 3:
            return np.dot(numpy_data[:, :, :3], [0.299, 0.587, 0.114])
        else:
            return numpy_data


class Resize(ImageProcessing):
    """
    Resize the image to a specified size.
    """
    _processing_name = 'resize'

    def __init__(self, output_size):
        """
        :param output_size: a list or tuple of size 2 describing the output size.
        """
        if not isinstance(output_size, (list, tuple)) and not len(output_size) == 2:
            raise Exception('Cutout processing resize output_size parmaeter, {}, is wrong'.format(output_size))

        self._parameters = {
            'output_size': output_size
        }

    def process(self, numpy_data):
        output_size = self._parameters['output_size']
        log.debug('resizing data to {}'.format(output_size))
        return skimage_resize(numpy_data, output_size)


class Crop(ImageProcessing):
    """
    Crop the image given the row_start, row_end, col_start, col_end.
    """
    _processing_name = 'crop'

    def __init__(self, start_end):
        """
        :param output_size: a list or tuple of size 4 describing the start and stop rows and cols
        """
        if not isinstance(start_end, (list, tuple)) and not len(start_end) == 4:
            raise Exception('Data processing resize output_size parmaeter, {}, is wrong'.format(start_end))

        self._parameters = {
            'start_end': start_end
        }

    def process(self, numpy_data):
        start_end = self._parameters['start_end']
        log.debug('cropping data by {}'.format(start_end))
        rc = start_end
        return numpy_data[rc[0]:rc[1], rc[2]:rc[3]]


class Rotate(ImageProcessing):
    """
    Image rotation.

    Note that anything away from 0, 90, 180, 270 is going to
    have black regions around the image.
    """
    _processing_name = 'rotate'

    def __init__(self, angle=0.0):
        """
        Initializer for an image rotation.

        """

        self._parameters = {
            'angle': angle
        }

    def process(self, numpy_data):
        """
        Process the input numpy data to rescale the intenisity.

        Parameters
        ----------
        numpy_data : numpy array
            The image of data to process.

        Return
        ------
        processed_data : numpy array
            processed array of data, same size as input

        """
        log.info('rotate')
        return scipy.ndimage.interpolation.rotate(numpy_data, self._angle)


class FlipLR(ImageProcessing):
    """
    Flip the image LR.
    """
    _processing_name = 'fliplr'

    def __init__(self):
        """
        No parameters needed, so set empty.
        """
        self._parameters = {}

    def process(self, numpy_data):
        """
        Process the input numpy data to rescale the intenisity.

        Parameters
        ----------
        numpy_data : numpy array
            The image of data to process.

        Return
        ------
        processed_data : numpy array
            processed array of data, same size as input

        """
        log.info('fliplr')
        return numpy_data[:, ::-1]


class FlipUD(ImageProcessing):

    _processing_name = 'flipud'

    def __init__(self):
        """
        Initializer for an image Flip up/down.

        """
        self._parameters = {}

    def process(self, numpy_data):
        """
        Process the input numpy data to rescale the intenisity.

        Parameters
        ----------
        numpy_data : numpy array
            The image of data to process.

        Return
        ------
        processed_data : numpy array
            Numpy array flipped up/down.

        """
        log.info('flipud')
        return numpy_data[::-1]


class RescaleIntensity(ImageProcessing):

    _processing_name = 'rescale_intensity'

    def __init__(self, lower_percentile=2, upper_percentile=98):
        """
        Initializer

        Parameters
        ----------
        lower_percentile : number
            Lower percentile value passed in to scikit.expousre,rescale_intenisty
        upper_percentile : number
            Upper percentile value passed in to scikit.expousre,rescale_intenisty

        """
        if (not isinstance(lower_percentile, (int, float)) or
            not isinstance(upper_percentile, (int, float))):
            raise ValueError('Lower and upper percentiles must be a number')

        self._parameters = {
            'lower_percentile': lower_percentile,
            'upper_percentile': upper_percentile
        }

    def process(self, numpy_data):
        """
        Process the input numpy data to rescale the intenisity.

        Parameters
        ----------
        numpy_data : numpy array
            The image of data to process.

        Return
        ------
        processed_data : numpy array
            processed array of data, same size as input

        """
        lp = self._parameters['lower_percentile']
        up = self._parameters['upper_percentile']
        log.info('rescale intensity to percentile range {} {}'.format(lp, up))
        return exposure.rescale_intensity(numpy_data, in_range=(lp, up))


class HistogramEqualization(ImageProcessing):

    _processing_name = 'histogram_equalization'

    def __init__(self):
        self._parameters = {}

    def process(self, numpy_data):
        """
        Process the input numpy data using histogram equalization.

        Parameters
        ----------
        numpy_data : numpy array
            The image of data to process.

        Return
        ------
        processed_data : numpy array
            processed array of data, same size as input

        """
        log.info('histogram equalization')
        return exposure.equalize_hist(numpy_data)


class AdaptiveHistogramEqualization(ImageProcessing):

    _processing_name = 'adapative_histogram_equalization'

    def __init__(self, clip_limit=0.03):
        """
        Initializer

        Parameters
        ----------
        clip_limit : number
            Clip limit will be passed to the scikit-image method.

        """
        self._parameters = {
            'clip_limit': clip_limit
        }

    def process(self, numpy_data):
        """
        Process the input numpy data using adaptive histogram equalization.

        Parameters
        ----------
        numpy_data : numpy array
            The image of data to process.

        Return
        ------
        processed_data : numpy array
            processed array of data, same size as input

        """
        log.info('adaptive histogram equalization with clip_limit {}'.format(**self._parameters))
        return exposure.equalize_adapthist(numpy_data, **self._parameters)
