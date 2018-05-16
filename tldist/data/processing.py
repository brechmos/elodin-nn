import skimage
import numpy as np
from ..tl_logging import get_logger
import logging
log = get_logger('data processing')


class DataProcessing(object):

    @staticmethod
    def load(thedict):
        if thedict['data_processing_type'] == 'resize':
            return Resize(thedict)
        elif thedict['data_processing_type'] == 'crop':
            return Crop(thedict)
        elif thedict['data_processing_type'] == 'gray_scale':
            return GrayScale()


class Resize(DataProcessing):

    def __init__(self, output_size):
        """
        :param output_size: a list or tuple of size 2 describing the output size.
        """

        # Not beautiful code
        if isinstance(output_size, dict):
            output_size = output_size['parameters']['output_size']

        if not isinstance(output_size, (list, tuple)) and not len(output_size) == 2:
            raise Exception('Data processing resize output_size parmaeter, {}, is wrong'.format(output_size))

        self._output_size = output_size

    def save(self):
        return {
            'data_processing_type': 'resize',
            'parameters': {
                'output_size': self._output_size
            }
        }

    def load(self, thedict):

        if not thedict['data_processing_type'] == 'resize':
            raise Exception('wrong data processing type {} for resize')

        self._output_size = thedict['output_size']

    def process(self, data):
        log.info('resizeing data to {}'.format(self._output_size))

        for ii in range(self._output_size.shape[2]):
            if ii == 0:
                output = np.zeros((self._output_size[0], self._output_size[1], self.data.shape[2]))

            output[:,:,ii] = skimage.transform.resize(data[:,:,ii], self._output_size)
        return output


        return skimage.transform.resize(data, self._output_size)


class Crop(DataProcessing):

    def __init__(self, output_size):
        """
        :param output_size: a list or tuple of size 4 describing the start and stop rows and cols
        """
        # Not beautiful code
        if isinstance(output_size, dict):
            output_size = output_size['parameters']['output_size']

        if not isinstance(output_size, (list, tuple)) and not len(output_size) == 4:
            raise Exception('Data processing resize output_size parmaeter, {}, is wrong'.format(output_size))

        self._output_size = output_size

    def save(self):
        return {
            'data_processing_type': 'crop',
            'parameters': {
                'output_size': self._output_size
            }
        }

    def load(self, thedict):

        if not thedict['data_processing_type'] == 'crop':
            raise Exception('wrong data processing type {} for crop')

        self._output_size = thedict['output_size']

    def process(self, array):
        log.info('cropping data by {}'.format(self._output_size))
        rc = self._output_size
        return array[rc[0]:rc[1], rc[2]:rc[3]]


class GrayScale(DataProcessing):
    """
    Convert color image to gray scale
    """

    def __init__(self):
        """
        :param output_size: a list or tuple of size 4 describing the start and stop rows and cols
        """
        pass

    def save(self):
        return {
            'data_processing_type': 'gray_scale',
            'parameters': {
            }
        }

    def load(self, thedict):

        if not thedict['data_processing_type'] == 'gray_scale':
            raise Exception('wrong data processing type {} for crop')

    def process(self, array):
        log.info('')
        if len(array.shape) == 3:
            return np.dot(array[:, :, :3], [0.299, 0.587, 0.114])
        else:
            return array
