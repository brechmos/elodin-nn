from skimage.transform import resize as skimage_resize
from ..tl_logging import get_logger
import logging
log = get_logger('cutout processing')


class CutoutProcessing(object):

    @staticmethod
    def load(thedict):
        log.info('dict is {}'.format(thedict))
        if thedict['cutout_processing_type'] == 'resize':
            return Resize(thedict)
        elif thedict['cutout_processing_type'] == 'crop':
            return Crop(thedict)


class Resize(CutoutProcessing):

    def __init__(self, output_size):
        """
        :param output_size: a list or tuple of size 2 describing the output size.
        """

        # Not beautiful code
        if isinstance(output_size, dict):
            output_size = output_size['parameters']['output_size']

        if not isinstance(output_size, (list, tuple)) and not len(output_size) == 2:
            raise Exception('Cutout processing resize output_size parmaeter, {}, is wrong'.format(output_size))

        self._output_size = output_size

    def __str__(self):
        return 'Cutout Resize to {}'.format(self._output_size)

    def save(self):
        log.info('')
        return {
            'cutout_processing_type': 'resize',
            'parameters': {
                'output_size': self._output_size
            }
        }

    def load(self, thedict):
        log.info('')

        if not thedict['cutout_processing_type'] == 'resize':
            raise Exception('wrong cutout processing type {} for resize')

        self._output_size = thedict['output_size']

    def process(self, numpy_data):
        log.debug('resizing data to {}'.format(self._output_size))
        return skimage_resize(numpy_data, self._output_size)


class Crop(CutoutProcessing):

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

    def __str__(self):
        return 'Cutout Crop to {}'.format(self._output_size)

    def save(self):
        log.info('')
        return {
            'cutout_processing_type': 'crop',
            'parameters': {
                'output_size': self._output_size
            }
        }

    def load(self, thedict):
        log.info('')

        if not thedict['cutout_processing_type'] == 'crop':
            raise Exception('wrong data processing type {} for crop')

        self._output_size = thedict['output_size']

    def process(self, numpy_data):
        log.debug('cropping data by {}'.format(self._output_size))
        rc = self._output_size
        return numpy_data[rc[0]:rc[1], rc[2]:rc[3]]
