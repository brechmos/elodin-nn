import logging

import skimage

logging.basicConfig(format='%(levelname)-6s: %(asctime)-15s %(name)-10s %(funcName)-10s %(message)s')
log = logging.getLogger("data")
fhandler = logging.FileHandler(filename='/tmp/mylog.log', mode='a')
log.addHandler(fhandler)
log.setLevel(logging.INFO)


class DataProcessing(object):

    @staticmethod
    def load(thedict):
        if thedict['data_processing_type'] == 'resize':
            return Resize(thedict)
        elif thedict['data_processing_type'] == 'crop':
            return Crop(thedict)


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
        log.debug('resizeing data to {}'.format(self._output_size))
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
        log.debug('cropping data by {}'.format(self._output_size))
        rc = self._output_size
        return array[rc[0]:rc[1], rc[2]:rc[3]]
