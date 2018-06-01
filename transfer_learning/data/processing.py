import skimage
import numpy as np
from ..tl_logging import get_logger
from ..misc import image_processing
log = get_logger('data processing')


class DataProcessing(object):

    @staticmethod
    def load(thedict):
        """
        Class static loader method to call the correct
        loader based on the paramters in ``thedict``.

        Parameters
        -----------
        thedict : dict
            Dictionary of saved parameters, hopefully from one of the
            save() methods below.

        Raises
        ------
        ValueError
            If no DataProcessing subclass found, then raise an error.

        """
        print(thedict)
        if 'data_processing_type' in thedict:
            for subclass in DataProcessing.__subclasses__():
                if subclass._name == thedict['data_processing_type']:
                    return subclass.load(thedict)

        raise ValueError('No class found for data processing.')


class Resize(DataProcessing):

    _name = 'resize'

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
            'data_processing_type': self._name,
            'parameters': {
                'output_size': self._output_size
            }
        }

    @staticmethod
    def load(thedict):
        if not thedict['data_processing_type'] == Resize._name:
            raise Exception('wrong data processing type {} for resize')

        return Resize(output_size=thedict['output_size'])

    def process(self, data):
        log.info('resizeing data to {}'.format(self._output_size))

        for ii in range(self._output_size.shape[2]):
            if ii == 0:
                output = np.zeros((self._output_size[0], self._output_size[1], self.data.shape[2]))

            output[:, :, ii] = skimage.transform.resize(data[:, :, ii], self._output_size)
        return output


class Crop(DataProcessing):

    _name = 'crop'

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
            'data_processing_type': self._name,
            'parameters': {
                'output_size': self._output_size
            }
        }

    @staticmethod
    def load(thedict):

        if not thedict['data_processing_type'] == Crop._name:
            raise Exception('wrong data processing type {} for crop')

        return Crop(output_size=thedict['output_size'])

    def process(self, array):
        log.info('cropping data by {}'.format(self._output_size))
        rc = self._output_size
        return array[rc[0]:rc[1], rc[2]:rc[3]]


class GrayScale(DataProcessing):
    """
    Convert color image to gray scale
    """

    _name = 'grayscale'

    def __init__(self):
        """
        :param output_size: a list or tuple of size 4 describing the start and stop rows and cols
        """
        pass

    def save(self):
        return {
            'data_processing_type': self._name,
            'parameters': {
            }
        }

    @staticmethod
    def load(thedict):

        if not thedict['data_processing_type'] == GrayScale._name:
            raise Exception('wrong data processing type {} for grayscale')

        return GrayScale()

    def process(self, array):
        log.info('')
        if len(array.shape) == 3:
            return np.dot(array[:, :, :3], [0.299, 0.587, 0.114])
        else:
            return array


class RescaleIntensity(DataProcessing):

    _name = 'rescale_intensity'

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

        self._lower_percentile = lower_percentile
        self._upper_percentile = upper_percentile

    def __str__(self):
        return 'Cutout RescaleIntensity {} {}'.format(self._lower_percentile, self._upper_percentile)

    def save(self):
        """
        Save the class to a dict

        Return
        ------
        thdict : dict
            Parameters to re-create this instance.

        """
        log.info('')
        return {
            'data_processing_type': self._name,
            'parameters': {
                'lower_percentile': self._lower_percentile,
                'upper_percentile': self._upper_percentile
            }
        }

    @staticmethod
    def load(thedict):
        """
        Load the parameters from a dict.

        Parameters
        ----------
        thedict : dict
            Dictionary from the ``save()`` method above.

        Return
        ------
        rescaler: instance of RescaleIntenisty

        """
        log.info('')

        if not thedict['data_processing_type'] == RescaleIntensity._name:
            raise Exception('wrong data processing type {} for rescale_intensity')

        return RescaleIntensity(lower_percentile=thedict['parameters']['lower_percentile'],
                                upper_percentile=thedict['parameters']['upper_percentile'])

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
        log.info('rescale intensity to percentile range {} {}'.format(
            self._lower_percentile, self._upper_percentile))
        return image_processing.rescale_intensity(numpy_data,
                                                  in_range=(self._lower_percentile, self._upper_percentile))


class HistogramEqualization(DataProcessing):

    _name = 'histogram_equalization'

    def __init__(self):
        pass

    def __str__(self):
        return 'Cutout Histogram Equalization'

    def save(self):
        """
        Save the class to a dict

        Return
        ------
        thdict : dict
            Parameters to re-create this instance.

        """
        log.info('')
        return {
            'data_processing_type': self._name,
            'parameters': {}
        }

    @staticmethod
    def load(thedict):
        """
        Load the parameters from a dict.

        Parameters
        ----------
        thedict : dict
            Dictionary from the ``save()`` method above.

        """
        log.info('')

        if not thedict['data_processing_type'] == HistogramEqualization._name:
            raise Exception('wrong data processing type {} for histogram equalization')

        return HistogramEqualization()

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
        return image_processing.histogram_equalization(numpy_data)


class AdaptiveHistogramEqualization(DataProcessing):

    _name = 'adaptive_histogram_equalization'

    def __init__(self, clip_limit=0.03):
        """
        Initializer

        Parameters
        ----------
        clip_limit : number
            Clip limit will be passed to the scikit-image method.

        """
        self._clip_limit = clip_limit

    def __str__(self):
        return 'Cutout Histogram Adaptive Equalization'

    def save(self):
        """
        Save the class to a dict

        Return
        ------
        thdict : dict
            Parameters to re-create this instance.

        """
        log.info('')
        return {
            'data_processing_type': self._name,
            'parameters': {
                'clip_limit': self._clip_limit
            }
        }

    @staticmethod
    def load(thedict):
        """
        Load the parameters from a dict.

        Parameters
        ----------
        thedict : dict
            Dictionary from the ``save()`` method above.

        """
        log.info('')

        if not thedict['data_processing_type'] == AdaptiveHistogramEqualization._name:
            raise Exception('wrong data processing for adaptive histogram equalization')

        return AdaptiveHistogramEqualization(clip_limit=thedict['parameters']['clip_limit'])

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
        log.info('adaptive histogram equalization with clip_limit {}'.format(self._clip))
        return image_processing.equalize_adapthist(numpy_data, clip_limit=self._clip_limit)
