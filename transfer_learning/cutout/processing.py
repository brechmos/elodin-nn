from skimage.transform import resize as skimage_resize
from skimage import exposure
from ..tl_logging import get_logger
import logging
log = get_logger('cutout processing')


class CutoutProcessing(object):

    @staticmethod
    def load(thedict):
        log.info('dict is {}'.format(thedict))
        if thedict['cutout_processing_type'] == 'resize':
            return Resize.load(thedict)
        elif thedict['cutout_processing_type'] == 'crop':
            return Crop.load(thedict)
        elif thedict['cutout_processing_type'] == 'rescale_intensity':
            return RescaleIntensity.load(thedict)
        elif thedict['cutout_processing_type'] == 'equalize_histogram':
            return HistogramEqualization.load(thedict)
        elif thedict['cutout_processing_type'] == 'adaptive_histogram_equalization':
            return AdaptiveHistogramEqualization.load(thedict)


class Resize(CutoutProcessing):

    def __init__(self, output_size):
        """
        :param output_size: a list or tuple of size 2 describing the output size.
        """
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

    @staticmethod
    def load(self, thedict):
        """
        Static class method that takes a dict object from ``save()`` and
        creates an instance of Resize based on the parmaeters in thedict.

        Parameters
        ----------
        thedict : dict
            Dictionary of information from ``save()``

	Return
	------
        resize : Instance of class Resize.
            Will load the parameters.

        """
        log.info('')

        if not thedict['cutout_processing_type'] == 'resize':
            raise Exception('wrong cutout processing type {} for resize')

        return Resize(thedict['output_size'])

    def process(self, numpy_data):
        log.debug('resizing data to {}'.format(self._output_size))
        return skimage_resize(numpy_data, self._output_size)


class Crop(CutoutProcessing):

    def __init__(self, output_size):
        """
        :param output_size: a list or tuple of size 4 describing the start and stop rows and cols
        """
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

    @staticmethod
    def load(self, thedict):
        """
        Static class method that takes a dict object from ``save()`` and
        creates an instance of Crop based on the parmaeters in thedict.

        Parameters
        ----------
        thedict : dict
            Dictionary of information from ``save()``

	Return
	------
        resize : Instance of class Crop.
            Will load the parameters.

        """
        log.info('')

        if not thedict['cutout_processing_type'] == 'crop':
            raise Exception('wrong data processing type {} for crop')

        return Crop(output_size=thedict['output_size'])

    def process(self, numpy_data):
        log.debug('cropping data by {}'.format(self._output_size))
        rc = self._output_size
        return numpy_data[rc[0]:rc[1], rc[2]:rc[3]]


class RescaleIntensity(CutoutProcessing):

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
        # *Not beautiful code*
        if isinstance(lower_percentile, dict):
            lower_percentile = output_size['parameters']['lower_percentile']
            upper_percentile = output_size['parameters']['upper_percentile']

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
            'cutout_processing_type': 'rescale_intensity',
            'parameters': {
                'lower_percentile': self._lower_percentile,
                'upper_percentile': self._upper_percentile
            }
        }

    @staticmethod
    def load(self, thedict):
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

        if not thedict['cutout_processing_type'] == 'rescale_intensity':
            raise Exception('wrong data processing type {} for rescale_intensity')

        return RescaleIntensity(lower_percentile=self._lower_percentile,
                                upper_percentile=self._upper_percentile)

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
        return exposure.rescale_intensity(numpy_data, 
                                          in_range=(self._lower_percentile, self._upper_percentile))


class HistogramEqualization(CutoutProcessing):

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
            'cutout_processing_type': 'histogram_equalization',
            'parameters': { }
        }

    @staticmethod
    def load(self, thedict):
        """
        Load the parameters from a dict.

        Parameters
        ----------
        thedict : dict
            Dictionary from the ``save()`` method above.

        """
        log.info('')

        if not thedict['cutout_processing_type'] == 'histogram_equalization':
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
        return exposure.equalize_hist(numpy_data)


class AdaptiveHistogramEqualization(CutoutProcessing):

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
            'cutout_processing_type': 'adaptive_histogram_equalization',
            'parameters': { 
                'clip_limit': self._clip_limit
            }
        }

    @staticmethod
    def load(self, thedict):
        """
        Load the parameters from a dict.

        Parameters
        ----------
        thedict : dict
            Dictionary from the ``save()`` method above.

        """
        log.info('')

        if not thedict['cutout_processing_type'] == 'adaptive_histogram_equalization':
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
        return exposure.equalize_adapthist(numpy_data, clip_limit=self._clip_limit)
