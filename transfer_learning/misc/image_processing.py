from skimage import exposure
from ..tl_logging import get_logger
log = get_logger('cutout processing')


def rescale_intensity(self, numpy_data, lower_percentile, upper_percentile):
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
        lower_percentile, upper_percentile))
    return exposure.rescale_intensity(numpy_data,
                                      in_range=(lower_percentile, upper_percentile))


def histogram_equalization(numpy_data):
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


def adaptive_histogram_equalization(numpy_data, clip_limit=0.03):
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
    log.info('adaptive histogram equalization with clip_limit {}'.format(clip_limit))
    return exposure.equalize_adapthist(numpy_data, clip_limit=clip_limit)
