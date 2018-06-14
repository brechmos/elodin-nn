import uuid
import itertools
import numpy as np

from skimage import measure
from skimage import filters

from transfer_learning.data import Data, DataCollection
from transfer_learning.cutout import Cutout, CutoutCollection

from ..tl_logging import get_logger
log = get_logger('cutout generator')

"""
The Cutout Generators should be instanciated and then run on a Data object in order
to generate Cutout objects. Might be one or more cutouts per call.
"""


class BasicCutoutGenerator:
    """
    The BasicCutouts class will create and return cutouts based on sliding window idea given a step_size.

    If there are any cutout_processing elements they are passed into each created cutout.
    """

    def __init__(self, output_size, step_size):
        """
        Basic cutout generator based on a sliding window technique.

        The idea is one creates this genrerator and then the ``create_cutouts()``
        is called on a ``transfer_learning.data.Data`` instance.

        Parameters
        -----------
        output_size : tuple
            Size of the cutout, must be a pair of numbers (e.g., (224, 224) ).

        step_size : number
            Size of the steps taken for the sliding window cutouts. At this point
            step_size is used in both row and column direction.

        """
        self._output_size = output_size
        self._step_size = step_size
        self._uuid = str(uuid.uuid4())

    def __str__(self):
        return 'Basic Cutout (step_size={})'.format(self._step_size)

    def number_cutouts(self, data):
        """
        Number of cutouts given the data, this was implemented primarily for the progress bar.  If the create_cutouts
        function changes then this one will have to as well.

        Parameters
        -----------
        data : `transfer_learning.data.Data`
            The dataset.

        Return
        ------
        ncutouts: number
            number of cutouts that will be created in `create_cutouts()` below

        """

        N = self._output_size // 2

        #
        # Determine the centers to use for the fingerprint calculation
        #

        nrows, ncols = data.shape[:2]
        rows = range(N, nrows-N, self._step_size)
        cols = range(N, ncols-N, self._step_size)

        return len(list(itertools.product(rows, cols)))

    def create_cutouts(self, data, cutout_processing=None):
        """
        Calculate the fingerprints for each subsection of the image in each file.

        Parameters
        -----------
        data : `transfer_learning.data.Data` or `transfer_learning.data.DataCollection`
            The dataset.

        cutout_processing : list, optional
            List of instances of ``transfer_learning.cutout.cutout_processing``.

        Return
        ------
        cutouts: CutoutCollection
            The cutouts that came from this generator.
        """
        log.info('Creating basic sliding window cutouts')

        if isinstance(data, Data):
            return self._create_cutouts_data(data, cutout_processing)
        else:
            cutout_collection = CutoutCollection()

            for datum in data:
                cutout_collection += self._create_cutouts_data(datum, cutout_processing)

        return cutout_collection

    def _create_cutouts_data(self, data, cutout_processing=None):
        """
        Calculate the fingerprints for each subsection of the image in each file.

        Parameters
        -----------
        data : `transfer_learning.data.Data` or `transfer_learning.data.DataCollection`
            The dataset.

        cutout_processing : list, optional
            List of instances of ``transfer_learning.cutout.cutout_processing``.

        Return
        ------
        cutouts: CutoutCollection
            The cutouts that came from this generator.
        """
        log.info('Creating basic sliding window cutouts')

        N = self._output_size // 2

        if cutout_processing is not None and isinstance(cutout_processing, list):
            raise ValueError('Cutout processing must be empty or a list')

        #
        # Determine the centers to use for the fingerprint calculation
        #

        nrows, ncols = data.shape[:2]
        rows = range(N, nrows-N, self._step_size)
        cols = range(N, ncols-N, self._step_size)

        #
        # Run over all the rows and columns and yield the result
        #

        cutout_collection = CutoutCollection()
        for ii, (row, col) in enumerate(itertools.product(rows, cols)):
            cutout_collection.add(Cutout(data, [row-N, row+N, col-N, col+N], self.save(),
                                  cutout_processing=cutout_processing))

        return cutout_collection

    def save(self):
        """
        This method is used to create a representation of this instance of a cutout which can be saved
        in a pickle file and then later loaded.

        Return
        ------
        dict_representation : dict
            Information required to re-create this instance.

        """

        return {
            'cutout_type': self.__class__.__name__,
            'output_size': self._output_size,
            'step_size': self._step_size,
            'uuid': self._uuid
        }

    def _load(self, parameters):
        """
        Load the instance internal variables from the parameters dictionary

        Parameters
        -----------
        parameters : dict
            dict of info to recreate the instance.

        Notes
        -----
        This must match what is in the save().

        """
        self._step_size = parameters['step_size']
        self._output_size = parameters['output_size']
        self._uuid = parameters['uuid']


class FullImageCutoutGenerator:
    """
    The FullImageCutout class will create a single cutout based on resizing the input image to the output size.
    """

    def __init__(self, output_size):
        """
        Initialize with the output_size (assumed square at this point and a step_size

        Parameters
        -----------
        output_size : tuple
            Size of the cutout, must be a pair of numbers (e.g., (224, 224) ).

        """
        if not isinstance(output_size, (list, tuple)):
            raise Exception('FullImageCutout must be passed a list or tuple, not {}'.format(output_size))

        self._output_size = output_size
        self._uuid = str(uuid.uuid4())

    def __str__(self):
        return 'Full Image Cutout'

    def number_cutouts(self, data):
        """
        Number of cutouts is 1.

        Parameters
        -----------
        data : `transfer_learning.data.Data`
            The dataset.

        """
        return 1

    def create_cutouts(self, data, cutout_processing=None):
        """
        Create the cutouts based on the data input.  Though, only one
        cutout is created per datum.

        Parameters
        -----------
        data : `transfer_learning.data.Data` or `transfer_learning.data.DataCollection`
            The dataset.

        cutout_processing : list, optional
            List of instances of ``transfer_learning.cutout.cutout_processing``.

        """
        log.info('Creating new cutout from data {}'.format(data.uuid))
        if isinstance(data, Data):
            return Cutout(data, [0, self._output_size[0], 0, self._output_size[1]], self.save(),
                          cutout_processing=cutout_processing)
        elif isinstance(data, DataCollection):
            cc = CutoutCollection()
            for datum in data:
                cc.add(Cutout(datum, [0, self._output_size[0], 0, self._output_size[1]], self.save(),
                              cutout_processing=cutout_processing))
            return cc

    def save(self):
        """
        This method is used to create a representation of this instance of a cutout which can be saved
        in a pickle file and then later loaded.

        Return
        ------
        dict_representation : dict
            Information required to re-create this instance.

        """
        return {
            'cutout_type': self.__class__.__name__,
            'output_size': self._output_size,
            'uuid': self._uuid
        }

    def _load(self, parameters):
        """
        Load the instance internal variables from the parameters dictionary

        Parameters
        -----------
        parameters : dict
            dict of info to recreate the instance.

        Notes
        -----
        This must match what is in the save().

        """
        self._output_size = parameters['output_size']
        self._uuid = parameters['uuid']


class BlobCutoutGenerator:
    """
    Blob cutouts is a little different way of creating cutouts. Rather than doing a sliding window we are going
    to use some image processing and labeling.  The basic algorithm is to smooth the image, label the connected
    components and then return 224x224 cutouts based on the center of each connected component.
    """

    def __init__(self, output_size, mean_threshold=2.0, gaussian_smoothing_sigma=10, label_padding=80):
        """
        Initialize the labeled connected component cutout creator.

        Parameters
        -----------
        output_size : tuple
            Output size, assumed square at this point

        mean_threshold : number
            Signal threshold passed in to the labeling in order to determine connectedness

        gaussian_smoothing_sigma : number
            Amount of smoothing to apply to the image before labeling

        label_padding : number
            How much do we want to pad around the output image.

        """
        # list of data processing elements
        self._output_size = output_size
        self._mean_threshold = mean_threshold
        self._gaussian_smoothing_sigma = gaussian_smoothing_sigma
        self._label_padding = label_padding
        self._uuid = str(uuid.uuid4())

    def __str__(self):
        return 'Blob Cutout (mean_threshold={}, gaussian_smoothign_sigma={})'.format(self._mean_threshold,
                                                                                     self._gaussian_smoothing_sigma)

    def _create_labels(self, data):
        """
        Helper function that actually creates the blobs as this is used
        in the number_cutouts() method and the create_cutouts() method.

        Parameters
        -----------
        data : `transfer_learning.data.Data`
            The dataset.

        """
        gray_image = np.dot(data[:, :, :3], [0.299, 0.587, 0.114])
        im = filters.gaussian_filter(gray_image, sigma=self._gaussian_smoothing_sigma)
        blobs = im > self._mean_threshold * im.mean()
        blobs_labels = measure.label(blobs, background=0)

        return blobs_labels

    def number_cutouts(self, data):
        """
        Number of cutouts given the data, this was implemented primarily for the progress bar.  If the create_cutouts
        function changes then this one will have to as well.

        Parameters
        -----------
        data : `transfer_learning.data.Data`
            The dataset.

        """
        blobs = self._create_labels(data)
        return blobs.max()

    def create_cutouts(self, data, cutout_processing=None):
        """
        This will create a number of cutouts. Each cutout is based on a connected component labeling of the image
        after a little smoothing. This is a reasonably standard method of doing the connnected component labeling.

        Parameters
        -----------
        data : `transfer_learning.data.Data`
            The dataset.

        cutout_processing : list, optional
            List of instances of ``transfer_learning.cutout.cutout_processing``.

        Notes
        -----
        There is some magic that happens here. The short and the long
        of it is the image is converted to gray-scale, a threshold is
        applied and then labeled (all using standard methods).

        Once the labels are found, then they are cutout with a small
        padding around each labeled blob. Then for each small cutout
        resize to the required output size.

        See also
        --------
        NDDataArray
        """

        if isinstance(data, Data):
            return self._create_cutouts_data(data, cutout_processing)
        else:
            cutout_collection = CutoutCollection()

            for datum in data:
                cutout_collection += self._create_cutouts_data(datum, cutout_processing)

        return cutout_collection

    def _create_cutouts_data(self, data, cutout_processing=None):

        #
        # Find the blobs - make gray scale, smooth, find blobs, then label them
        #

        blobs_labels = self._create_labels(data.get_data())

        #
        # Loop over the labeled blob
        #

        cutouts = CutoutCollection()
        for ii in range(1, blobs_labels.max()):
            b = blobs_labels == ii

            #
            # Find the extent of the image
            #

            rows, cols = np.nonzero(b)
            min_row, max_row = int(rows.min()), int(rows.max())
            min_col, max_col = int(cols.min()), int(cols.max())

            log.debug('label mins and maxs {} {}   {} {}'.format(min_row, max_row, min_col, max_col))

            #
            # Pad the blob a bit
            #

            min_row = max(min_row - self._label_padding, 0)
            max_row = min(max_row + self._label_padding, data.shape[0])
            min_col = max(min_col - self._label_padding, 0)
            max_col = min(max_col + self._label_padding, data.shape[1])

            log.debug('label mins and maxs with padding {} {}   {} {}'.format(
                min_row, max_row, min_col, max_col))

            #
            # Correct so it is square
            #

            rr = max_row - min_row
            cc = max_col - min_col
            if rr > cc:
                min_col = min_col - (rr - cc) // 2
                max_col = max_col - (rr - cc) // 2
            else:
                min_row = min_row - (cc - rr) // 2
                max_row = max_row - (cc - rr) // 2

            #
            # If the cutout extends outside the original image then we are going to drop it for now
            # TODO: Need to create a way to shift the cutout if outside the original image.
            #

            if min_row < 0 or min_col < 0 or max_row >= data.shape[0] or max_col >= data.shape[1]:
                continue

            log.debug('Cutting out {} {}   {} {}'.format(min_row, max_row, min_col, max_col))

            #
            # This will need to match the standard return method for cutouts.
            #

            cutouts.add(Cutout(data, [min_row, max_row, min_col, max_col], self.save(),
                               cutout_processing=cutout_processing))

        return cutouts

    def save(self):
        """
        This method is used to create a representation of this instance of a cutout which can be saved
        in a pickle file and then later loaded.

        Return
        ------
        dict_representation : dict
            Information required to re-create this instance.

        """
        return {
            'cutout_type': self.__class__.__name__,
            'output_size': self._output_size,
            'mean_threshold': self._mean_threshold,
            'gaussian_smoothing_sigma': self._gaussian_smoothing_sigma,
            'label_padding': self._label_padding,
            'uuid': self._uuid
        }

    def _load(self, parameters):
        """
        Load the instance internal variables from the parameters dictionary

        Parameters
        -----------
        parameters : dict
            dict of info to recreate the instance.

        Notes
        -----
        This must match what is in the save().

        """

        self._output_size = parameters['output_size']
        self._gaussian_smoothing_sigma = parameters['gaussian_smoothing_sigma']
        self._label_padding = parameters['label_padding']
        self._mean_threshold = parameters['mean_threshold']
        self._uuid = parameters['uuid']
