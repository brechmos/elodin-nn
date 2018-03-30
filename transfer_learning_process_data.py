import imageio
import numpy as np
import uuid
import multiprocessing

from astropy.io import fits

import utils
from data_processing import DataProcessing
from fingerprint import Fingerprint
from cutouts import Cutouts

import progressbar

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("TransferLearningProcessData")
log.setLevel(logging.WARNING)


class TransferLearningProcessData:
    """
    This contains data processed on a file using one type of data_processing and once calculate is called
    will contain the fingerprints for each cutout.

    The two differences this module needs to encapsulate from other related processing:

        1. Input files
        2. Data Processing method

    For example, we might have 8-10 files that need to be processed and each one we want to process
    with two different pre-processing methods (maybe for zooming or something).

    So, we really only need to encapsulate the file/data and data pre-processing methods.
    """

    def __init__(self, file_meta, data_processing):
        """
        The file_meta input should be a dictionary that contains the keys:
            filename
            radec
            meta
        """

        log.info('Creating TransferLearningProcessData instance with {} and {}'.format(
            file_meta['filename'], data_processing))

        self._uuid = str(uuid.uuid4())

        # The file_meta is a dictionary that contains keys:
        #    filename - string filename
        #    radec - tuple of (ra dec)
        #    meta - dictionary of meta information
        # radec and meta could be empty
        if not set(file_meta.keys()) == set(['filename', 'radec', 'meta']):
            log.error('First parameter to TransferLearningProcessData must be a dict containing filename, radec and meta')
            raise ValueError('First parameter to TransferLearningProcessData must be a dict containing filename, radec and meta')
        self._file_meta = file_meta

        # We need to check to see if the input to this is a dictionary or not
        if len(data_processing) > 0 and isinstance(data_processing[0], dict):
            self._data_processing = [DataProcessing.load_parameters(x) for x in data_processing]
        else:
            self._data_processing = data_processing

        self._original_data = None
        self._processed_data = self._load_image_data(file_meta['filename'])

        # Set in the calculate function
        self._cutout_creator = None
        self._fingerprint_calculator = None
        self._fingerprints = []

    @property
    def data_processing(self):
        return self._data_processing

    @property
    def filename(self):
        return self._filename

    @property
    def fingerprints(self):
        return self._fingerprints

    def _load_image_data(self, filename):
        """
        Load the file and apply the processing.

        :param filename:
        :return:
        """

        if any(filename.lower().endswith(s) for s in ['tiff', 'tif', 'jpg']):
            log.debug('Loading TIFF/JPG file {}'.format(filename))
            data = np.array(imageio.imread(filename))
        elif 'fits' in filename:
            log.debug('Loading FITS file {}'.format(filename))
            data = fits.open(filename)[1].data
            log.debug('FITS data is {}'.format(data))

            # There are some cases where the data might be NaN or Inf.  In those cases we'll set to 0.
            data[~np.isfinite(data)] = 0
        else:
            log.warning('Could not determine filetype for {}'.format(filename))
            return []

        self._data_original = data

        # Apply the data processing to the loaded dataset
        for dp in self._data_processing:
            log.debug('Doing pre-processing {}, input data shape {}'.format(dp, data.shape))
            data = dp.process(data)
            log.debug('    Now input data shape {}'.format(data.shape))

        # Make RGB (3 channel) if only gray scale (single channel)
        if len(data.shape) == 2:
            data = utils.gray2rgb(data)

        return data

    def calculate(self, cutout_creator, fingerprint_calculator):
        """
        Calculate the fingerprints for each cutout based on the cutout creator and fingerprint calculator.

        :param cutout_creator:
        :param fingerprint_calculator:
        :return:
        """
        log.info("Calculating fingerprints using {} and {}".format(cutout_creator, fingerprint_calculator))

        self._fingerprints = []
        self._cutout_creator = cutout_creator
        self._fingerprint_calculator = fingerprint_calculator

        # Create the progressbar
        N = self._cutout_creator.number_cutouts(self._processed_data)
        pbar = progressbar.ProgressBar(max_value=N)

        # Loop over each cutout, calculate the predictions based on imagenet and then store into a dictionary
        for ii, (row_min, row_max, col_min, col_max, td) in enumerate(self._cutout_creator.create_cutouts(self._processed_data)):

            predictions = self._fingerprint_calculator.calculate(td)

            self._fingerprints.append({
                'row_min': row_min,
                'row_max': row_max,
                'col_min': col_min,
                'col_max': col_max,
                'predictions': predictions
            })

            pbar.update(ii)

    def display(self, row_minmax, col_minmax):
        """
        Send the data back to the calling routine to display the data.

        :param row_minmax:
        :param col_minmax:
        :return:
        """
        return self._processed_data[row_minmax[0]:row_minmax[1], col_minmax[0]:col_minmax[1]]

    def display_original(self, row_minmax, col_minmax):
        """
        Send the data back to the calling routine to display the data.

        :param row_minmax:
        :param col_minmax:
        :return:
        """
        return self._original_data[row_minmax[0]:row_minmax[1], col_minmax[0]:col_minmax[1]]

    def save(self):
        """
        Save is to a dictionary as it is used higher up the food chain.

        :return: dictionary of all the relevant information for this instance of the class
        """
        return {
            'uuid': self._uuid,
            'file_meta': self._file_meta,
            'data_processing': [x.save() for x in self._data_processing],
            'cutout_creator': self._cutout_creator.save(),
            'fingerprint_calculator': self._fingerprint_calculator.save(),
            'fingerprints': self._fingerprints
        }

    def _load(self, parameters):
        """
        Load the parameters back into the instance.  Will need to create the instances as we load them in.

        :param parameters: Dictionary of parametesr that comes from the above `save()` command
        :return:
        """
        log.debug('TransferLearningProcessData loading {}'.format(parameters['file_meta']['filename']))

        self._uuid = parameters['uuid']
        self._file_meta = parameters['file_meta']
        log.debug('TransferLearningProcessData loading data pocessing')
        self._data_processing = [DataProcessing.load_parameters(x) for x in parameters['data_processing']]
        log.debug('TransferLearningProcessData loading ciutouts')
        self._cutout_creator = Cutouts.load_parameters(parameters['cutout_creator'])
        log.debug('TransferLearningProcessData loading fingerprints')
        self._fingerprint_calculator = Fingerprint.load_parameters(parameters['fingerprint_calculator'])
        self._fingerprints = parameters['fingerprints']

        log.debug('TransferLearningProcessData ... done')

    @staticmethod
    def load(parameters):
        """
        Static class method that will create an instance of the Transfer LearningProcessData class with the
        data loaded in based on the saved parameters.

        :param parameters:
        :return: instance of TLPD loaded with all the appropriate goodness
        """

        tldp = TransferLearningProcessData(parameters['file_meta'], parameters['data_processing'])
        tldp._load(parameters)

        return tldp
