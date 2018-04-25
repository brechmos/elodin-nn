from transfer_learning.similarity import tSNE, Jaccard
import glob
import uuid
import pickle
import os
import sys

from transfer_learning.transfer_learning_process_data import TransferLearningProcessData
from transfer_learning.transfer_learning_display import TransferLearningDisplay
from transfer_learning.fingerprint import Fingerprint
from transfer_learning.cutouts import Cutouts

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("TransferLearning")
log.setLevel(logging.WARNING)


class TransferLearning:
    def __init__(self, cutout_creator=None, data_processing=[], fingerprint_calculator=None):
        # list of data processing elements
        self._data_processing = data_processing
        self._cutout_creator = cutout_creator
        self._fingerprint_calculator = fingerprint_calculator
        self._processing_dict = []
        self._fingerprints = []
        self._uuid = str(uuid.uuid4())

        # number of pixels around the image to include in the display
        self._image_display_margin = 50

        # Processed data
        self._tldp = []

    def set_processing(self, processing_dict):
        self._processing_dict = processing_dict

    def display(self, filename, row_minmax, col_minmax):

        if filename not in self._data_cache:
            log.info('Caching image data {}...'.format(filename))
            self._data_cache[filename] = self._load_image_data(filename)

        return self._data_cache[filename][row_minmax[0]:row_minmax[1], col_minmax[0]:col_minmax[1]]

    def calculate_stream(self, data_stream):
        """
        Calculate the fingerprints for each subsection of the image in each file.

        THe processing_dict should be a dictionary that contains keys of 
          filename -> The filename to process
          radec -> RA DEC
          meta -> all the other meta information that should be stored for further processing

        :param stepsize:
        :param display:
        :return:
        """
        log.debug('Calculate the fingerprints for each file and data processing step')

        self._processing_dict = data_stream

        # Run through each file.
        for to_process in self._processing_dict:
            self.calculate(to_process)

    def calculate(self, data):

        # Use the stream processing if the input is a list.
        if isinstance(data, list):
            self.calculate_stream(data)
            return

        for dp_set in self._data_processing:
            log.info("Processing filename {} with {}".format(data['filename'], dp_set))
            tldp_temp = TransferLearningProcessData(data, dp_set)
            tldp_temp.calculate(self._cutout_creator, self._fingerprint_calculator)
            self._tldp.append(tldp_temp)

    @property
    def fingerprints(self):
        """
        Retrieve the fingerprints from all the processed values. 

        :return: List of fingerprints
        """

        return_fingerprints = []
        for tldp in self._tldp:
            for fp in tldp.fingerprints:
                fp2 = fp.copy()
                fp2.update({'tldp': tldp})
                return_fingerprints.append(fp2)

        return return_fingerprints

    def save(self, output_location):
        """
        Save the information to the output location. If the output_location is a directory then
        save as a standard name. If output_location is a filename then save as it.

        :param output_location: Filename or directory of where to save
        :return:
        """

        # Create dictionary to save
        d = {
            'cutout_creator': self._cutout_creator.save(),
            'fingerprint_calculator': self._fingerprint_calculator.save(),
            'uuid': self._uuid,
            'processing_dict': self._processing_dict,
            'processed_data': [x.save() for x in self._tldp]
        }

        if output_location.endswith('pck'):
            output_filename = output_location
        else:
            output_filename = os.path.join(output_location, 'data_{}.pck'.format(self._uuid))

        # Save to a pickle file.
        with open(output_filename, 'wb') as fp:
            pickle.dump(d, fp)

    def _load(self, input_filename):
        """
        Load the dictionary of information from the pickle file listed as input_filename.

        :param input_filename:
        :return:
        """

        log.info('Loading TL from {}'.format(input_filename))
        with open(input_filename, 'rb') as fp:
            parameters = pickle.load(fp)

            self._cutout_creator = Cutouts.load_parameters(parameters['cutout_creator'])
            self._fingerprint_calculator = Fingerprint.load_parameters(parameters['fingerprint_calculator'])
            self._uuid = parameters['uuid']
            self._processing_dict = parameters['processing_dict']
            self._tldp = [TransferLearningProcessData.load(x) for x in parameters['processed_data']]



    @staticmethod
    def load(input_filename):
        """
        Static load method. Create an instance to TransferLearning and then load the information
        from the input filename.

        :param input_filename:
        :return:
        """
        d = TransferLearning()
        d._load(input_filename)

        return d


if __name__ == "__main__":

    input_file_pattern = '/Users/crjones/christmas/hubble/carina/data/carina.tiff'
    directory = '/tmp/resnet/'

    # input_file_pattern = '/Users/crjones/christmas/hubble/HSTHeritage/data/*.???'
    # directory = '/tmp/hst_heritage_gray'

    # input_file_pattern = '/Users/crjones/christmas/hubble/HSTHeritage/data/*.???'
    # directory = '/tmp/hst_heritage'

    # input_file_pattern = '/Users/crjones/christmas/hubble/MAGPIS/G371D.tiff'
    # directory = '/tmp/magpis'

    # input_file_pattern = '/Users/crjones/christmas/hubble/MAGPIS/G371D.tiff'
    # directory = '/tmp/magpis_gray'

    # input_file_pattern = '/Users/crjones/christmas/hubble/MAGPIS/G371D.tiff'
    # directory = '/tmp/magpis_gray'

    # input_file_pattern = '/Users/crjones/christmas/hubble/MAGPIS/G371D.tiff'
    # directory = '/tmp/magpis_gray_zoom'

    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except:
            log.error('Could not create output directory {}'.format(directory))
            sys.exit(-1)

    filenames = glob.glob(os.path.join(directory, '*pck'))

    if len(filenames) == 0:
        print('Processing data...')

        from data_processing import MedianFilterData, ZoomData, RotateData, GrayScaleData
        from fingerprint import FingerprintResnet, FingerprintInceptionV3

        stepsize = 400

        input_filenames = glob.glob(input_file_pattern)

        fingerprint_model = FingerprintInceptionV3()
        # fingerprint_model = FingerprintResnet()

        # # calculate fingerpirnts for median filtered
        log.info('Setting up median filter data')
        #data_processing = [MedianFilterData((3, 3, 1)), GrayScaleData()]
        data_processing = [GrayScaleData()]
        tl = TransferLearning(fingerprint_model, data_processing)
        tl.set_files(input_filenames)
        fingerprints = tl.calculate(stepsize=stepsize, display=True)
        tl.save(directory)

    else:

        fingerprints = []
        for filename in filenames:
            data = TransferLearning.load(filename)
            fingerprints.extend(data.fingerprints)

        similarity = Jaccard(fingerprints)
        #similarity = tSNE(fingerprints)
        tld = TransferLearningDisplay(similarity)
        tld.show(fingerprints)
