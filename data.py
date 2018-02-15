import uuid
import numpy as np
import pickle
import imageio
from astropy.io import fits
import progressbar
import itertools
import os

import matplotlib.pyplot as plt

from data_processing import DataProcessing
from fingerprint import Fingerprint
import utils

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("Data")
log.setLevel(logging.INFO)


class Data:
    def __init__(self, fingerprint_calculator=None, data_processing=[]):
        # list of data processing elements
        self._data_processing = data_processing
        self._fingerprint_calculator = fingerprint_calculator
        self._filenames = []
        self._fingerprints = []
        self._uuid = str(uuid.uuid4())

        self._data_cache = {}

    def _load_image_data(self, filename):
        if any(filename.lower().endswith(s) for s in ['tiff', 'tif', 'jpg']):
            log.debug('Loading TIFF/JPG file {}'.format(filename))
            data = np.array(imageio.imread(filename))
        elif 'fits' in filename:
            log.debug('Loading FITS file {}'.format(filename))
            data = fits.open(filename)[1].data
            log.debug('FITS data is {}'.format(data))

            data[~np.isfinite(data)] = 0
        else:
            log.warning('Could not determine filetype for {}'.format(filename))
            return []

        # Do the pre-processing of the data
        for dp in self._data_processing:
            log.debug('Doing pre-processing {}, input data shape {}'.format(dp, data.shape))
            data = dp.process(data)
            log.debug('    Now input data shape {}'.format(data.shape))

        # Make RGB (3 channel) if only gray scale (single channel)
        if len(data.shape) == 2:
            data = utils.gray2rgb(data)

        return data

    def set_files(self, filenames):
        self._filenames = filenames

    def display(self, filename, row, col):

        if filename not in self._data_cache:
            self._data_cache[filename] = self._load_image_data(filename)

        return self._data_cache[filename][row-112:row+112, col-112:col+112]
        log.info('Display {} {} {} {}'.format(self, self._data_processing, row, col))

    @property
    def fingerprints(self):
        return self._fingerprints

    def calculate(self, stepsize, display=False):
        """
        Calculate the fingerprints for each subsection of the image in each file.

        :param stepsize:
        :param display:
        :return:
        """

        self._fingerprints = []

        if display:
            plt.ion()
            fig = plt.figure()
            imaxis = None

        log.debug('After the plot display')
        # Run through each file.
        for filename in self._filenames:
            log.info('Processing filename {}'.format(filename))

            # Load the data
            data = self._load_image_data(filename)

            # Determine the centers to use for the fingerprint calculation
            nrows, ncols = data.shape[:2]
            rows = range(112, nrows-112, stepsize)
            cols = range(112, ncols-112, stepsize)

            # Run over all combinations of rows and columns
            with progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',
                                                  progressbar.Bar(), ' (', progressbar.ETA(), ') ', ],
                                         max_value=len(rows)*len(cols)) as bar:
                for ii, (row, col) in enumerate(itertools.product(rows, cols)):

                    td = data[row-112:row+112, col-112:col+112]

                    if display:
                        if len(td.shape) == 2:
                            ttdd = utils.gray2rgb(td)
                        else:
                            ttdd = td
                        ttdd = utils.rgb2plot(ttdd)

                        if imaxis is None:
                            print('displaying via imshow')
                            imaxis = plt.imshow(ttdd)
                        else:
                            print('displaying via set_data')
                            imaxis.set_data(ttdd)
                        plt.pause(0.0001)

                    predictions = self._fingerprint_calculator.calculate(td)

                    self._fingerprints.append(
                        {
                            'predictions': predictions,
                            'filename': filename,
                            'row_center': row,
                            'column_center': col
                        }
                    )

        if display:
            plt.close(fig)

        return self._fingerprints

    def save(self, output_directory):
        d = {
            'data_processing': [t.save() for t in self._data_processing],
            'fingerprint_calculator': self._fingerprint_calculator.save(),
            'uuid': self._uuid,
            'filenames': self._filenames,
            'fingerprints': self._fingerprints
        }

        with open(os.path.join(output_directory, 'data_{}.pck'.format(self._uuid)), 'wb') as fp:
            pickle.dump(d, fp)

    def _load(self, input_filename):
        log.info('Loading Data from {}'.format(input_filename))
        with open(input_filename, 'rb') as fp:
            tt = pickle.load(fp)

            self._data_processing = []
            for dp in tt['data_processing']:
                self._data_processing.append(DataProcessing.load_parameters(dp))
            self._fingerprint_calculator = Fingerprint.load_parameters(tt['fingerprint_calculator'])
            self._uuid = tt['uuid']
            self._filenames = tt['filenames']
            self._fingerprints = tt['fingerprints']

        log.debug('    data_processing is {}'.format(self._data_processing))
        log.debug('    fingerprint_calcualtor is {}'.format(self._fingerprint_calculator))
        log.debug('    uuid is {}'.format(self._uuid))
        log.debug('    filenames is {}'.format(self._filenames))

    @staticmethod
    def load(input_filename):
        d = Data()
        d._load(input_filename)
        return d
