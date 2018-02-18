from similarity import tSNE, Jaccard
import glob
import uuid
import numpy as np
import pickle
import imageio
from astropy.io import fits
import progressbar
import itertools
import os
import sys

import matplotlib.pyplot as plt

from data_processing import DataProcessing
from fingerprint import Fingerprint
import utils

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("TransferLearning")
log.setLevel(logging.INFO)


class TransferLearning:
    def __init__(self, fingerprint_calculator=None, data_processing=[]):
        # list of data processing elements
        self._data_processing = data_processing
        self._fingerprint_calculator = fingerprint_calculator
        self._filenames = []
        self._fingerprints = []
        self._uuid = str(uuid.uuid4())

        # number of pixels around the image to include in the display
        self._image_display_margin = 50

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
            log.info('Caching image data {}...'.format(filename))
            self._data_cache[filename] = self._load_image_data(filename)

        # print(filename, self._data_cache[filename].shape)
        # nrows, ncols = self._data_cache[filename].shape[:2]
        #
        # start_row = max(0, row - 112 - self._image_display_margin)
        # end_row = min(nrows, row + 112 + self._image_display_margin)
        # start_col = max(0, col - 112 - self._image_display_margin)
        # end_col = min(ncols, col + 112 + self._image_display_margin)

        return self._data_cache[filename][row-112:row+112, col-112:col+112]

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
        self._stepsize = stepsize

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
                            imaxis = plt.imshow(ttdd)
                        else:
                            imaxis.set_data(ttdd)
                        plt.pause(0.001)

                    predictions = self._fingerprint_calculator.calculate(td)

                    self._fingerprints.append(
                        {
                            'data': self,
                            'predictions': predictions,
                            'filename': filename,
                            'row_center': row,
                            'column_center': col
                        }
                    )

        if display:
            plt.close(fig)

        return self._fingerprints

    def save(self, output_location):
        """
        Save the information to the output location. If the output_location is a directory then
        save as a standard name. If output_location is a filename then save as it.

        :param output_location: Filename or directory of where to save
        :return:
        """

        # Delete the reference to the TransferLearning instance. This will be
        # added back in on load.
        fingerprints =  []
        for item in self._fingerprints:
            tt = item.copy()
            del tt['data']
            fingerprints.append(tt)

        # Create dictionary to save
        d = {
            'data_processing': [t.save() for t in self._data_processing],
            'fingerprint_calculator': self._fingerprint_calculator.save(),
            'stepsize': self._stepsize,
            'uuid': self._uuid,
            'filenames': self._filenames,
            'fingerprints': fingerprints
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
            tt = pickle.load(fp)

            self._data_processing = []
            for dp in tt['data_processing']:
                self._data_processing.append(DataProcessing.load_parameters(dp))
            self._fingerprint_calculator = Fingerprint.load_parameters(tt['fingerprint_calculator'])
            self._uuid = tt['uuid']
            self._stepsize = tt['stepsize']
            self._filenames = tt['filenames']
            self._fingerprints = tt['fingerprints']

            # Save the reference to this transfer learning instance as it is needed when
            # we lookup data later.
            for item in self._fingerprints:
                item.update({'data': self})

        log.debug('    data_processing is {}'.format(self._data_processing))
        log.debug('    fingerprint_calcualtor is {}'.format(self._fingerprint_calculator))
        log.debug('    uuid is {}'.format(self._uuid))
        log.debug('    filenames is {}'.format(self._filenames))

        # Pre cache the image data for faster response time
        for fn in self._filenames:
            if fn not in self._data_cache:
                print('Caching image data {}...'.format(fn))
                self._data_cache[fn] = self._load_image_data(fn)


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


class TransferLearningDisplay:
    def __init__(self, similarity_measure):
        self.similarity = similarity_measure
        self.fig = None
        self.axis = None
        self.info_axis = None
        self.info_text = None
        self.sub_windows = None

    def show(self, fingerprints):
        plt.show(block=False)
        plt.ion()

        self.similarity.calculate(fingerprints)

        self.fig = plt.figure(1, figsize=[10, 6])
        plt.gcf()
        self.axis = plt.axes([0.05, 0.05, 0.45, 0.45])

        self.axis_closest = plt.axes([0.5, 0.01, 0.2, 0.2])
        self.axis_closest.set_xticks([])
        self.axis_closest.set_yticks([])
        self.axis_closest.set_xlabel('')
        self.axis_closest.set_ylabel('')
        self._data_closest = self.axis_closest.imshow(np.zeros((224, 224)))

        self.similarity.display(self.axis)

        self.info_axis = plt.axes([0.7, 0.11, 0.3, 0.05])
        self.info_axis.set_axis_off()
        self.info_axis.set_xticks([])
        self.info_axis.set_yticks([])
        self.info_axis.set_xlabel('')
        self.info_axis.set_ylabel('')
        self.info_text = self.info_axis.text(0, 0, 'Loading...', fontsize=12)

        self._cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._onmove)

        self.sub_windows = []
        self.sub_data = []
        for row in range(3):
            for col in range(3):
                # rect = [left, bottom, width, height]
                tt = plt.axes([0.5 + 0.14 * col, 0.75 - 0.25 * row, 0.2, 0.2])
                tt.set_xticks([])
                tt.set_yticks([])
                sd = tt.imshow(np.zeros((224, 224)))
                self.sub_windows.append(tt)
                self.sub_data.append(sd)
        plt.show(block=False)

    def _update_text(self, thetext):
        print('updateing text to {}'.format(thetext))
        self.info_text.set_text(thetext)
        plt.draw()

    def _onmove(self, event):
        log.debug('Moving to {}'.format(event))
        if event.inaxes == self.axis:
            point = event.ydata, event.xdata
            close_fingerprint = self.similarity.find_similar(point, n=1)[0][1]

            log.debug('Closest fingerprints {}'.format(close_fingerprint))

            self._data_closest.set_data(utils.rgb2plot(
                close_fingerprint['data'].display(close_fingerprint['filename'],
                                                  close_fingerprint['row_center'],
                                                  close_fingerprint['column_center'])
            ))
            self.axis_closest.set_title(close_fingerprint['filename'].split('/')[-1], fontsize=8)
            self.fig.canvas.blit(self.axis_closest.bbox)
            self.axis_closest.redraw_in_frame()

    def _onclick(self, event):
        log.debug('Clicked {}'.format(event))
        import time
        if event.inaxes == self.axis:
            point = event.ydata, event.xdata
#            self.axis.cla()
#            self.similarity.display(self.axis)

            log.debug('Loading data')

            self._update_text('Loading data...')
            close_fingerprints = self.similarity.find_similar(point)

            self._update_text('Displaying result...')

            for ii, (distance, fingerprint) in enumerate(close_fingerprints):

                # Zero out and show we are loading -- should be fast.3
                self.sub_windows[ii].set_title('Loading...', fontsize=8)
                self.sub_data[ii].set_data(np.zeros((224, 224)))
                self.sub_windows[ii].redraw_in_frame()

                # Show new data and set title
                self.sub_data[ii].set_data(utils.rgb2plot(
                    fingerprint['data'].display(fingerprint['filename'],
                                                fingerprint['row_center'],
                                                fingerprint['column_center'])
                ))

                self.sub_windows[ii].set_title('{:0.3f} {} ({}, {})'.format(
                    distance,
                    os.path.basename(fingerprint['filename']),
                    fingerprint['row_center'],
                    fingerprint['column_center']), fontsize=8)
                self.sub_windows[ii].redraw_in_frame()

            self._update_text('Click in the tSNE plot...')

            log.debug('Done the onlcick')

    def _display_for_subwindow(self, index, aa):
        print('index is {} and aa is {}'.format(index, aa))
        distance, fingerprint = aa

        # Zero out and show we are loading -- should be fast.3
        log.debug('Displaying fingerprint {}'.format(index))
        self.sub_windows[index].set_title('Loading...', fontsize=8)
        self.sub_data[index].set_data(np.zeros((224, 224)))
        self.sub_windows[index].redraw_in_frame()

        # Show new data and set title
        self.sub_data[index].set_data(utils.rgb2plot(
            fingerprint['data'].display(fingerprint['filename'],
                                        fingerprint['row_center'],
                                        fingerprint['column_center'])
        ))
        self.sub_windows[index].set_title('{:0.3f} {} ({}, {})'.format(
            distance,
            os.path.basename(fingerprint['filename']),
            fingerprint['row_center'],
            fingerprint['column_center']), fontsize=8)

        self.sub_windows[index].redraw_in_frame()


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
