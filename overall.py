import uuid
import glob
import time
import numpy as np

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("TransferLearning")
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------

import pickle
import imageio
from astropy.io import fits
import progressbar
import itertools
import os

class Data:
    def __init__(self, fingerprint_calculator=None, data_processing=[]):
        # list of data processing elements
        self._data_processing = data_processing
        self._fingerprint_calculator = fingerprint_calculator
        self._filenames = []
        self._fingerprints = []
        self._uuid = str(uuid.uuid4())

        self._data_cache = {}

    def _gray2rgb(self, data):
        """
        Convert 2D data set to 3D gray scale

        :param data:
        :return:
        """
        data_out = np.zeros((data.shape[0], data.shape[1], 3))
        data_out[:, :, 0] = data
        data_out[:, :, 1] = data
        data_out[:, :, 2] = data

        return data_out

    def _load_image_data(self, filename):
        if filename.endswith('tiff'):
            log.debug('Loading TIFF file {}'.format(filename))
            data = np.array(imageio.imread(filename))
        elif 'fits' in filename:
            log.debug('Loading FITS file {}'.format(filename))
            data = fits.open(filename)[1].data
            log.debug('FITS data is {}'.format(data))

            data[~np.isfinite(data)] = 0

        # Make RGB (3 channel) if only gray scale (single channel)
        if len(data.shape) == 2:
            data = self._gray2rgb(data)

        # Do the pre-processing of the data
        for dp in self._data_processing:
            log.debug('Doing pre-processing {}, input data shape {}'.format(dp, data.shape))
            data = dp.process(data)
            log.debug('    Now input data shape {}'.format(data.shape))

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

    def calculate(self, stepsize):
        # calculate the fingerprints

        self._fingerprints = []
        for filename in self._filenames:
            log.info('Processing filename {}'.format(filename))

            data = self._load_image_data(filename)

            # Calculate predictions for each sub-area
            nrows, ncols = data.shape[:2]

            rows = range(112, nrows-112, stepsize)
            cols = range(112, ncols-112, stepsize)

            # Run over all combinations of rows and columns
            with progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',
                                                  progressbar.Bar(), ' (', progressbar.ETA(), ') ', ],
                                         max_value=len(rows)*len(cols)) as bar:
                for ii, (row, col) in enumerate(itertools.product(rows, cols)):

                    td = data[row-112:row+112, col-112:col+112]

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

            return self._fingerprints

    def save(self, output_directory):
        d = {
            'data_processing': self._data_processing,
            'fingerprint_calculator': self._fingerprint_calculator,
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

            self._data_processing = tt['data_processing']
            self._fingerprint_calculator = tt['fingerprint_calculator']
            self._uuid = tt['uuid']
            self._filenames = tt['filenames']
            self._fingerprints = tt['fingerprints']

        log.debug('    data_processing is {}'.format(self._data_processing))
        log.debug('    fingerprint_calcualtor is {}'.format(self._fingerprint_calculator))
        log.debug('    uuid is {}'.format(self._uuid))
        log.debug('    filenames is {}'.format(self._filenames))

    def load(input_filename):
        d = Data()
        d._load(input_filename)
        return d

# ---------------------------------------------------------------------------


class DataProcessing:
    def __init__(self):
        pass

import scipy.ndimage
class ZoomData(DataProcessing):

    def __init__(self, zoom_level=1):
        self._zoom_level = zoom_level

    def __repr__(self):
        return 'ZoomData (level {})'.format(self._zoom_level)

    def process(self, input_data):

        if len(input_data.shape) == 2:
            return scipy.ndimage.zoom(input_data, self._zoom_level)
        else:
            output_data = None
            for ii in range(input_data.shape[2]):
                od = scipy.ndimage.zoom(input_data[:, :, ii], self._zoom_level)
                if output_data is None:
                    output_data = np.zeros((od.shape[0], od.shape[1], 3))
                output_data[:, :, ii] = od
            return output_data

import scipy.ndimage.filters
class MedianFilterData(DataProcessing):
    def __init__(self, kernel_size):
        self._kernel_size = kernel_size

    def __repr__(self):
        return 'MedianFilterData (kernel size {})'.format(self._kernel_size)

    def process(self, input_data):
        return scipy.ndimage.filters.median_filter(input_data, size=self._kernel_size)

# ---------------------------------------------------------------------------


class Fingerprint:
    def __init__(self):
        self._uuid = str(uuid.uuid4())
        self._predictions = []

    def save(self, output_directory):
        pass

    def _gray2rgb(data):
        """
        Convert 2D data set to 3D gray scale

        :param data:
        :return:
        """
        data_out = np.zeros((224, 224, 3))
        data_out[:, :, 0] = data
        data_out[:, :, 1] = data
        data_out[:, :, 2] = data

        return data_out



from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

resnet50_model = ResNet50(weights='imagenet')


class FingerprintResnet(Fingerprint):
    def __init__(self):
        super(FingerprintResnet, self).__init__()

    def __repr__(self):
        return 'Fingerprint (renet50, imagenet)'

    def calculate(self, data):
        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = Fingerprint._gray2rgb(data)
        else:
            x = data.astype(np.float64)

        # Set the data into the expected format
        x = np.expand_dims(x, axis=0)

        # Do keras model image pre-processing
        x = preprocess_input(x)

        # There was an error at one point when the image was completely 0
        # In this case I just set a single prediction with low weight.
        # TODO:  Check to see if this is still an issue.
        if np.sum(np.abs(x)) > 0.0001:
            preds = resnet50_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions

# ---------------------------------------------------------------------------

from sklearn.manifold import TSNE
# https://stackoverflow.com/questions/456672/class-factory-in-python
class Similarity:
    _similarity_type = ''

    def __init__(self, similarity_type, *args, **kwargs):
        self.similarity_type = similarity_type

    def _get(domain):
        for cls in Similarity.__subclasses__():
            if cls.is_similarity_for(domain):
                return cls()
        raise ValueError

    def calculate(self, files):
        pass

    def display(self):
        pass

    def save(self):
        pass

class tSNE(Similarity):

    _similarity_type = 'tsne'

    def __init__(self, *args, **kwargs):
        super(tSNE, self).__init__(tSNE._similarity_type, *args, **kwargs)

        log.info('Created {}'.format(__class__._similarity_type))

        # Each line / element in these should correpsond
        self._Y = None
        self._fingerprints = []

    @classmethod
    def is_similarity_for(cls, similarity_type):
        return cls._similarity_type == similarity_type

    def calculate(self, fingerprints):
        log.info('Going to calculate tSNE from {} fingerprints'.format(len(fingerprints)))

        # Calculate the unique labels
        labels = []
        values = {}
        for ii, fp in enumerate(fingerprints):
            log.debug('    fingerprint is {}'.format(fp))
            # Add to unique label list
            labels.extend([pred[1] for pred in fp['predictions'] if pred[1] not in labels])

            # Store the predictions for next processing
            values[ii] = fp['predictions']

            self._fingerprints.append(fp)
        log.info('Unique labels {}'.format(labels))

        X = np.zeros((len(fingerprints), len(labels)))
        for ii, fp in enumerate(fingerprints):
            inds = [labels.index(prediction[1]) for prediction in values[ii]]
            X[ii][inds] = [prediction[2] for prediction in values[ii]]

        log.debug('X is {}'.format(X))
        log.debug('Fingerprint list {}'.format(self._fingerprints))

        # Compute the tSNE of the data.
        log.info('Calculating the tSNE...')
        self._Y = TSNE(n_components=2).fit_transform(X)
        log.debug('self._Y is {}'.format(self._Y))
        log.info('Done calculation')

    def displayY(self, axes):
        """
        Display the Y values in an axis element.

        :param axes:
        :param args:
        :param kwargs:
        :return:
        """
        axes.plot(self._Y[:, 0], self._Y[:, 1], '.')

    def find_similar(self, point, n=9):

        distances = np.sqrt(np.sum((self._Y - np.array(point)) ** 2, axis=1))
        inds = np.argsort(distances)
        log.debug('Closest indexes are {}'.format(inds[:n]))
        log.debug('Size of the fingerprint list {}'.format(len(self._fingerprints)))

        #log.info('Most similar finger prints {}'.format([self._fingerprints[ind] for ind in inds[:n]]))

        toreturn = []
        for ind in inds[:n]:
            toreturn.append([distances[ind], self._fingerprints[ind]])

        return toreturn


# ---------------------------------------------------------------------------


class TransferLearning:
    def __init__(self):
        pass

    def save(self, output_directory):
        pass

    def display(self, fingerprints):
        pass


def rgb2plot(data):
    """
    Convert the input data to RGB. This is basically clipping and cropping the intensity range for display

    :param data:
    :return:
    """

    mindata, maxdata = np.percentile(data[np.isfinite(data)], (0.01, 99.0))
    return np.clip((data - mindata) / (maxdata-mindata) * 255, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # input_file_pattern = '/Users/crjones/christmas/hubble/Carina/data/carina.tiff'
    # output_directory = '/tmp/resnet_acs/'

    input_file_pattern = '/Users/crjones/christmas/hubble/ACS_Halpha/data/*/*fits.gz'
    output_directory = '/tmp/acs_halpha'

    input_files = glob.glob(input_file_pattern)

    stepsize = 112

    calculate = True
    if calculate:
        tl = TransferLearning()

        fingerprints = []

        fingerprint_resnet = FingerprintResnet()

        # # calculate fingerpirnts for median filtered
        log.info('Setting up median filter data')
        data_processing = [MedianFilterData((5,5,1))]
        data = Data(fingerprint_resnet, data_processing)
        data.set_files(input_files)
        fingerprints = data.calculate(stepsize=stepsize)
        data.save(output_directory)

        # calculate fingerprints for median filtered and sub-sampled
        log.info('Setting up median filter, zoom 2 data')
        data_processing = [MedianFilterData((5,5,1)), ZoomData(2)]
        data_median_supersample = Data(fingerprint_resnet, data_processing)
        data_median_supersample.set_files(input_files)
        fingerprints_median_supersample = data_median_supersample.calculate(stepsize=stepsize)
        data_median_supersample.save(output_directory)

        # calcualte finterprints for median fitlered and super-sampled
        log.info('Setting up median filter, zoom 0.5 data')
        data_processing = [MedianFilterData((5,5,1)), ZoomData(0.5)]
        data_median_subsample = Data(fingerprint_resnet, data_processing)
        data_median_subsample.set_files(input_files)
        fingerprints_median_subsample = data_median_subsample.calculate(stepsize=stepsize)
        data_median_subsample.save(output_directory)

        data = [data, data_median_supersample, data_median_subsample]
        fingerprints = [fingerprints, fingerprints_median_supersample, fingerprints_median_subsample]

        tl.save(output_directory)

    else:
        files = glob.glob(os.path.join(output_directory, 'data_*pck'))

        fingerprints = []
        for file in files:

            data = Data.load(file)
            log.info('Loaded data {}'.format(data))

            fingerprints.extend(data.fingerprints)

        tsne_similarity = tSNE(fingerprints)
        tsne_similarity.calculate(fingerprints)

        plt.figure(1)
        plt.clf()
        axis = plt.axes([0.05, 0.05, 0.45, 0.45])

        sub_windows = []
        for row in range(3):
            for col in range(3):
                # rect = [left, bottom, width, height]
                tt = plt.axes([0.5 + 0.17 * col, 0.75 - 0.25 * row, 0.15, 0.15])
                tt.set_xticks([])
                tt.set_yticks([])
                sub_windows.append(tt)

        while True:

            axis.cla()
            tsne_similarity.displayY(axis)

            point = plt.ginput(1)

            if point:
                close_fingerprints = tsne_similarity.find_similar(point)
                for ii, (distance, fingerprint) in enumerate(close_fingerprints):
                    print(fingerprint)
                    sub_windows[ii].imshow( rgb2plot(
                        fingerprint['data'].display(fingerprint['filename'],
                                                    fingerprint['row_center'],
                                                    fingerprint['column_center'])
                    ))
                    sub_windows[ii].set_title('{} ({}, {})'.format(
                        os.path.basename(fingerprint['filename']),
                        fingerprint['row_center'],
                        fingerprint['column_center']), fontsize=8)
            else:
                break


    #tl.display(fingerprints)