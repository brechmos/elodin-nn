import pickle
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')
model_name = 'resnet50'

import logging
logging.basicConfig(format='%(levelname)s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("Fingerprint")
log.setLevel(logging.INFO)


class Fingerprint:
    """
    The Fingerprint class represents one set of image data and the associated fingerprint.
    It will need to be sub-classed in order to change the create_subdata function.

    """

    def __init__(self, data):

        self._row_center = None
        self._column_center = None
        self._predictions = []
        self._data = data

    def __repr__(self):
        return '{}: ({}, {})'.format(self._data, self._row_center, self._column_center)

    @property
    def data(self):
        return self._data

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

    def _preprocess(self, data):
        """
        This should be overloaded if other preprocessing is wanted (e.g., median filtering or smoothing)

        :param data:
        :return:
        """

        return data

    def create_subdata(self, row_center, column_center):
        """
        This should be overloaded if functionality other than just normal cutout is wanted (e.g., sub/super sampling.

        :param row_center:
        :param column_center:
        :return:
        """

        # Must return a data object (probably 224 x 224)
        return None

    def calculate(self, row_center, column_center):
        """
        Create the subdata and then run through keras

        :param row_center:
        :param column_center:
        :return:
        """

        # Must fill self._predictions
        pass

    @property
    def predictions(self):
        return self._predictions

    def save(self, directory=None):
        """
        The things that need to be saved for this finger print:
           input data file
           location of the center of the cutout
           the specific instance of Fingerprint this was created from

        :param directory:
        :return:
        """

        # Make sure the directory exists
        # TODO: will need to check on the path of directories
        if not os.path.isdir(directory):
            os.mkdir(directory)

        filename = os.path.join(directory, 'filename_{:04}_{:04}.pck'.format(self._row_center, self._column_center))

        blah = {
            'filename': self._data.filename,
            'fingerprint': self.__class__,
            'data': self._data,
            'middle': [self._row_center, self._column_center],
            'predictions': self._predictions
        }

        log.info('Saving to {}'.format(filename))
        pickle.dump(blah, open(filename, 'wb'))

    def load(filename):
        """
        Static loader which will determine the type based on the contents of the file.

        :return:
        """
        with open(filename, 'rb') as fp:
            tt = pickle.load(fp)
            fptype = tt['fingerprint']

        # Create the instance and load the data into the instance.
        log.debug('Going to load data into {}'.format(fptype))
        tt = fptype(None)
        tt._load(filename)

        return tt

    def _load(self, filename):
        """

        :param filename:
        :return:
        """

        log.info('Loading from file {}'.format(filename))
        with open(filename, 'rb') as fp:
            tt = pickle.load(fp)
            self._row_center, self._column_center = tt['middle']
            self._predictions = tt['predictions']
            self._data = tt['data']

    def display(self, axis, title=''):
        print('{} {} {} {}'.format( self._data._input_data.shape,
            self._row_center, self._column_center,
            self._data[self._row_center-112:self._row_center+112, self._column_center-112:self._column_center+112].shape))
        axis.imshow(
            self._data[self._row_center-112:self._row_center+112, self._column_center-112:self._column_center+112]
        )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

        #clow, chigh = np.percentile(self._data[np.isfinite(self._data)], (1, 99))
        #axis.get_images()[0].set_clim((clow, chigh))


class ResnetFingerprint(Fingerprint):

    def __init__(self, data):
        super( ResnetFingerprint, self ).__init__(data)

    def create_subdata(self, row_center, column_center):
        """
        This should be overloaded if functionality other than just normal cutout is wanted (e.g., sub/super sampling.

        :param row_center:
        :param column_center:
        :return:
        """

        if not self._data:
            self._load_data()
            self._data = self._preprocess(self._input_data)

        self._row_center = row_center
        self._column_center = column_center

        log.info('Create sub data {} with middle {} {} '.format(self._data[row_center-112:row_center+112, column_center-112:column_center+112].shape, row_center, column_center))
        return self._data[row_center-112:row_center+112, column_center-112:column_center+112]

    def calculate(self, row_center, column_center):
        """
        Create the subdata and then run through keras

        :param row_center:
        :param column_center:
        :return:
        """

        data_orig = self.create_subdata(row_center, column_center)

        start_time = time()

        # Set the data into the expected format
        if len(data_orig.shape) < 3:
            x = Fingerprint._gray2rgb(data_orig)
        else:
            x = data_orig.astype(np.float64)

        # Set the data into the expected format
        x = np.expand_dims(x, axis=0)

        # Do keras model image pre-processing
        x = preprocess_input(x)

        # There was an error at one point when the image was completely 0
        # In this case I just set a single prediction with low weight.
        # TODO:  Check to see if this is still an issue.
        if np.sum(np.abs(x)) > 0.0001:
            preds = model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))
