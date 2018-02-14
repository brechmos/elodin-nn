import uuid
import time
import numpy as np

from utils import gray2rgb

from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.resnet50 import decode_predictions as resnet50_decode_predictions
from keras.applications.resnet50 import ResNet50
resnet50_model = ResNet50(weights='imagenet')

from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg16 import decode_predictions as vgg16_decode_predictions
from keras.applications.vgg16 import VGG16
vgg16_model = VGG16(weights='imagenet')

from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from keras.applications.vgg19 import decode_predictions as vgg19_decode_predictions
from keras.applications.vgg19 import VGG19
vgg19_model = VGG19(weights='imagenet')

from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from keras.applications.inception_v3 import decode_predictions as inception_v3_decode_predictions
from keras.applications.inception_v3 import InceptionV3
inception_v3_model = InceptionV3(weights='imagenet')

from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_v2_preprocess_input
from keras.applications.inception_resnet_v2 import decode_predictions as inception_resnet_v2_decode_predictions
from keras.applications.inception_resnet_v2 import InceptionResNetV2
inception_resnet_v2_model = InceptionResNetV2(weights='imagenet')

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("Fingerprint")
log.setLevel(logging.INFO)


class Fingerprint:
    def __init__(self):
        self._uuid = str(uuid.uuid4())
        self._predictions = []

    def save(self, output_directory):
        pass

    @staticmethod
    def select_fingerprint():
        """
        The function will display all the fingerprinting methods and return a class
        of the one of interest.  This can then be used to construct the class and
        to use in further calculations.

        :return:  selected class
        """
        subclasses = Fingerprint.__subclasses__()

        selected = False
        N = 0
        while not selected:
            # Show the fingerprints in order to allow for the person to select one
            print('Select a pre-trained network to use (q to quit:')
            for ii, x in enumerate(subclasses):
                print('   {}) {}'.format(ii, x.__str__()))
                N = ii

            s = input('Select Number > ')

            if s == 'q':
                return

            try:
                s = int(s)
                if s >= 0 and s < N:
                    # create fingerprint
                    return subclasses[s]
            except:
                pass

    @staticmethod
    def load(parameters):
        if parameters['class_name'] == 'FingerprintResnet':
            return FingerprintResnet(parameters)
        elif parameters['class_name'] == 'FingerprintVGG16':
            return FingerprintVGG16(parameters)
        elif parameters['class_name'] == 'FingerprintVGG19':
            return FingerprintVGG19(parameters)
        elif parameters['class_name'] == 'FingerprintInceptionV3':
            return FingerprintInceptionV3(parameters)
        elif parameters['class_name'] == 'FingerprintInceptionResnetV2':
            return FingerprintInceptionResNetV2(parameters)
        else:
            raise ValueError('Unknown class name {}'.format(parameters['class_name']))


class FingerprintResnet(Fingerprint):

    def __init__(self):
        super(FingerprintResnet, self).__init__()

    def __str__(self):
        return 'Fingerprint (renet50, imagenet)'

    def calculate(self, data):

        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
        else:
            x = data.astype(np.float64)

        # Set the data into the expected format
        x = np.expand_dims(x, axis=0)

        # Do keras model image pre-processing
        x = resnet50_preprocess_input(x)

        # There was an error at one point when the image was completely 0
        # In this case I just set a single prediction with low weight.
        # TODO:  Check to see if this is still an issue.
        if np.sum(np.abs(x)) > 0.0001:
            preds = resnet50_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = resnet50_decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions


class FingerprintVGG16(Fingerprint):

    def __init__(self):
        super(FingerprintVGG16, self).__init__()

    def __str__(self):
        return 'Fingerprint (vgg16, imagenet)'

    def calculate(self, data):
        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
        else:
            x = data.astype(np.float64)

        # Set the data into the expected format
        x = np.expand_dims(x, axis=0)

        # Do keras model image pre-processing
        x = vgg16_preprocess_input(x)

        # There was an error at one point when the image was completely 0
        # In this case I just set a single prediction with low weight.
        # TODO:  Check to see if this is still an issue.
        if np.sum(np.abs(x)) > 0.0001:
            preds = vgg16_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = vgg16_decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions


class FingerprintVGG19(Fingerprint):

    def __init__(self):
        super(FingerprintVGG19, self).__init__()

    def __str__(self):
        return 'Fingerprint (vgg19, imagenet)'

    def calculate(self, data):
        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
        else:
            x = data.astype(np.float64)

        # Set the data into the expected format
        x = np.expand_dims(x, axis=0)

        # Do keras model image pre-processing
        x = vgg19_preprocess_input(x)

        # There was an error at one point when the image was completely 0
        # In this case I just set a single prediction with low weight.
        # TODO:  Check to see if this is still an issue.
        if np.sum(np.abs(x)) > 0.0001:
            preds = vgg19_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = vgg19_decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions


class FingerprintInceptionV3(Fingerprint):

    def __init__(self):
        super(FingerprintInceptionV3, self).__init__()

    def __str__(self):
        return 'Fingerprint (inception_v3, imagenet)'

    def calculate(self, data):
        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
        else:
            x = data.astype(np.float64)

        # Set the data into the expected format
        x = np.expand_dims(x, axis=0)

        # Do keras model image pre-processing
        x = inception_v3_preprocess_input(x)

        # There was an error at one point when the image was completely 0
        # In this case I just set a single prediction with low weight.
        # TODO:  Check to see if this is still an issue.
        if np.sum(np.abs(x)) > 0.0001:
            preds = inception_v3_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = inception_v3_decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions


class FingerprintInceptionResNetV2(Fingerprint):

    def __init__(self):
        super(FingerprintInceptionResNetV2, self).__init__()

    def __str__(self):
        return 'Fingerprint (inception_resnet_v2, imagenet)'

    def calculate(self, data):
        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
        else:
            x = data.astype(np.float64)

        # Set the data into the expected format
        x = np.expand_dims(x, axis=0)

        # Do keras model image pre-processing
        x = inception_resnet_v2_preprocess_input(x)

        # There was an error at one point when the image was completely 0
        # In this case I just set a single prediction with low weight.
        # TODO:  Check to see if this is still an issue.
        if np.sum(np.abs(x)) > 0.0001:
            preds = inception_resnet_v2_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = inception_resnet_v2_decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions

