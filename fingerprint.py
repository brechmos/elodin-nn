import uuid
import time
import numpy as np

from utils import gray2rgb

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


class FingerprintResnet(Fingerprint):

    def __init__(self):
        super(FingerprintResnet, self).__init__()

        from keras.applications.resnet50 import ResNet50
        self._resnet50_model = ResNet50(weights='imagenet')

    def __repr__(self):
        return 'Fingerprint (renet50, imagenet)'

    def calculate(self, data):
        from keras.applications.resnet50 import preprocess_input, decode_predictions

        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
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
            preds = self._resnet50_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions


class FingerprintVGG16(Fingerprint):

    def __init__(self):
        super(FingerprintVGG16, self).__init__()

        from keras.applications.vgg16 import VGG16
        self._vgg16_model = VGG16(weights='imagenet')

    def __repr__(self):
        return 'Fingerprint (vgg16, imagenet)'

    def calculate(self, data):
        from keras.applications.vgg16 import preprocess_input, decode_predictions

        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
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
            preds = self._vgg16_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions


class FingerprintVGG19(Fingerprint):

    def __init__(self):
        super(FingerprintVGG19, self).__init__()

        from keras.applications.vgg19 import VGG19
        self._vgg19_model = VGG19(weights='imagenet')

    def __repr__(self):
        return 'Fingerprint (vgg19, imagenet)'

    def calculate(self, data):
        from keras.applications.vgg19 import preprocess_input, decode_predictions

        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
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
            preds = self._vgg19_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions


class FingerprintInceptionV3(Fingerprint):

    def __init__(self):
        super(FingerprintInceptionV3, self).__init__()

        from keras.applications.inception_v3 import InceptionV3
        self._inception_v3_model = InceptionV3(weights='imagenet')

    def __repr__(self):
        return 'Fingerprint (inception_v3, imagenet)'

    def calculate(self, data):
        from keras.applications.inception_v3 import preprocess_input, decode_predictions

        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
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
            preds = self._inception_v3_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions


class FingerprintInceptionResNetV2(Fingerprint):

    def __init__(self):
        super(FingerprintInceptionResNetV2, self).__init__()

        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        self._inception_resnet_v2_model = InceptionResNetV2(weights='imagenet')

    def __repr__(self):
        return 'Fingerprint (inception_resnet_v2, imagenet)'

    def calculate(self, data):
        from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
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
            preds = self._inception_resnet_v2_model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=200)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions

