import uuid
import time
import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

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
