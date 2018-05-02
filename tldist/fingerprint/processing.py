import uuid
import time
import numpy as np
import weakref
import requests
import imageio
from io import BytesIO

from transfer_learning.utils import gray2rgb

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("Fingerprint")
log.setLevel(logging.WARNING)

def calculate(data, fc_save):
    """
    Calculate the fingerprint from a list of data.  The data
    must be of the form 
         [ {'uuid': <somtehing>, 'location': <somewhere>, 'meta': {<meta data} }... ]
    """

    if not isinstance(data, list) and not isinstance(data[0], dict):
        log.error('Data must be a list of dictionaries')
        raise Exception('Data must be a list of dictionaries')

    # Load the fingerprint calculator based on dictionary information
    fc = Fingerprint.load_parameters(fc_save)

    # Now run through each datum and calculate the fingerprint
    fingerprints_return = []
    for ii, datum in enumerate(data):

        # Update the progress if we are using the task version of this.
        if hasattr(calculate, 'update_state'):
            calculate.update_state(state='PROGRESS', meta={'progress': ii})

        # Load the data
        if 'location' not in datum:
            log.error('Data does not have a location key {}'.format(datum))
            raise Exception('Data does not have a location key {}'.format(datum))

        response = requests.get(datum['location'])

        if not response.status_code == 200:
            log.error('Problem loading the data {}'.format(datum['location']))
            raise Exception('Problem loading the data {}'.format(datum['location']))

        nparray = np.array(imageio.imread(BytesIO(response.content)))

        # Calculate the predictions
        log.debug('calcuating predictions for  {} data is {}'.format(datum['location'], type(nparray)))
        try:
            predictions = fc.calculate(nparray[:224,:224])
        except:
            predictions = []

        # Clean the predictions so the json conversion is happy
        cleaned_predictions = [(x[0], x[1], float(x[2])) for x in predictions]

        # Load up the return list.
        fingerprints_return.append({
            'uuid': str(uuid.uuid4()), 
            'data_uuid': datum['uuid'], 
            'predictions': cleaned_predictions
            })

    return fingerprints_return


class Fingerprint:

    # What is currently happening is the Fingerprint calculator gets created for each TransferLearningProcessData
    # instance which is not good. What we want to do is create an instance if one does not exist with the
    # specified uuid. So the "load" for Fingerprint needs to be a little smarter to check to see if an instance
    # already exists for that uuid, and if it does, return it, otherwise create a new one.
    _instances = set()

    @classmethod
    def getinstances(cls, search_uuid=None):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None and (search_uuid is None or (search_uuid is not None and search_uuid == obj.uuid)):
                return obj
            elif obj is None:
                dead.add(ref)
        cls._instances -= dead
        return None

    def __init__(self):
        self._uuid = str(uuid.uuid4())
        self._predictions = []

        self._instances.add(weakref.ref(self))

    @property
    def uuid(self):
        return self._uuid

    def save(self, output_directory):
        raise NotImplementedError("Please Implement this method")

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
    def load_parameters(parameters):
        # First let's see if we have an instance with this UUID already created
        newinstance = Fingerprint.getinstances(parameters['uuid'])

        if newinstance is None:
            # If there is not an instance with that uuid, THEN we will create a new instance of that subclass
            for class_ in Fingerprint.__subclasses__():
                if class_.__name__ == parameters['class_name']:
                    tt = class_()
                    tt.load(parameters)
                    return tt
        else:
            return newinstance

    def load(self, parameters):
        """
        Load the instance information from the parameters

        :param parameters: dictionary that is of the format in the instances save() function
        :return: nothing...
        """

        self._uuid = parameters['uuid']


class FingerprintResnet(Fingerprint):

    def __init__(self, max_fingerprints=200):
        super(FingerprintResnet, self).__init__()

        from keras.applications.resnet50 import ResNet50

        # Load it when needed in calculate()
        log.debug('FingerprintResnet __init__')
        self._model = ResNet50(weights='imagenet')

        self._max_fingerprints = max_fingerprints

    def __str__(self):
        return 'Fingerprint (renet50, imagenet)'

    def calculate(self, data):
        from keras.applications.resnet50 import preprocess_input as preprocess_input
        from keras.applications.resnet50 import decode_predictions as decode_predictions

        log.debug('FingerprintResnet: calculate')

        # if self._model is None:
        #     self._model = ResNet50(weights='imagenet')

        start_time = time.time()

        # Set the data into the expected format
        if len(data.shape) < 3:
            x = gray2rgb(data)
        else:
            x = data.astype(np.float64)

        # Set the data into th expected format
        x = np.expand_dims(x, axis=0)

        # Do keras model image pre-processing
        x = preprocess_input(x)

        # There was an error at one point when the image was completely 0
        # In this case I just set a single prediction with low weight.
        # TODO:  Check to see if this is still an issue.
        if np.sum(np.abs(x)) > 0.0001:
            preds = self._model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=self._max_fingerprints)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions

    def save(self):
        return {
            'class_name': self.__class__.__name__,
            'uuid': self._uuid
        }



class FingerprintVGG16(Fingerprint):

    def __init__(self, max_fingerprints=200):
        super(FingerprintVGG16, self).__init__()

        from keras.applications.vgg16 import VGG16
        self._model = VGG16(weights='imagenet')

        self._max_fingerprints = max_fingerprints

    def __str__(self):
        return 'Fingerprint (vgg16, imagenet)'

    def calculate(self, data):
        from keras.applications.vgg16 import preprocess_input
        from keras.applications.vgg16 import decode_predictions

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
            preds = self._model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=self._max_fingerprints)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions

    def save(self):
        return {
            'class_name': self.__class__.__name__,
            'uuid': self._uuid
        }


class FingerprintVGG19(Fingerprint):

    def __init__(self, max_fingerprints=200):
        super(FingerprintVGG19, self).__init__()

        from keras.applications.vgg19 import VGG19
        self._model = VGG19(weights='imagenet')

        self._max_fingerprints = max_fingerprints

    def __str__(self):
        return 'Fingerprint (vgg19, imagenet)'

    def calculate(self, data):
        from keras.applications.vgg19 import preprocess_input
        from keras.applications.vgg19 import decode_predictions

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
            preds = self._model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=self._max_fingerprints)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions

    def save(self):
        return {
            'class_name': self.__class__.__name__,
            'uuid': self._uuid
        }


class FingerprintInceptionV3(Fingerprint):

    def __init__(self, max_fingerprints=200):
        super(FingerprintInceptionV3, self).__init__()

        from keras.applications.inception_v3 import InceptionV3
        self._model = InceptionV3(weights='imagenet')

        self._max_fingerprints = max_fingerprints

    def __str__(self):
        return 'Fingerprint (inception_v3, imagenet)'

    def calculate(self, data):
        from keras.applications.inception_v3 import preprocess_input
        from keras.applications.inception_v3 import decode_predictions

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
            preds = self._model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=self._max_fingerprints)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions

    def save(self):
        return {
            'class_name': self.__class__.__name__,
            'uuid': self._uuid
        }



class FingerprintInceptionResNetV2(Fingerprint):

    def __init__(self, max_fingerprints):
        super(FingerprintInceptionResNetV2, self).__init__()

        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        self._model = InceptionResNetV2(weights='imagenet')

        self._max_fingerprints = max_fingerprints

    def __str__(self):
        return 'Fingerprint (inception_resnet_v2, imagenet)'

    def calculate(self, data):
        from keras.applications.inception_resnet_v2 import preprocess_input
        from keras.applications.inception_resnet_v2 import decode_predictions

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
            preds = self._model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            self._predictions = decode_predictions(preds, top=self._max_fingerprints)[0]
        else:
            self._predictions = [('test', 'beaver', 0.0000000000001), ]

        end_time = time.time()
        log.debug('Predictions: {}'.format(self._predictions[:10]))
        log.info('Calculate {} predictions took {}s'.format(len(self._predictions), end_time - start_time))

        return self._predictions

    def save(self):
        return {
            'class_name': self.__class__.__name__,
            'uuid': self._uuid
        }
