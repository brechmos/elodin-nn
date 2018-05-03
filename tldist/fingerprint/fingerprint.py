import uuid

class Fingerprint:

    @staticmethod
    def fingerprint_factory(thedict):
        return Fingerprint(data_uuid=thedict['data_uuid'], predictions=thedict['predictions'])

    def __init__(self, data_uuid=None, predictions=[]):
        self._uuid = str(uuid.uuid4())
        self._data_uuid = data_uuid
        self._predictions = predictions

    def __str__(self):
        return 'Fingerprint {} based on data {} with predictions {}'.format(
                self.uuid, self.data_uuid, [x for x in self.predictions[:3]])

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = value

    @property
    def data_uuid(self):
        return self._data_uuid

    @data_uuid.setter
    def data_uuid(self, value):
        self._data_uuid = value

    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, value):
        self._predictions = value

    def load(self, thedict):
        self.uuid = thedict['uuid']
        self.data_uuid = thedict['data_uuid']
        self.predictions = thedict['predictions']

    def save(self):
        return {
             'uuid': self.uuid,
             'data_uuid': self.data_uuid,
             'predictions': [(x[0], x[1], float(x[2])) for x in self.predictions]
        }
