import uuid

class Fingerprint:

    @staticmethod
    def fingerprint_factory(thedict):
        return Fingerprint(cutout_uuid=thedict['cutout_uuid'], predictions=thedict['predictions'])

    def __init__(self, cutout_uuid=None, predictions=[]):
        self._uuid = str(uuid.uuid4())
        self._cutout_uuid = cutout_uuid
        self._predictions = predictions

    def __str__(self):
        return 'Fingerprint {} based on cutout {} with predictions {}'.format(
                self.uuid, self.cutout_uuid, [x for x in self.predictions[:3]])

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = value

    @property
    def cutout_uuid(self):
        return self._cutout_uuid

    @cutout_uuid.setter
    def cutout_uuid(self, value):
        self._cutout_uuid = value

    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, value):
        self._predictions = value

    def load(self, thedict):
        self.uuid = thedict['uuid']
        self.cutout_uuid = thedict['cutout_uuid']
        self.predictions = thedict['predictions']

    def save(self):
        return {
             'uuid': self.uuid,
             'cutout_uuid': self.cutout_uuid,
             'predictions': [(x[0], x[1], float(x[2])) for x in self.predictions]
        }
