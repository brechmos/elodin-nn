import uuid

from ..tl_logging import get_logger
log = get_logger('fingerprint')


class Fingerprint:

    _fingerprint_collection = {}

    @staticmethod
    def fingerprint_factory(parameter):
        if isinstance(parameter, str):
            return Fingerprint._fingerprint_collection[parameter]
        else:
            return Fingerprint(cutout_uuid=parameter['cutout_uuid'],
                               predictions=parameter['predictions'],
                               uuid_in=parameter['uuid'])

    def __init__(self, cutout_uuid=None, predictions=[], uuid_in=None):
        if uuid_in is None:
            self._uuid = str(uuid.uuid4())
        else:
            if self._uuid in self._fingerprint_collection:
                return self._fingerprint_collection[self._uuid]
            self._uuid = uuid_in
        self._cutout_uuid = cutout_uuid
        self._predictions = predictions

        self._fingerprint_collection[self._uuid] = self

    def __del__(self):
        del Fingerprint._fingerprint_collection[self._uuid]

    def __str__(self):
        return 'Fingerprint {} based on cutout {} with predictions {}'.format(
                self._uuid, self._cutout_uuid, [x for x in self._predictions[:3]])

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
        self._uuid = thedict['uuid']
        self._cutout_uuid = thedict['cutout_uuid']
        self._predictions = thedict['predictions']

    def save(self):
        return {
             'uuid': self._uuid,
             'cutout_uuid': self._cutout_uuid,
             'predictions': [(x[0], x[1], float(x[2]))
                             for x in self._predictions]
        }
