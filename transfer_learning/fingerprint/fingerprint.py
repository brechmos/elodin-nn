import uuid
import weakref

from ..tl_logging import get_logger
from transfer_learning.cutout import Cutout
import logging
log = get_logger('fingerprint', level=logging.WARNING)


class Fingerprint:

    #_fingerprint_collection = weakref.WeakValueDictionary()
    _fingerprint_collection = {}

    @staticmethod
    def factory(parameter, db=None):
        if isinstance(parameter, str):
            if parameter in Fingerprint._fingerprint_collection:
                return Fingerprint._fingerprint_collection[parameter]
            elif db is not None:
                fingerprint = db.find('fingerprint', parameter)
                fingerprint._cutout = db.find('cutout', fingerprint.cutout_uuid)
                return fingerprint
        elif isinstance(parameter, dict):
            if 'uuid' in parameter and parameter['uuid'] in Fingerprint._fingerprint_collection:
                return Fingerprint._fingerprint_collection[parameter['uuid']]

            return Fingerprint(cutout_uuid=parameter['cutout_uuid'],
                               predictions=parameter['predictions'],
                               uuid_in=parameter['uuid'])

    def __init__(self, cutout_uuid=None, predictions=[], uuid_in=None):
        if uuid_in is not None:
            self._uuid = uuid_in
        else:
            self._uuid = str(uuid.uuid4())
        self._cutout_uuid = cutout_uuid
        self._predictions = predictions
        self._cutout = None

        Fingerprint._fingerprint_collection[self._uuid] = self

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
    def cutout(self):
        return self._cutout

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

    def load(self, thedict, db=None):
        self._uuid = thedict['uuid']
        self._cutout_uuid = thedict['cutout_uuid']
        self._cutout = db.find('cutout', self._cutout_uuid)
        self._predictions = thedict['predictions']

        if db is not None:
            self._cutout = Cutout.factory(self._cutout_uuid, db)

    def save(self):
        return {
             'uuid': self._uuid,
             'cutout_uuid': self._cutout_uuid,
             'predictions': [(x[0], x[1], float(x[2]))
                             for x in self._predictions]
        }
