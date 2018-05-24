import uuid
import weakref

from ..tl_logging import get_logger
from transfer_learning.cutout import Cutout
import logging
log = get_logger('fingerprint')


class FingerprintFilter(object):
    """
    Simple fingerprint filter object, primarily used for
    filtering of sets of fingerprints.

    TODO: Decide if this is going to be a collection class
          or just a filter class.
    """

    def __init__(self, inclusion_patterns=None, exclusion_patterns=None):
        self._inclusion_patterns = inclusion_patterns
        self._exclusion_patterns = exclusion_patterns

    #
    #  Filters
    #

    def filter(self, fingerprints):
        """
        If only inclusion_patterns is specified, only the names which match
           one or more patterns are returned.
        If only exclusion_patterns is specified, only the names which do not
           match any pattern are returned.
        If both are specified, the exclusion patterns take precedence.
        If neither is specified, the input is returned as-is.

        inclusion/exclusion pattern can be either:
           {'key': 'value'}  If the key exists then the value will be checked
           'value' If no key, then it will check all values in the dict

        """
        included = self.multi_filter(fingerprints, self._inclusion_patterns) if self._inclusion_patterns else fingerprints
        excluded = self.multi_filter(fingerprints, self._exclusion_patterns) if self._exclusion_patterns else []
        return set(included) - set(excluded)

    def multi_filter(self, fingerprints, patterns):
        """
        Generator function which yields the names that match one or more of the patterns.
        """
        for fingerprint in fingerprints:
            if any(self._fingerprint_meta_checker(fingerprint, pattern) for pattern in patterns):
                yield fingerprint

    def _fingerprint_meta_checker(self, fingerprint, pattern):
        """
        Check to see that the pattern is in the meta of the fingerprint
        which lives in fingerprint.cutout.data.meta.
        """

        # Grab the meta
        print('fingerprint {}'.format(fingerprint))
        print('cutout {}'.format(fingerprint.cutout))
        print('data {}'.format(fingerprint.cutout.data))
        print('meta {}'.format(fingerprint.cutout.data.meta))
        meta = fingerprint.cutout.data.meta

        # If a string, then just see if it fits in any value
        log.debug('pattern is {}'.format(pattern))
        log.debug('meta is {}'.format(meta))
        if isinstance(pattern, str):
            return any(pattern in str(v) for k, v in meta.items())

        # If pattern is a dict then check each key, value pair and
        # they should all be true for this to pass.
        elif isinstance(pattern, dict):
            return all(pv in meta[pk] for pk, pv in pattern.items())


class Fingerprint(object):

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

    def __init__(self, cutout_uuid=None, cutout=None, predictions=[], uuid_in=None):
        if uuid_in is not None:
            self._uuid = uuid_in
        else:
            self._uuid = str(uuid.uuid4())
        if cutout is None:
            self._cutout_uuid = cutout_uuid
            self._cutout = None
        else:
            self._cutout = cutout
            self._cutout_uuid = cutout.uuid
        self._predictions = predictions

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
