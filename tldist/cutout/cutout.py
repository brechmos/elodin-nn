import uuid
import json
import numpy as np

from tldist.data import Data

class Cutout:
    """
    This cutout class represents one cutout of an image.  Likely a fingerprint
    will be calculated from this cutout.
    """

    def __init__(self, data, bounding_box):
        self._uuid = str(uuid.uuid4())
        self._data = data
        self._bounding_box = bounding_box

    def __str__(self):
        return 'Cutout for data {} with box {}'.format(
                self._data, self._bounding_box)

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = uuid

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(Data, value):
            log.error('Data must be of type data')
            raise Exception('Data must be of type data')

        self._data = data

    @property
    def bounding_box(self):
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, value):

        if not isinstance(value, (list, tuple)) and not len(value) == 4:
            log.error('Bounding box must be a list of 4 integers')
            raise Exception('Bounding box must be a list of 4 integers')
        
        self._bounding_box = bounding_box

    def get_data(self):
        
        bb = self._bounding_box
        data = self._data.get_data()[bb[0]:bb[1], bb[2]:bb[3]]
        return data

    def save(self):
        return {
            'uuid': self._uuid,
            'data': self._data.save(),
            'bounding_box': self._bounding_box
        }

    def load(self, thedict):
        self._uuid = thedict['uuid']
        self._data = Data(thedict['data'])
        self._bounding_box = thedict['bounding_box']
