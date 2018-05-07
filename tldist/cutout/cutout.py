import uuid
import json
import numpy as np

from tldist.data import Data

class Cutout:
    """
    This cutout class represents one cutout of an image.  Likely a fingerprint
    will be calculated from this cutout.
    """

    def __init__(self, data, bounding_box, generator_parameters):
        """
        :param data: Data object
        :param bounding_box: [row_start, row_end, col_start, col_end]
        :param generator_parameters: parameters used to generate the cutout
        """
        self._uuid = str(uuid.uuid4())
        self._data = data
        self._bounding_box = bounding_box
        self._generator_parameters = generator_parameters
        self._cutout_processing = []

        # This is the "original data"
        # TODO: create a cached version as we really don't need this as it 
        #       is simply recreated
        bb = self._bounding_box
        self._original_data = self._data.get_data()[bb[0]:bb[1], bb[2]:bb[3]]

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
            'bounding_box': self._bounding_box,
            'cutout_processing': self._cutout_processing
        }

    def load(self, thedict):
        self._uuid = thedict['uuid']
        self._data = Data(thedict['data'])
        self._cutout_processing = Data(thedict['cutout_processing'])
        self._bounding_box = thedict['bounding_box']
