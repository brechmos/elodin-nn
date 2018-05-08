import uuid
import logging

from tldist.data import Data

logging.basicConfig(format='%(levelname)-6s: %(asctime)-15s %(name)-10s %(funcName)-10s %(message)s')
log = logging.getLogger("cutout")
log.setLevel(logging.INFO)


class Cutout:
    """
    This cutout class represents one cutout of an image.  Likely a fingerprint
    will be calculated from this cutout.
    """

    # Collection to confirm we have unique instances based on uuid
    _cutout_collection = {}

    @staticmethod
    def cutout_factory(parameter):
        if isinstance(parameter, str):
            return Cutout._cutout_collection[parameter]
        else:
            data = Data.data_factory(parameter['data'])
            return Cutout(data, parameter['bounding_box'], parameter['generator_parameters'], uuid_in=parameter['uuid'])

    def __init__(self, data, bounding_box, generator_parameters, uuid_in=None):
        """
        :param data: Data object
        :param bounding_box: [row_start, row_end, col_start, col_end]
        :param generator_parameters: parameters used to generate the cutout
        """

        if uuid_in is None:
            self._uuid = str(uuid.uuid4())
        else:
            if uuid_in in self._cutout_collection:
                return self._cutout_collection[uuid_in]
            self._uuid = uuid_in

        self._data = data
        self._bounding_box = bounding_box
        self._generator_parameters = generator_parameters
        self._cutout_processing = []

        # This is the "original data"
        # TODO: create a cached version as we really don't need this as it
        #       is simply recreated
        bb = self._bounding_box
        self._original_data = self._data.get_data()[bb[0]:bb[1], bb[2]:bb[3]]

        self._cutout_collection[self._uuid] = self

    def __del__(self):
        del self._cutout_collection[self._uuid]

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

        self._data = value

    @property
    def bounding_box(self):
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, value):

        if not isinstance(value, (list, tuple)) and not len(value) == 4:
            log.error('Bounding box must be a list of 4 integers')
            raise Exception('Bounding box must be a list of 4 integers')

        self._bounding_box = value

    def get_data(self):

        bb = self._bounding_box
        data = self._data.get_data()[bb[0]:bb[1], bb[2]:bb[3]]
        return data

    def save(self):
        return {
            'uuid': self._uuid,
            'data': self._data.save(),
            'bounding_box': self._bounding_box,
            'generator_parameters': self._generator_parameters,
            'cutout_processing': self._cutout_processing
        }

    def load(self, thedict):
        self._uuid = thedict['uuid']
        self._data = Data(thedict['data'])
        self._generator_parameters = Data(thedict['generator_parameters'])
        self._cutout_processing = Data(thedict['cutout_processing'])
        self._bounding_box = thedict['bounding_box']
