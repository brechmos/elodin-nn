import weakref
import uuid

from transfer_learning.data import Data
from transfer_learning.cutout.processing import CutoutProcessing
from transfer_learning.image import Image

import numpy as np
from ..tl_logging import get_logger
import logging
log = get_logger('cutout')

class CutoutImage(Image):
    def __init__(self, cutout):
        self._cutout = cutout

class Cutout(object):
    """
    This cutout class represents one cutout of an image.  Likely a fingerprint
    will be calculated from this cutout.
    """

    # Collection to confirm we have unique instances based on uuid
    # _cutout_collection = weakref.WeakValueDictionary()
    _cutout_collection = {}

    @staticmethod
    def factory(parameter, db=None):
        if isinstance(parameter, str):
            if parameter in Cutout._cutout_collection:
                return Cutout._cutout_collection[parameter]
            elif db is not None:
                return db.find('cutout', parameter)
        elif isinstance(parameter, dict):

            if 'uuid' in parameter and parameter['uuid'] in Cutout._cutout_collection:
                return Cutout._cutout_collection[parameter['uuid']]

            data = Data.factory(parameter['data'], db)
            return Cutout(data, parameter['bounding_box'], parameter['generator_parameters'],
                          cutout_processing=parameter['cutout_processing'],
                          uuid_in=parameter['uuid'])

    def __init__(self, data, bounding_box, generator_parameters, cutout_processing=None, uuid_in=None):
        """
        :param data: Data object
        :param bounding_box: [row_start, row_end, col_start, col_end]
        :param generator_parameters: parameters used to generate the cutout
        """
        log.info(' calling with cutout_processing {}'.format(cutout_processing))
        if uuid_in is None:
            self._uuid = str(uuid.uuid4())
        else:
            self._uuid = uuid_in

        self._data = data
        self._bounding_box = bounding_box
        self._generator_parameters = generator_parameters

        self._cutout_processing = [] if cutout_processing is None else [CutoutProcessing.load(x) for x in cutout_processing]

        bb = self._bounding_box

        # This is the "original data"
        self._original_data = self._data.get_data()[bb[0]:bb[1], bb[2]:bb[3]]

        self._cached_output = None

        Cutout._cutout_collection[self._uuid] = self

        log.info('Creaing new cutout with data = {},  bounding_box = {}, generator_parameters = {}, cutout_rpcoessing = {}, uuid_in {}'.format(
            data, bounding_box, generator_parameters, cutout_processing, uuid_in))


    def __str__(self):
        return 'Cutout for data {} with box {} and processing {}'.format(
                self._data, self._bounding_box, self._cutout_processing)

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = uuid

    @property
    def data(self):
        return self._data

    @property
    def generator_parameters(self):
        return self._generator_parameters

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

    @property
    def cutout_processing(self):
        return self._cutout_processing

    @cutout_processing.setter
    def cutout_processing(self, value):
        self._cutout_processing = value

    def add_processing(self, cutout_processing):
        log.info('Adding processing {}'.format(cutout_processing))
        self._cutout_processing.append(cutout_processing)

    def get_data(self):
        log.info('')

        if self._cached_output is None:
            data = self._original_data

            # Apply the processing
            for processing in self._cutout_processing:
                data = processing.process(data)

            self._cached_output = data

        return self._cached_output

    def save(self):
        log.info('')
        return {
            'uuid': self._uuid,
            'data': self._data.save(),
            'bounding_box': self._bounding_box,
            'generator_parameters': self._generator_parameters,
            'cutout_processing': [x.save() for x in self._cutout_processing]
        }

    def load(self, thedict):
        log.info('Loading cutout')

        self._uuid = thedict['uuid']
        self._data = Data(thedict['data'])
        self._generator_parameters = Data(thedict['generator_parameters'])
        self._cutout_processing = [CutoutProcessing.load(x) for x in thedict['cutout_processing']]
        self._bounding_box = thedict['bounding_box']
