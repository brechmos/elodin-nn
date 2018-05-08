import uuid
import re
from io import BytesIO

import numpy as np
import imageio
import requests

from .processing import DataProcessing

from ..tl_logging import get_logger
log = get_logger('cutout generator', '/tmp/mylog.log')


def stringify(dictionary):
    return {k: str(v) for k, v in dictionary.items()}


class Data:

    # Collection to confirm we have unique instances based on uuid
    _data_collection = {}

    @staticmethod
    def data_factory(parameter):

        # If parameter is a UUID
        if isinstance(parameter, str):
            return Data._data_collection[parameter]
        else:
            return Data(location=parameter['location'], processing=parameter['processing'],
                        radec=parameter['radec'], meta=parameter['meta'],
                        uuid_in=parameter['uuid'])

    def __init__(self, location='', processing=[], radec=[], meta={}, uuid_in=None):
        if uuid_in is None:
            self._uuid = str(uuid.uuid4())
        else:
            self._uuid = uuid_in

        self._location = location
        self._radec = radec
        self._processing = processing
        self._meta = meta

        # 2D or 3D Numpy array
        self._cached_data = None

        self._data_collection[self._uuid] = self

    def __del__(self):
        del self._data_collection[self._uuid]

    def __str__(self):
        return 'Data located {} at RA/DEC {}'.format(
                self.location, self.radec)


    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = uuid

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        self._location = value

    @property
    def radec(self):
        return self._radec

    @radec.setter
    def radec(self, value):
        self._radec = value

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        self._meta = value

    @property
    def shape(self):
        if self._cached_data is None:
            self._cached_data = self.get_data()
        return self._cached_data.shape

    def _load_and_process(self):
        # If we have the data already loaded then don't
        # need to reload all the data.
        if self._cached_data is not None:
            return self._cached_data

        log.debug('Data is not cached, so will need to load it')
        regex = r".*[jpg|tif|tiff]$"

        # Distant dataset
        if 'http' in self.location:
            response = requests.get(self.location)

            if not response.status_code == 200:
                log.error('Problem loading the data {}'.format(self.location))
                raise Exception('Problem loading the data {}'.format(self.location))

            self._cached_data = np.array(imageio.imread(BytesIO(response.content)))

        # Local dataset
        elif re.match(regex, self.location):
            self._cached_data = np.array(imageio.imread(self.location))

        # Unknown dataset
        else:
            log.error('Unknown file type of file {}'.format(self.location))
            raise Exception('Unknown type of file {}'.format(self.location))

        for x in self._processing:
            processor = DataProcessing.load(x)
            self._cached_data = processor.process(self._cached_data)

    def get_data(self):
        """
        Retrieve the numpy array of data

        :return: 2D or 3D data array
        """
        log.info('Data going to be returned')

        if self._cached_data is None:
            self._load_and_process()

        return self._cached_data

    def add_processing(self, processing):

        if not isinstance(processing, dict):
            raise Exception('Data processing must be a dict that describes the type of processing.')

        self._processing.append(processing)

    def save(self):
        return {
            'uuid': self._uuid,
            'location': self._location,
            'radec': self._radec,
            'processing': self._processing,
            'meta': stringify(self._meta)
        }

    def load(self, thedict):
        self._uuid = thedict['uuid']
        self._location = thedict['location']
        self._radec = thedict['radec']
        self._processing = thedict['processing']
        self._meta = thedict['meta']
