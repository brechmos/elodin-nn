import uuid
import json
import re
import logging
from io import BytesIO

import numpy as np
import imageio
import requests

logging.basicConfig(format='%(levelname)-6s: %(asctime)-15s %(name)-10s %(funcName)-10s %(message)s')
log = logging.getLogger("data")
fhandler = logging.FileHandler(filename='/tmp/mylog.log', mode='a')
log.addHandler(fhandler)
log.setLevel(logging.DEBUG)

def stringify(dictionary):
    return {k: str(v) for k, v in dictionary.items()}

class Data:

    @staticmethod
    def data_factory(thedict):
        return Data(location=thedict['location'], radec=thedict['radec'], meta=thedict['meta'])

    def __init__(self, location='', radec=[], meta={}):
        self._uuid = str(uuid.uuid4())
        self._location = location
        self._radec = radec
        self._meta = meta

        self._cached_data = None

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
        self._location = location

    @property
    def radec(self):
        return self._radec

    @radec.setter
    def radec(self, value):
        self._radec = radec

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

    def get_data(self):
        log.info('Data going to be returned')

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
                log.error('Problem loading the data {}'.format(datum.location))
                raise Exception('Problem loading the data {}'.format(datum.location))

            self._cached_data = np.array(imageio.imread(BytesIO(response.content)))
            return self._cached_data

        # Local dataset
        elif re.match(regex, self.location):
            self._cached_data = np.array(imageio.imread(self.location))
            return self._cached_data

        # Unknwon dataset
        else:
            log.error('Unknown file type of file {}'.format(self.location))
            raise Exception('Unknown type of file {}'.format(self.location))

    def save(self):
        return {
            'uuid': self._uuid,
            'location': self._location,
            'radec': self._radec,
            'meta': stringify(self._meta)
        }

    def load(self, thedict):
        self._uuid = thedict['uuid']
        self._location = thedict['location']
        self._radec = thedict['radec']
        self._meta = thedict['meta']
