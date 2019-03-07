import uuid
import re
from io import BytesIO

import numpy as np
import imageio
import requests
from cachetools.func import lru_cache

from ..misc.image_processing import ImageProcessing

from ..tl_logging import get_logger
log = get_logger('data')


def stringify(dictionary):
    """
    Used to make everything a string in the value part of the
    dict.  A couple things in meta were not, so needed this.
    """
    return {k: str(v) for k, v in dictionary.items()}


class DataCollection(object):
    """
    This is a collection of Data objects. There isn't necessarily
    much that ties them together other than just being in the collection.
    """

    # The main dictionary that will store all the actual
    # data elements.
    _collection = {}

    @staticmethod
    def _add(data):
        DataCollection._collection[data.uuid] = data

    def __init__(self, datas=None):

        self._uuid = str(uuid.uuid4)

        #
        #  The local _collection is a list of UUIDs to get
        #  the actual info, need to do a lookup into the
        #  static _collection.
        #

        if datas is not None and isinstance(datas, list):

            # Store the UUIDs
            self._collection = [x.uuid for x in datas]

            # Set into main dictionary
            for data in datas:
                DataCollection._collection[data.uuid] = data

        elif datas is not None and isinstance(datas, DataCollection):
            self._collection = datas._collection.copy()

        else:
            self._collection = []

    #
    # Properties
    #

    @property
    def uuid(self):
        return self._uuid

    @property
    def data(self):
        """
        Return a list of the data.
        """
        yield [DataCollection._collection[x] for x in self._collection]

    #
    # Internal methods
    #

    def __len__(self):
        return len(self._collection)

    def __getitem__(self, index):
        """
        Used for indexing.
        """
        return DataCollection._collection[self._collection[index]]

    def __iter__(self):
        """
        Used for 'for' loops
        """
        self.__collection_pos__ = 0
        return self

    def __next__(self):
        """
        Used for 'for' loops
        """
        if self.__collection_pos__ >= len(self._collection):
            raise StopIteration
        d = DataCollection._collection[self._collection[self.__collection_pos__]]
        self.__collection_pos__ = self.__collection_pos__ + 1
        return d

    #
    # Public methods
    #

    def find(self, uuid):
        """
        Retrieve the data if it exists in here.
        """
        return DataCollection._collection[uuid] if uuid in self._collection else None

    def add(self, data):
        """
        Add a data into the collection.
        """

        # Add to the main collection. Essentially an update
        # if it already exists in the data collection
        DataCollection._collection[data.uuid] = data

        # Add to this collection.
        self._collection.append(data.uuid)

    #
    # Save and load
    #

    def save(self):
        """
        Save all the data that is in the data collection.
        """

        return {
            'data_collection': [DataCollection._collection[x].save()
                                for x in self._collection]
        }

    @staticmethod
    def load(self, thedict):
        for data_dict in thedict['data_collection']:
            d = Data().load(data_dict)
            self._add(d)


class Data:

    @staticmethod
    def factory(parameter):

        if parameter['uuid'] in DataCollection._collection:
            return DataCollection._collection[parameter['uuid']]
        else:
            data = Data()
            data.load(parameter)
            return data

    def __init__(self, location='', processing=None, radec=(), meta={}, uuid_in=None):
        """
        Initialize the data object.

        Parameters
        -----------
        location : str
            The file or URL of the location of the data.

        processing : list of ImageProcessing or list of dict
            List of instances of ImageProcessing objects or list of `save()`
            versions of ImageProcessing objects.
            Defaults to ``None``.

        radec : tuple
            (RA, DEC) of the location of the data object.
            Default is ().

        meta : dict
            Meta information about the data, typically this will come
            from the astroquery data or something else.
            Default is {}.

        uuid_in : UUID, optional
            Primary Key

        """

        #
        #  Set the UUID if passed in or create it if not passed in.
        #

        if uuid_in is None:
            self._uuid = str(uuid.uuid4())
        else:
            self._uuid = uuid_in

        #
        # Set the main parameters.
        #

        self._location = location
        self._radec = radec
        if processing is None:
            self._processing = []
        else:
            self._processing = [ImageProcessing.load(p) if isinstance(p, dict) else p for p in processing]
        self._meta = meta

        #
        # Cache of the 2D or 3D numpy array data
        #

        self._cached_data = None

        #
        # Store self in the data_collection.
        #

        DataCollection._add(self)

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
        if not isinstance(value, (tuple, list)):
            raise ValueError('RA/DEC must be a tuple of two numbers')
        self._radec = value

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        if not isinstance(value, dict):
            raise ValueError('Meta must be a dict')
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

        #
        # Load the dataset from a URL
        #

        if 'http' in self.location:
            success = True
            try:
                response = requests.get(self.location)
            except requests.exceptions.RequestException as e:
                success = False

            if not response.status_code == 200 or not success:
                log.error('Problem loading the data {}'.format(self.location))
                raise Exception('Problem loading the data {}'.format(self.location))

            self._cached_data = np.array(imageio.imread(BytesIO(response.content)))

        #
        # Load a local dataset.
        #

        elif re.match(regex, self.location):
            self._cached_data = np.array(imageio.imread(self.location))

        #
        # Unknown dataset - raise an exception
        #

        else:
            log.error('Unknown file type of file {}'.format(self.location))
            raise Exception('Unknown type of file {}'.format(self.location))

        #
        # Apply the processing and store in the cached data
        #

        for processor in self._processing:
            self._cached_data = processor.process(self._cached_data)

        #
        # If 2D, make 3D
        #

        if len(self._cached_data.shape) == 2:
            self._cached_data = self._gray2rgb(self._cached_data)

    def _gray2rgb(self, data):
        """
        Convert 2D data set to 3D gray scale

        :param data:
        :return:
        """
        data_out = np.zeros((data.shape[0], data.shape[1], 3))
        data_out[:, :, 0] = data
        data_out[:, :, 1] = data
        data_out[:, :, 2] = data
        return data_out

    @lru_cache(maxsize=1024)
    def get_data(self):
        """
        Retrieve the numpy array of data

        :return: 2D or 3D data array
        """
        log.info('Retrieving data...')

        # If we have the data already loaded then don't
        # need to reload all the data.

        log.debug('Data is not cached, so will need to load it')
        regex = r".*[jpg|tif|tiff]$"

        #
        # Load the dataset from a URL
        #

        if 'http' in self.location:
            success = True
            try:
                response = requests.get(self.location)
            except requests.exceptions.RequestException as e:
                success = False

            if not response.status_code == 200 or not success:
                log.error('Problem loading the data {}'.format(self.location))
                raise Exception('Problem loading the data {}'.format(self.location))

            data = np.array(imageio.imread(BytesIO(response.content)))

        #
        # Load a local dataset.
        #

        elif re.match(regex, self.location):
            data = np.array(imageio.imread(self.location))

        #
        # Unknown dataset - raise an exception
        #

        else:
            log.error('Unknown file type of file {}'.format(self.location))
            raise Exception('Unknown type of file {}'.format(self.location))

        #
        # Apply the processing and store in the cached data
        #

        for processor in self._processing:
            data = processor.process(data)

        #
        # If 2D, make 3D
        #

        if len(data.shape) == 2:
            data = self._gray2rgb(data)

        return data 

    def get_data_OLD(self):
        """
        Retrieve the numpy array of data

        :return: 2D or 3D data array
        """
        log.info('Retrieving data...')

        if self._cached_data is None:
            self._load_and_process()

        return self._cached_data

    def add_processing(self, processing):

        if not isinstance(processing, ImageProcessing):
            raise Exception('Must be a ImageProcessing instance.')

        self._processing.append(processing)

    def save(self):
        return {
            'uuid': self._uuid,
            'location': self._location,
            'radec': self._radec,
            'processing': [x.save() for x in self._processing],
            'meta': stringify(self._meta)
        }

    def load(self, thedict):
        self._uuid = thedict['uuid']
        self._location = thedict['location']
        self._radec = thedict['radec']
        self._processing = [ImageProcessing.load(x) for x in thedict['processing']]
        self._meta = thedict['meta']

        # Add data to the main collection
        DataCollection._add(self)
