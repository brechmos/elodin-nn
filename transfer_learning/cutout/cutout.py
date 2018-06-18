import uuid

from cachetools.func import lru_cache
import numpy as np

from transfer_learning.data import Data
from transfer_learning.misc.image_processing import ImageProcessing

from ..tl_logging import get_logger
log = get_logger('cutout')


class CutoutCollection(object):
    """
    This is a collection of Data objects. There isn't necessarily
    much that ties them together other than just being in the collection.
    """

    # The main dictionary that will store all the actual
    # data elements.
    _collection = {}

    def _add(cutout):
        CutoutCollection._collection[cutout.uuid] = cutout

    def __init__(self, cutouts=None):

        self._uuid = str(uuid.uuid4())

        #
        # The dictionary of data as part of this. We are going
        # to store it as a dictionary so the UUID lookup is faster.
        #

        if isinstance(cutouts, list) and len(cutouts) > 0 and isinstance(cutouts[0], Cutout):

            # Store the UUIDs
            self._collection = [x.uuid for x in cutouts]

            # Set into main dictionary
            for data in cutouts:
                CutoutCollection._collection[data.uuid] = data

        #
        # If the cutouts passed in is a list of UUIDs then just
        # store them.
        #

        elif isinstance(cutouts, list) and len(cutouts) > 0 and isinstance(cutouts[0], str):

            # Store the UUIDs
            self._collection = [x for x in cutouts]

        else:
            self._collection = []

    #
    #  Properties
    #

    @property
    def uuid(self):
        return self._uuid

    @property
    def cutouts(self):
        """
        Return a list of the cutouts.
        """
        return [CutoutCollection._collection[k] for k in self._collection]

    #
    # Regular methods
    #

    def find(self, uuid):
        """
        Retrieve the cutout if it exists in here.
        """
        return CutoutCollection._collection[uuid] if uuid in self._collection else None

    def add(self, cutout):
        """
        Add a cutout into the collection.
        """
        CutoutCollection._collection[cutout.uuid] = cutout
        self._collection.append(cutout.uuid)

    def __add__(self, cutout_collection):
        return CutoutCollection(self._collection + cutout_collection._collection)

    #
    # Internal functions
    #

    def __len__(self):
        return len(self._collection)

    def __getitem__(self, index):
        return CutoutCollection._collection[self._collection[index]]

    def __iter__(self):
        self.__collection_pos__ = 0
        return self

    def __next__(self):

        if self.__collection_pos__ >= len(self._collection):
            raise StopIteration
        d = CutoutCollection._collection[self._collection[self.__collection_pos__]]
        self.__collection_pos__ = self.__collection_pos__ + 1
        return d

    #
    # Load and save
    #

    def save(self):
        return {
            'cutout_collection': [CutoutCollection._collection[x].save()
                                  for x in self._collection]
        }

    def load(self, thedict):
        for cutout_dict in thedict['cutout_collection']:
            c = Cutout().load(cutout_dict)
            self.add(c)


class BoundingBox(object):
    """
    Encompass a bounding box.  I did this as it got very confusing
    due to the large number of ways a bounding box can be defined.
    """

    def __init__(self, left, right, bottom, top):
        if right < left or top < bottom:
            raise ValueError('BoundingBox parameters, {} {} {} {}, are not correct'.format(
                left, right, bottom, top))

        self._bounding_box = [left, right, bottom, top]

    def __str__(self):
        return 'BoundingBox {} {} {} {}'.format(*self._bounding_box)

    def save(self):
        return {'bounding_box': self._bounding_box}

    #
    # Static methods
    #

    @staticmethod
    def load(thedict):
        bb = thedict['bounding_box']
        return BoundingBox(bb[0], bb[1], bb[2], bb[3])

    #
    # Regular Methods
    #

    def overlap(self, bounding_box):
        """
        Does this bounding box overlap with the one passed in...

        Parameters
        -----------
        bounding_box : BoundingBox
            Another bounding box.

        Return
        ------
        bool
            True if they overlap otherwise False
        """

        hoverlaps = True
        voverlaps = True

        if (self.left > bounding_box.right) or (self.right < bounding_box.left):
            hoverlaps = False

        if (self.top < bounding_box.bottom) or (self.bottom > bounding_box.top):
            voverlaps = False

        return hoverlaps and voverlaps

    def isin(self, point):
        """
        Check to see if the point is in the bounding box.

        Parameters
        -----------
        point: list/tuple of 2 elements
            Two points defined in space.

        """
        log.info('point {}'.format(point))

        if point[0] >= self.bottom and point[0] <= self.top and point[1] >= self.left and point[1] <= self.right:
            return True
        else:
            return False

    def distance(self, point):
        """
        Measure the distance from the center of a bounding_box to a point.

        Parameters
        -----------
        point: list/tuple of 2 elements
            Two points defined in space.
        """
        log.info('point {}'.format(point))

        # Calculate the center of the bounding box.
        center = ((self.left+self.right)/2.0, (self.bottom+self.top)/2.0)

        # Calculate the distance
        distance = np.sqrt((center[0]-point[1])**2+(center[1]-point[0])**2)

        return distance

    #
    # Properties
    #

    @property
    def left(self):
        return self._bounding_box[0]

    @property
    def right(self):
        return self._bounding_box[1]

    @property
    def bottom(self):
        return self._bounding_box[2]

    @property
    def top(self):
        return self._bounding_box[3]

    @property
    def width(self):
        return self._bounding_box[1] - self._bounding_box[0]

    @property
    def height(self):
        return self._bounding_box[3] - self._bounding_box[2]


class Cutout(object):
    """
    This cutout class represents one cutout of an image.  Likely a fingerprint
    will be calculated from this cutout.
    """

    @staticmethod
    def factory(parameter):
        if parameter['uuid'] in CutoutCollection._collection:
            return CutoutCollection._collection[parameter['uuid']]
        else:
            cutout = Cutout()
            cutout.load(parameter)
            return cutout

    def __init__(self, data=None, bounding_box=None, generator_parameters=None,
                 cutout_processing=None, uuid_in=None):
        """
        Cutout initializer.

        Parameters
        ----------
        data : Data
            Data object from where the cutout came from
        bounding_box: BoundingBox
            Bounding box of the data which represents this cutout.
        generator_paramters: Cutout Generator
            The cutout generator that created the cutout.
        cutout_processing: list of instances from processing.py
            Data will be first processed by the processors in cutout_processing and then returned.
        uuid_in: UUID
            unique uuid

        Return
        ------
        resize : Instance of class Resize.
            Will load the parameters.

        """
        log.info(' calling with cutout_processing {}'.format(cutout_processing))

        #
        #  Set the incoming parameters
        #

        if uuid_in is None:
            self._uuid = str(uuid.uuid4())
        else:
            self._uuid = uuid_in

        self._data = data
        self._bounding_box = bounding_box
        self._generator_parameters = generator_parameters

        self._base_cutout_uuid = None
        if cutout_processing is None:
            self._cutout_processing = []
        else:
            self._cutout_processing = [ImageProcessing.load(x) if isinstance(x, dict) else x for x in cutout_processing]

        #
        # This is the "original data"
        #
        self._original_data = None
        self._cached_output = None

        CutoutCollection._add(self)

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

        if not isinstance(value, BoundingBox):
            log.error('Bounding box must be a Boundingbox')
            raise Exception('Bounding box must be a list of 4 integers')

        self._bounding_box = BoundingBox(*value)

    @property
    def cutout_processing(self):
        return self._cutout_processing

    @cutout_processing.setter
    def cutout_processing(self, value):
        self._cutout_processing = value

    def add_processing(self, cutout_processing, base_cutout_uuid=None):
        """
        Add cutout processing (e.g., histogram equalization) to this
        cutout.  We might want to store the base cutout uuid to link
        it back to the original, unprocessed cutout.

        Parameters
        ----------
        cutout_processing: list of instances from processing.py
            The list of instances from processing.py
        base_cutout_uuid: UUID
            The unprocessed cutout.

        """
        log.info('Adding processing {}'.format(cutout_processing))

        self._cutout_processing = cutout_processing

        #
        # Add in info about the base cutout uuid
        #

        if base_cutout_uuid is not None:
            self._base_cutout_uuid = base_cutout_uuid

    def duplicate_with_processing(self, cutout_processing):
        """
        Duplicate this cutout and add the cutout processing.

        Parameters
        ----------
        cutout_processing: list of instances from processing.py
            The list of instances from processing.py

        Return
        ------
        cutout : Cutout
            Duplicate of this cutout, and with processing.

        """

        #
        # Create the new cutout
        #

        cutout = Cutout(self._data, self._bounding_box,
                        self._generator_parameters,
                        self._cutout_processing)

        #
        # Add in the cutout processing
        #

        cutout.add_processing(cutout_processing, base_cutout_uuid=self.uuid)

        return cutout

    @lru_cache(maxsize=1024)
    def get_data(self):
        """
        Retrieve the data

        Return
        ------
        data : numpy array
            cutout data, with any processing
        """
        log.info('')

        #
        # Cache the original data if not already set.
        #

        bb = self._bounding_box
        data = self._data.get_data()[bb.left:bb.right, bb.bottom:bb.top]

        #
        # Apply the processing
        #

        for processing in self._cutout_processing:
            data = processing.process(data)

        return data

    def save(self):
        log.info('')
        return {
            'uuid': self._uuid,
            'data': self._data.save(),
            'bounding_box': self._bounding_box.save(),
            'generator_parameters': self._generator_parameters,
            'base_cutout_uuid': self._base_cutout_uuid,
            'cutout_processing': [x.save() for x in self._cutout_processing]
        }

    def load(self, thedict):
        log.info('Loading cutout')

        self._uuid = thedict['uuid']
        self._data = Data.factory(thedict['data'])
        self._generator_parameters = thedict['generator_parameters']

        self._bounding_box = BoundingBox.load(thedict['bounding_box'])

        self._base_cutout_uuid = thedict['base_cutout_uuid']
        self._cutout_processing = [ImageProcessing.load(x) for x in thedict['cutout_processing']]

        # Add to the cutout collection
        CutoutCollection._add(self)
