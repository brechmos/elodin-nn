import weakref
import uuid
import itertools

import numpy as np
from sklearn.manifold import TSNE
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist, squareform
from tldist.fingerprint import Fingerprint

from ..tl_logging import get_logger
import logging
log = get_logger('similarity', '/tmp/mylog.log', level=logging.WARNING)


def calculate(fingerprints, similarity_calculator, serialize_output=False):
    """
    Similarity calculator.

    This function might be called locally and in that case we want to return the
    actual similarity calculator instance.  Or it might be run rmeotely (via celery)
    and in this case we want to reutrn the serialized version of the similarity instance.
    """
    log.info('Start threaded real_calculate {} fingerprints and simcalc {}'.format(
        len(fingerprints), similarity_calculator))

    # Create the right similarity calculator
    if similarity_calculator == 'tsne':
        sim = tSNE()
    elif similarity_calculator == 'jaccard':
        sim = Jaccard()
    elif similarity_calculator == 'distance':
        sim = Distance()

    # Calculate the similarity
    sim.calculate(fingerprints)

    # Return the thing
    if serialize_output:
        return sim.save()
    else:
        return sim


class Similarity:

    _similarity_collection = weakref.WeakValueDictionary()

    @staticmethod
    def factory(thedict):
        if isinstance(thedict, str):
            return Similarity._similarity_collection[thedict]
        else:
            if thedict['similarity_type'] == 'tsne':
                sim = tSNE()
            elif thedict['similarity_type'] == 'jaccard':
                sim = Jaccard()
            elif thedict['similarity_type'] == 'distance':
                sim = Distance()
            sim.load(thedict)
            return sim

    def __init__(self, similarity_type=None, similarity=None, fingerprint_uuids=[], uuid_in=None):
        if uuid_in is None:
            self._uuid = str(uuid.uuid4())
        else:
            self._uuid = uuid_in
        self._similarity_type = similarity_type
        self._similarity = similarity
        self._fingerprint_uuids = fingerprint_uuids
        self._parameters = {}

        self._similarity_collection[self._uuid] = self

    def __str__(self):
        return 'Similarity {} based on {}...'.format(
                self._similarity_type, self._fingerprint_uuids[:3])

    def __repr__(self):
        return self.__str__()

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = value

    @property
    def similarity_type(self):
        return self._similarity_type

    @similarity_type.setter
    def similarity_type(self, value):
        self._similarity_type = value

    @property
    def similarity(self):
        return self._similarity

    @similarity.setter
    def similarity(self, value):
        self._similarity = value

    @property
    def fingerprint_uuids(self):
        return self._fingerprint_uuids

    @fingerprint_uuids.setter
    def fingerprint_uuids(self, value):

        if not isinstance(value, list):
            raise ValueError('Fingerprint_uuids must be a list')

        self._fingerprint_uuids = value

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    def save(self):
        raise Exception('save function must be defined in subclass')

    def load(self, thedict):
        raise Exception('load function must be defined in subclass')


class tSNE(Similarity):

    _similarity_type = 'tsne'

    def __init__(self, *args, **kwargs):
        if 'display_type' in kwargs:
            display_type = kwargs['display_type']
            del kwargs['display_type']
        else:
            display_type = 'plot'

        super().__init__(tSNE._similarity_type, *args, **kwargs)

        log.info('Created {}'.format(self._similarity_type))

        # Each line / element in these should correpsond
        self._Y = None
        self._fingerprints = []
        self._filename_index = []
        self._distance_measure = 'l2'

        # Display types
        self._display_types = ['plot', 'hexbin', 'mpl']
        self._display_type = display_type

        self._distance_measures = {
            'l2': lambda Y, point:  np.sqrt(np.sum((Y - np.array(point))**2, axis=1)),
            'l1': lambda Y, point: np.sum(np.abs((Y - np.array(point)), axis=1)),
        }

    #
    #  Calculation Methods
    #

    def calculate(self, fingerprints):
        log.info('Going to calculate tSNE from {} fingerprints'.format(len(fingerprints)))

        # Calculate the unique labels
        labels = []
        values = {}
        for ii, fp in enumerate(fingerprints):
            log.debug('    fingerprint is {}'.format(fp))
            # Add to unique label list
            labels.extend([pred[1] for pred in fp.predictions if pred[1] not in labels])

            # Store the predictions for next processing
            values[ii] = fp.predictions

            self._fingerprints.append(fp)
        log.info('Unique labels {}'.format(labels))

        X = np.zeros((len(fingerprints), len(labels)))
        for ii, fp in enumerate(fingerprints):
            inds = [labels.index(prediction[1]) for prediction in values[ii]]
            X[ii][inds] = [prediction[2] for prediction in values[ii]]

        log.debug('X is {}'.format(X))
        log.debug('Fingerprint list {}'.format(self._fingerprints))

        # Compute the tSNE of the data.
        log.info('Calculating the tSNE...')
        self._Y = TSNE(n_components=2).fit_transform(X)
        log.debug('self._Y is {}'.format(self._Y))
        log.info('Done calculation')

    #
    #  Utility Methods
    #

    def save(self):
        log.info('Returning the dictionary of information')
        return {
            'uuid': self._uuid,
            'similarity_type': self._similarity_type,
            'similarity': self._Y.tolist(),
            'fingerprint_uuids': [fp.uuid for fp in self._fingerprints],
            'parameters': {
                'distance_measure': self._distance_measure
            }
        }

    def load(self, thedict, db=None):
        log.info('Loading the dictionary of information')
        self._uuid = thedict['uuid']
        self._similarity_type = thedict['similarity_type']
        self._Y = np.array(thedict['similarity'])
        self._fingerprint_uuids = thedict['fingerprint_uuids']
        self._parameters = thedict['parameters']
        self._distance_measure = self._parameters['distance_measure']


        if db is not None:
            self._fingerprints = [Fingerprint.factory(thedict['fingerprint_uuids'], db)]

    #
    #  Display methods
    #

    def set_display_type(self, display_type):
        """
        Set the display type. Currently 'plot', 'hexbin', and 'mpl' are defined.

        :param display_type:
        :return:
        """
        if display_type in self._display_types:
            self._display_type = display_type
        else:
            raise ValueError('Display type {} not in {}'.format(display_type, self._display_types))

    def select_distance_measure(self, distance_measure=None):
        """
        The function will display all the fingerprinting methods and return a class
        of the one of interest.  This can then be used to construct the class and
        to use in further calculations.

        :return:  selected class
        """

        if not distance_measure:
            dm_options = self._distance_measures.keys()

            selected = False
            N = 0
            while not selected:
                # Show the fingerprints in order to allow for the person to select one
                print('Select distance measure to use (q to quit:')
                for ii, x in enumerate(dm_options):
                    print('   {}) {}'.format(ii, x))
                    N = ii

                s = input('Select Number > ')

                if s == 'q':
                    return

                try:
                    s = int(s)
                    if s >= 0 and s < N:
                        self._distance_measure = self._distance_measures[s]
                except Exception:
                    pass
        else:
            if distance_measure in self._distance_measures:
                self._distance_measure = distance_measure
            else:
                self._distance_measure = self._distance_measures.keys()[0]
                print('ERROR: No definition for {} so using {} instead.'.format(distance_measure, self._distance_measure))

    def display(self, tsne_axis):
        """
        Display the Y values in an axis element.

        :param axes:
        :param args:
        :param kwargs:
        :return:
        """

        if self._display_type == 'plot':
            self._display_plot(tsne_axis)
        elif self._display_type == 'hexbin':
            return self._display_hexbin(tsne_axis)
#        elif self._display_type == 'mpl':
#            self._display_mpl(tsne_axis)
        else:
            raise ValueError('Plot type {} is not in the valid list {}'.format(self._display_type, self._display_types))

    def _display_plot(self, tsne_axis):
        # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        # for ii, fi in enumerate(set(self._filename_index)):
        #     inds = np.nonzero(self._filename_index == fi)[0]
        tsne_axis.plot(self._Y[:, 0], self._Y[:, 1])#, '.')
        tsne_axis.grid('on')
        tsne_axis.set_title('tSNE [{}]'.format(self._distance_measure))

    def _display_hexbin(self, tsne_axis):
        output = tsne_axis.hexbin(self._Y[:, 0], self._Y[:, 1], cmap='hot')
        tsne_axis.grid('on')
        tsne_axis.set_title('tSNE [{}]'.format(self._distance_measure))

        # Set the color limits so that it is a little brighter
        limmax = np.percentile(output.get_array(), 99.9)
        output.set_clim((0, limmax))

        return output

    def find_similar(self, point, n=9):
        log.info('Searching based on point {}'.format(point))
        distances = self._distance_measures[self._distance_measure](self._Y, point)
        inds = np.argsort(distances)
        log.debug('Closest indexes are {}'.format(inds[:n]))
        log.debug('Size of the fingerprint list {}'.format(len(self._fingerprints)))

        return [{
                    'tsne_point': self._Y[ind],
                    'distance': distances[ind],
                    'fingerprint_uuid': self._fingerprint_uuids[ind]
                } for ind in inds[:n]]


class Jaccard(Similarity):
    """
    Jaccard similarity implementation.  The Jaccard similarity is a set based method, which, implemented here,
    is computes the similiarity of two cutouts based on only the names of the imagenet images in the fingerprint for
    each.  So, given two cutouts C1 and C2 and their corresponding finger print imagenet sets S1 and S2.  The
    Jaccard similarity between the two is computed as len( S1 & S2 ) / len( S1 | S2 ).  The higher the number, the
    more similar the two cutouts.
    """

    _similarity_type = 'jaccard'

    def __init__(self, *args, **kwargs):
        """
        Create the empty instance of the similarity measure.

        :param args:
        :param kwargs:
        """
        super(Jaccard, self).__init__(Jaccard._similarity_type, *args, **kwargs)

        log.info('Created {}'.format(self._similarity_type))

        # Each line / element in these should correpsond
        self._fingerprints = []
        self._filename_index = []
        self._fingerprint_adjacency = None
        self._predictions = []

    #
    # Calculation Methods
    #

    def calculate(self, fingerprints):
        """
        Probably the best way to represent this is to calculate the Jaccard distance
        between every pair of points and represent it as an NxN matrix which can then be
        shown as an image.

        The algorithm to compute the Jaccard matrix here is from:
            https://stackoverflow.com/questions/40579415/computing-jaccard-similarity-in-python

        :param fingerprints:
        :return:
        """
        log.info('Going to calculate Jaccard from {} fingerprints'.format(len(fingerprints)))

        self._fingerprints = fingerprints

        self._predictions = [set([tt[1] for tt in fp.predictions]) for fp in fingerprints]

        up = list(set(list(itertools.chain(*[list(x) for x in self._predictions]))))

        A = np.zeros((len(self._predictions), len(up)))
        for ii, prediction in enumerate(self._predictions):
            indices = [up.index(x) for x in prediction]
            A[ii, indices] = 1.0

        sparse_adjacency = csc_matrix(A.transpose())
        self._fingerprint_adjacency = self.jaccard_similarities(sparse_adjacency).toarray().T

    def jaccard_similarities(self, mat):
        """
        Compute the jaccard similarities from the set matrix representation.

        :param mat:  Input matrix that defines set inclusion for all unique labels.
        :return: Jaccard similarity for each pair of rows and columns.
        """
        # https://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html
        log.debug('Calculate the Jaccard_similarities')

        cols_sum = mat.getnnz(axis=0)
        ab = mat.T * mat

        # for rows
        aa = np.repeat(cols_sum, ab.getnnz(axis=0))

        # for columns
        bb = cols_sum[ab.indices]

        similarities = ab.copy()
        similarities.data /= (aa + bb - ab.data)

        return similarities

    #
    #  Display Methods
    #

    def display(self, tsne_axis):
        """
        Display the fingerprint adjacency mastrix in the axis element.

        :param axes:
        :param args:
        :param kwargs:
        :return:
        """

        tsne_axis.imshow(self._fingerprint_adjacency, origin='upper')
        tsne_axis.grid('on')
        tsne_axis.set_title('Jaccard')

    def find_similar(self, point, n=9):
        """
        Find the n similar fingerprints based on the input point.

        :param point:
        :param n:
        :return:
        """

        row, col = int(point[0]), int(point[1])

        # find the Main fingerprint for this point in the image
        distances = self._fingerprint_adjacency[row]

        # Sort from highest to lowest.
        inds = np.argsort(distances)[::-1]

        log.debug('Closest indexes are {}'.format(inds[:n]))
        log.debug('Size of the fingerprint list {}'.format(len(self._fingerprints)))

        return [((row, ind), distances[ind], self._fingerprints[ind]) for ind in inds[:n]]

    #
    #  Utility Methods
    #

    def save(self):
        log.info('Returning the dictionary of information')
        return {
            'uuid': self._uuid,
            'similarity_type': self._similarity_type,
            'similarity': self._fingerprint_adjacency.tolist(),
            'fingerprint_uuids': [fp.uuid for fp in self._fingerprints],
            'parameters': {
            }
        }

    def load(self, thedict):
        log.info('Loading the dictionary of information')
        self._uuid = thedict['uuid']
        self._fingerprint_adjacency = np.array(thedict['similarity_type'])
        self._Y = np.array(thedict['similarity'])
        self._fingerprint_uuids = thedict['fingerprint_uuids']
        self._parameters = thedict['parameters']


class Distance(Similarity):
    """
    Based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
    """

    _similarity_type = 'distance'

    def __init__(self, metric='euclidean', *args, **kwargs):

        super(Distance, self).__init__(Distance._similarity_type, *args, **kwargs)

        log.info('Created {}'.format(self._similarity_type))

        # Each line / element in these should correpsond
        self._filename_index = []
        self._fingerprint_adjacency = None
        self._predictions = []

        if not isinstance(metric, (str)):
            raise TypeError('Metric needs to be a string')

        self._metric = metric

    @classmethod
    def is_similarity_for(cls, similarity_type):
        return cls._similarity_type == similarity_type

    def calculate(self, fingerprints):
        """
        Probably the best way to represent this is to calculate the Jaccard distance
        between every pair of points and represent it as an NxN matrix which can then be
        shown as an image.

        :param fingerprints:
        :return:
        """
        log.info('Going to calculate {} distance from {} fingerprints'.format(self._metric, len(fingerprints)))

        # Store as we need it for the find_nearest...
        self._fingerprints = fingerprints

        # Calculate the unique labels
        labels = []
        values = {}
        for ii, fp in enumerate(fingerprints):
            # Add to unique label list
            labels.extend([pred[1] for pred in fp.predictions if pred[1] not in labels])

            # Store the predictions for next processing
            values[ii] = fp.predictions

        unique_labels = list(set(labels))
        log.debug('Unique labels {}'.format(unique_labels))

        self._X = np.zeros((len(fingerprints), len(unique_labels)))
        for ii, fp in enumerate(fingerprints):
            inds = [unique_labels.index(prediction[1]) for prediction in values[ii]]
            self._X[ii][inds] = [prediction[2] for prediction in values[ii]]

        self._fingerprint_adjacency = squareform(pdist(self._X, metric=self._metric))

    def get_similarity(self):
        return Similarity('distance', self._fingerprint_adjacency.tolist(), [fp.uuid for fp in self._fingerprints])

    def display(self, tsne_axis):
        """
        Display the Y values in an axis element.

        :param axes:
        :param args:
        :param kwargs:
        :return:
        """

        tsne_axis.imshow(self._fingerprint_adjacency, origin='upper')
        tsne_axis.grid('on')
        tsne_axis.set_title('Distance [{}]'.format(self._metric))

    def find_similar(self, point, n=9):
        """
        Find similar given the input point.

        :param point:
        :param n:
        :return:
        """
        log.debug('Calling find_similar')

        row, col = int(point[0]), int(point[1])
        distances = self._fingerprint_adjacency[row]

        # TOOD: At this point this is assuming smallest distance is the best.
        inds = np.argsort(distances)

        return [((row, ind), distances[ind], self._fingerprints[ind]) for ind in inds[:n]]

    #
    #  Utility Methods
    #

    def save(self):
        log.info('Returning the dictionary of information')
        return {
            'uuid': self._uuid,
            'similarity_type': self._similarity_type,
            'similarity': self._fingerprint_adjacency.tolist(),
            'fingerprint_uuids': [fp.uuid for fp in self._fingerprints],
            'parameters': {
                'metric': self._metric
            }
        }

    def load(self, thedict):
        log.info('Loading the dictionary of information')
        self._uuid = thedict['uuid']
        self._fingerprint_adjacency = np.array(thedict['similarity_type'])
        self._Y = np.array(thedict['similarity'])
        self._fingerprint_uuids = thedict['fingerprint_uuids']
        self._parameters = thedict['parameters']
        self._metric = self._parameters['metric']
