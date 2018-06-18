import ast
import operator
import weakref
import uuid
import itertools

import numpy as np
from sklearn.manifold import TSNE
from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist, squareform
from transfer_learning.fingerprint import Fingerprint

from ..tl_logging import get_logger
import logging
log = get_logger('similarity', level=logging.DEBUG)


def calculate(fingerprints, similarity_calculator, serialize_output=False):
    """
    This function might be called locally and in that case we want to return the
    actual similarity calculator instance.  Or it might be run rmeotely (via celery)
    and in this case we want to reutrn the serialized version of the similarity instance.

    Parameters
    ----------
    fingerprints : list
       List of fingerprint objects to which the similarity is calculated.
    similarity_calculator : str
       String representation of the similarity calculator ('tsne', 'jaccard', 'distance')

    Returns
    -------
    Similarity Object
        The similariity object instantiated based on the string representation.
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

        for sc in Similarity.__subclasses__():
            print('Comparing {} {}'.format(sc._similarity_type, thedict['similarity_type']))
            if sc._similarity_type == thedict['similarity_type']:
                print('          same')
                sim = sc()
                sim.load(thedict)

        return sim

    def __init__(self, similarity_type=None, similarity=None, fingerprint_uuids=None, uuid_in=None):
        """
        This function might be called locally and in that case we want to return the
        actual similarity calculator instance.  Or it might be run rmeotely (via celery)
        and in this case we want to reutrn the serialized version of the similarity instance.
    
        Parameters
        ----------
        similarity_type : list
           String representation of the similarity calculator ('tsne', 'jaccard', 'distance')
        similarity : str
           String representation of the similarity calculator ('tsne', 'jaccard', 'distance')
        fingerprint_uuids : list
           List of fingerprint_uuids.
        uuid_in : str
           Unique identifier for this instance of the similarity calculator.
    
        Returns
        -------
        Similarity Object
            The similariity object instantiated based on the string representation.
        """
        if uuid_in is None:
            self._uuid = str(uuid.uuid4())
        else:
            self._uuid = uuid_in
        self._similarity_type = similarity_type
        self._similarity = similarity

        if fingerprint_uuids is None:
            self._fingerprint_uuids = []
        else:
            self._fingerprint_uuids = fingerprint_uuids

        # TODO: Ck to see why this is not set through the parameters
        self._parameters = {}

        self._fingerprint_filter = None

        self._similarity_collection[self._uuid] = self

        # These are the inds that will be used in the return
        # If empty, then send all.
        self._fingerprint_filter_inds = []

    def __str__(self):
        return 'Similarity {} based on {}...'.format(
                self._similarity_type, self._fingerprint_uuids[:3])

    def __repr__(self):
        return self.__str__()

    #
    # Setters and getters
    #

    @property
    def fingerprint_filter(self):
        return self._fingerprint_filter

    @fingerprint_filter.setter
    def fingerprint_filter(self, value):
        self._fingerprint_filter = value

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

    def set_filter_fingerprints(self, thefilter):
        """
        Return a list of all indices that match the string items
        in the list.

        Currently this is just an inclusive thing but in the future
        might need to change to have exclusion as well.

        Parameters
        -----------
        thelist : list
            List of strings of things to look for in the meta.

        overlapping_bounding_boxes : bool
            Allow overlapping bounding boxes

        """
        log.info('filter is {}'.format(thefilter))

        #
        # If nothing in the list, then show everything
        #

        if len(thefilter) == 0:
            self._fingerprint_filter_inds = list(range(len(self._fingerprints)))

        #
        # Otherwise how a subset.
        #

        else:
            # This is the meat of the filtering, and rgith now just looks to see
            # if any of the string elements in "thelist" are found within some
            # subset of the value of the meta in data.

            metas = [dict(list(x.cutout.data.meta.items())+[('id_number', ii), ('predictions', x.predictions)])
                     for ii, x in enumerate(self._fingerprints)]

            filtered_metas = self._eval_expr(thefilter, metas)
            self._fingerprint_filter_inds = [x['id_number'] for x in filtered_metas]

        return len(self._fingerprint_filter_inds)
    #
    # Next section are methods for parsing the fingerprint meta and predictions
    # for updating the display.
    #

    def _eval_expr(self, expr, tdict):
        return self.eval_(ast.parse(expr, mode='eval').body, tdict)

    def dp(self, meta, key):
        """
        This method determines if the key is a prediction query
        or just a regular meta lookup.
        """

        # Prediction query
        if key.startswith('π'):
            _, k = key.split('π')
            return next((p[2] for p in meta['predictions'] if p[1] == k), None)

        # Meta dictionary lookup
        else:
            return meta[key]

    def bounding_boxes_overlap(self, fingerprint1, fingerprint2):
        """
        Check to see if two bounding boxes overlap.

        Parameters
        -----------
        fingerprint1 : Fingerprint
            The first fingeprint.

        fingerprint2 : Fingerprint
            The second fingerprint.

        Return
        ------
        overlaps: bool
            Returns True/False depending on overlap

        """

        bb1 = fingerprint1.cutout.bounding_box
        bb2 = fingerprint2.cutout.bounding_box

        hoverlaps = True
        voverlaps = True
        if (bb1[0] > bb2[1]) or (bb1[1] < bb2[0]):
            hoverlaps = False
        if (bb1[3] < bb2[2]) or (bb1[2] > bb2[3]):
            voverlaps = False
        return hoverlaps and voverlaps

    def _compare(self, op, node, tdict):
        tocompare = self.eval_(node.comparators[0], tdict)
        if isinstance(tocompare, (int, float)):
            cast = float
        else:
            cast = str
        return [x for x in tdict
                if(self.dp(x, str(node.left.id)) is not None and
                   op(cast(self.dp(x, str(node.left.id))), cast(self.eval_(node.comparators[0], tdict)))
                   )
                ]

    def eval_(self, node, tdict):
        """
        Recursive node evaulator that was modeled after some code on the
        internets.

        Parameters
        -----------
        node : ast.node.Node
            A Python AST node.

        tdict : List of dicts
            List of dictionaries that contain the fingerprint info.

        Raises
        ------
        TypeError
            If there is an error in the parsing of the original query.

        """

        #
        # Number
        #

        if isinstance(node, ast.Num):  # <number>
            return node.n

        #
        #  String
        #

        elif isinstance(node, ast.Str):  # <operator> <operand> e.g., -1
            return node.s

        #
        #  Comparators i.e., > < >= <= ==
        #

        elif isinstance(node, ast.Compare):  # <operator> <operand> e.g., -1
            if isinstance(node.ops[0], ast.Eq):
                return [x for x in tdict if x[str(node.left.id)] == self.eval_(node.comparators[0], tdict)]

            elif isinstance(node.ops[0], ast.Gt):
                return self._compare(operator.gt, node, tdict)

            elif isinstance(node.ops[0], ast.Lt):
                return self._compare(operator.lt, node, tdict)

            elif isinstance(node.ops[0], ast.GtE):
                return self._compare(operator.gte, node, tdict)

            elif isinstance(node.ops[0], ast.LtE):
                return self._compare(operator.lte, node, tdict)

            elif isinstance(node.ops[0], ast.In):
                return [x for x in tdict if str(node.left.s) in x[node.comparators[0].id]]

        #
        #  Boolean operators i.e.,  And and Or
        #

        elif isinstance(node, ast.BoolOp):  # <operator> <operand> e.g., -1

            #
            # And
            #  Creating sets would be great, but dictionaries aren't hashable
            #  so had to do this based on an 'id_number' being injected into the dicts
            #

            if isinstance(node.op, ast.And):
                """
                Dictionaries must be injected with a key 'id_number' that has a unique value.
                """

                #
                # Create a list of lists of id_number
                #

                id_numbers = []
                for ii in range(len(node.values)):
                    li = self.eval_(node.values[ii], tdict)
                    id_numbers.append(set([x['id_number'] for x in li]))

                #
                #  Now need to determine the ones common in each
                #

                u = set.intersection(*id_numbers)

                return [x for x in li if x['id_number'] in u]

            #
            # Or
            #

            if isinstance(node.op, ast.Or):
                id_numbers = set()
                toreturn = []
                for nv in node.values:
                    li = self.eval_(nv, tdict)
                    toreturn = toreturn + [x for x in li 
                                           if x['id_number'] not in id_numbers and not id_numbers.add(x['id_number'])]
                return toreturn

        else:
            raise TypeError(node)


class tSNE(Similarity):

    _similarity_type = 'tsne'

    def __init__(self, *args, **kwargs):
        """
        This function might be called locally and in that case we want to return the
        actual similarity calculator instance.  Or it might be run rmeotely (via celery)
        and in this case we want to reutrn the serialized version of the similarity instance.

        Parameters
        ----------
        display_type : string
           String representation of the display, can be 'plot', 'hexbin'.

        Returns
        -------
        N/A
        """

        # Pull the display type out of kwargs if it is there. If not then we will
        # use 'plot' as the default.
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
        self._display_type = display_type
        self._display_types = ['plot', 'hexbin', 'mpl']
        if self._display_type not in self._display_types:
            raise Exception('Display type {} not one of {}'.format(
                self._display_type, self._display_types))

        # Define the distance measures. 
        self._distance_measures = {
            'l2': lambda Y, point: np.sqrt(np.sum((Y - np.array(point))**2, axis=1)),
            'l1': lambda Y, point: np.sum(np.abs((Y - np.array(point)), axis=1)),
        }

    @property
    def data(self):
        return self._Y

    @property
    def data_filtered(self):
        return self._Y[self._fingerprint_filter_inds]

    #
    #  Calculation Methods
    #

    def calculate(self, fingerprints):
        """
        Calculate the TSNE based on the fingerprints.

        Parameters
        ----------
        fingerprints : list of Fingerprint instances
           The fingerprints we want to calculate over.

        Returns
        -------
        N/A
        """
        log.info('Going to calculate tSNE from {} fingerprints'.format(len(fingerprints)))

        #
        #  Filter the fingerprints, if the filter is set.
        #

        if self._fingerprint_filter is not None:
            fingerprints = self._fingerprint_filter(fingerprints)

        #
        # Calculate the unique labels
        #

        labels = []
        values = {}
        for ii, fp in enumerate(fingerprints):
            log.debug('    fingerprint is {}'.format(fp))

            #
            # Add to unique label list
            #

            labels.extend([pred[1] for pred in fp.predictions if pred[1] not in labels])

            #
            # Store the predictions for next processing
            #

            values[ii] = fp.predictions

            self._fingerprints.append(fp)
        log.info('Unique labels {}'.format(labels))

        #
        # Set up the similarity matrix X based on the predictions
        #

        X = np.zeros((len(fingerprints), len(labels)))
        for ii, fp in enumerate(fingerprints):
            inds = [labels.index(prediction[1]) for prediction in values[ii]]
            X[ii][inds] = [prediction[2] for prediction in values[ii]]

        log.debug('X is {}'.format(X))
        log.debug('Fingerprint list {}'.format(self._fingerprints))

        #
        # Compute the tSNE of the data.
        #

        log.info('Calculating the tSNE...')
        self._Y = TSNE(n_components=2).fit_transform(X)
        log.debug('self._Y is {}'.format(self._Y))
        log.info('Done calculation')

    #
    #  Utility Methods
    #

    def save(self):
        """
        Save function converts the instance to a dict.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary representation of this instance.
        """
        log.info('Returning the dictionary of information')
        return {
            'uuid': self._uuid,
            'similarity_type': self._similarity_type,
            'similarity': self._Y.tolist(),
            'fingerprint': [fp.save() for fp in self._fingerprints],
            'parameters': {
                'distance_measure': self._distance_measure
            }
        }

    def load(self, thedict, db=None):
        """
        Reload the internal variables from the dictionary.

        Parameters
        ----------
        thedict : dict
            The first parameter.
        db : str
            database object

        Returns
        -------
        N/A
        """
        log.info('Loading the dictionary of information with database {}'.format(db))

        self._uuid = thedict['uuid']
        self._similarity_type = thedict['similarity_type']
        self._Y = np.array(thedict['similarity'])
        self._fingerprints = [Fingerprint.factory(x) for x in thedict['fingerprint']]
        self._parameters = thedict['parameters']
        self._distance_measure = self._parameters['distance_measure']

        self._fingerprint_filter_inds = list(range(len(self._fingerprints)))

    #
    #  Display methods
    #

    def set_display_type(self, display_type):
        """
        Set the display type. Currently 'plot', 'hexbin', and 'mpl' are defined.

        Parameters
        ----------
        display_type : string
            Display type: 'plot', 'hexbin', and 'mpl' are defined.

        Returns
        -------
        N/A
        """
        if display_type in self._display_types:
            self._display_type = display_type
        else:
            raise ValueError('Display type {} not in {}'.format(display_type, self._display_types))

    def select_distance_measure(self, distance_measure=None):
        """
        Select the distance measure.

        Parameters
        ----------
        distance_measure : string
            Display type: 'plot', 'hexbin', and 'mpl' are defined.

        Returns
        -------
        N/A
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
                log.error('ERROR: No definition for {} so using {} instead.'.format(
                    distance_measure, self._distance_measure))

    def display(self, tsne_axis):
        """
        Display the plot into the matplotlib axis in the
        parameter based on the plot type. This just determines
        the plot type and then calls the internal plot function.

        Parameters
        ----------
        tsne_axis : Matplotlib.axes.axis
            The matplotlib axis into which we want to display the plot.

        Returns
        -------
        N/A
        """
        if self._display_type == 'plot':
            self._display_plot(tsne_axis)
        elif self._display_type == 'hexbin':
            return self._display_hexbin(tsne_axis)
#        elif self._display_type == 'mpl':
#            self._display_mpl(tsne_axis)
        else:
            raise ValueError('Plot type {} is not in the valid list {}'.format(
                self._display_type, self._display_types))

    def _display_plot(self, tsne_axis):
        """
        Display the plot into the matplotlib axis as a regular scatter plot.

        Parameters
        ----------
        tsne_axis : Matplotlib.axes.axis
            The matplotlib axis into which we want to display the plot.

        Returns
        -------
        N/A
        """
        tsne_axis.plot(self._Y[:, 0], self._Y[:, 1])#, '.')
        tsne_axis.grid('on')
        tsne_axis.set_title('tSNE [{}]'.format(self._distance_measure))

    def _display_hexbin(self, tsne_axis):
        """
        Display the plot into the matplotlib axis as a hexbin.

        Parameters
        ----------
        tsne_axis : Matplotlib.axes.axis
            The matplotlib axis into which we want to display the plot.

        Returns
        -------
        N/A
        """
        output = tsne_axis.hexbin(self._Y[:, 0], self._Y[:, 1], cmap='hot')
        tsne_axis.grid('on')
        tsne_axis.set_title('tSNE [{}]'.format(self._distance_measure))

        # Set the color limits so that it is a little brighter
        limmax = np.percentile(output.get_array(), 99.9)
        output.set_clim((0, limmax))

        return output

    def find_similar(self, point, n=9, allow_overlapping_bounding_boxes=True):
        """
        Find fingerprints that are close to the input point.

        Parameters
        ----------
        point : tuple (int, int)
            A point in the plot.
        n : int
            Number to return.

        allow_overlapping_bounding_boxes: bool
            Whether to allow overlapping bb or not.

        Returns
        -------
        list
            List of dicts that describe the closest fingerprints.
        """
        log.info('')

        if self._fingerprint_filter_inds is None:
            self._fingerprint_filter_inds = list(range(len(self._fingerprints)))

        distances = self._distance_measures[self._distance_measure](self._Y, point)

        log.debug('Filtering based, n distances {}  n filter_inds {}'.format(
                  len(distances), len(self._fingerprint_filter_inds)))

        inds = []
        for ind in np.argsort(distances):

            # First, make sure this index is one of the filtered ones.
            if ind in self._fingerprint_filter_inds:

                # Next check to see if we allow overlapping bounding boxes
                # If not, make sure this one doesn't overlap with any in the list so far.
                if(allow_overlapping_bounding_boxes or
                   not any([self.bounding_boxes_overlap(self._fingerprints[ind], self._fingerprints[ii]) for ii in inds])):
                    inds.append(ind)
                    if len(inds) == n:
                        break

        # Now we want to look only in the "search_inds" if that is passed in
        return [{
                    'tsne_point': self._Y[ind],
                    'distance': distances[ind],
                    'fingerprint': self._fingerprints[ind]
                } for ind in inds[:n]]

    def cutout_point(self, cutout):
        """
        Given a cutout (and therefore a fingerprint), find the point in the
        tSNE plot that it corresponds to.

        Parameters
        -----------
        cutout : Cutout
            The cutout we want to find.

        Return
        ------
        tSNE point: tuple
            Point in the tSNE space the cutout corresponds to
        """
        log.info('cutout {}'.format(cutout))

        index = [fingerprint.cutout.uuid for fingerprint in self._fingerprints].index(cutout.uuid)
        return self._Y[index]

    def closest_cutout(self, data, point):
        """
        Given a cutout (and therefore a fingerprint), find the point in the
        tSNE plot that it corresponds to.

        Parameters
        -----------
        cutout : Cutout
            The cutout we want to find.

        Return
        ------
        tSNE point: tuple
            Point in the tSNE space the cutout corresponds to
        """
        log.info('data {}  point {}'.format(data, point))

        #
        # Get the cutouts assocated with the data passed in.
        #

        cutouts = [fingerprint.cutout for fingerprint in self._fingerprints if fingerprint.cutout.data == data]

        #
        # Compute distance between cutout bounding boxes centers and the point.
        #

        distances = [self._bounding_box_distance(c.bounding_box, point) for c in cutouts]

        #
        # Find the smallest.
        #

        index = np.argsort(distances)[0]

        log.debug('Closest cutout is with bb {} and dist {}'.format(cutouts[index].bounding_box, distances[index]))

        return cutouts[index]

    def _bounding_box_distance(self, bounding_box, point):
        log.info('bb {} point {}'.format(bounding_box, point))
        center = ((bounding_box[0]+bounding_box[1])/2.0, (bounding_box[2]+bounding_box[3])/2.0)
        distance = np.sqrt((center[0]-point[1])**2+(center[1]-point[0])**2)
        log.debug('    distance is {}'.format(distance))
        return distance

class Jaccard(Similarity):
    """
    Jaccard similarity implementation.  The Jaccard similarity is a set based
    method, which, implemented here, is computes the similiarity of two cutouts
    based on only the names of the imagenet images in the fingerprint for
    each.  So, given two cutouts C1 and C2 and their corresponding finger print
    imagenet sets S1 and S2.  The Jaccard similarity between the two is computed
    as len( S1 & S2 ) / len( S1 | S2 ).  The higher the number, the
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

        # Top n_predictions to use in the jaccard comparison
        # TODO: This might be good to be a funciton of # of fingerprints (?)
        self._n_predictions = 10

    @property
    def data(self):
        return self._fingerprint_adjacency

    @property
    def data_filtered(self):
        if self._fingerprint_filter_inds:
            return self._fingerprint_adjacency[self._fingerprint_filter_inds, :][:, self._fingerprint_filter_inds]
        else:
            return self._fingerprint_adjacency

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

        self._predictions = [set([tt[1] for tt in fp.predictions[:self._n_predictions]]) for fp in fingerprints]

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

        tsne_axis._axes.get_figure().canvas.blit(tsne_axis._axes.bbox)

    def find_similar(self, point, n=9, allow_overlapping_bounding_boxes=True):
        """
        Find fingerprints that are close to the input point.

        Parameters
        ----------
        point : tuple (int, int)
            A point in the plot.
        n : int
            Number to return.

        allow_overlapping_bounding_boxes: bool
            Whether to allow overlapping bb or not.

        Returns
        -------
        list
            List of dicts that describe the closest fingerprints.
        """
        log.info('')

        if self._fingerprint_filter_inds is None:
            self._fingerprint_filter_inds = list(range(len(self._fingerprints)))

        row, col = int(point[0]), int(point[1])

        # The point passed in is based directly off the image and
        # therefore we will need to map the point back to the full set.
        row = self._fingerprint_filter_inds[row]

        # find the Main fingerprint for this point in the image
        distances = self._fingerprint_adjacency[row]
        log.debug('length of dsistances is {}'.format(len(distances)))

        inds = []
        for ind in np.argsort(distances):

            # First, make sure this index is one of the filtered ones.
            if ind in self._fingerprint_filter_inds:

                # Next check to see if we allow overlapping bounding boxes
                # If not, make sure this one doesn't overlap with any in the list so far.
                if(allow_overlapping_bounding_boxes or
                   not any([self.bounding_boxes_overlap(self._fingerprints[ind], self._fingerprints[ii]) for ii in inds])):
                    inds.append(ind)
                    if len(inds) == n:
                        break

        log.debug('Closest indexes are {}'.format(inds[:n]))
        log.debug('Size of the fingerprint list {}'.format(len(self._fingerprints)))

        return [{
                    'tsne_point': self._fingerprint_adjacency[ind],
                    'distance': distances[ind],
                    'fingerprint': self._fingerprints[ind]
                } for ind in inds[:n]]

    def cutout_point(self, cutout):
        """
        Given a cutout (and therefore a fingerprint), find the point in the
        tSNE plot that it corresponds to.

        Parameters
        -----------
        cutout : Cutout
            The cutout we want to find.

        Return
        ------
        tSNE point: tuple
            Point in the tSNE space the cutout corresponds to
        """
        log.info('cutout {}'.format(cutout))

        index = [fingerprint.cutout.uuid for fingerprint in self._fingerprints].index(cutout.uuid)
        return (index,0)

    def closest_cutout(self, data, point):
        """
        Given a cutout (and therefore a fingerprint), find the point in the
        tSNE plot that it corresponds to.

        Parameters
        -----------
        cutout : Cutout
            The cutout we want to find.

        Return
        ------
        tSNE point: tuple
            Point in the tSNE space the cutout corresponds to
        """
        log.info('data {}  point {}'.format(data, point))

        #
        # Get the cutouts assocated with the data passed in.
        #

        cutouts = [fingerprint.cutout for fingerprint in self._fingerprints if fingerprint.cutout.data == data]

        #
        # Compute distance between cutout bounding boxes centers and the point.
        #

        distances = [self._bounding_box_distance(c.bounding_box, point) for c in cutouts]

        #
        # Find the smallest.
        #

        index = np.argsort(distances)[0]

        log.debug('Closest cutout is with bb {} and dist {}'.format(cutouts[index].bounding_box, distances[index]))

        return cutouts[index]

    def _bounding_box_distance(self, bounding_box, point):
        log.info('bb {} point {}'.format(bounding_box, point))
        center = ((bounding_box[0]+bounding_box[1])/2.0, (bounding_box[2]+bounding_box[3])/2.0)
        distance = np.sqrt((center[0]-point[1])**2+(center[1]-point[0])**2)
        log.debug('    distance is {}'.format(distance))
        return distance

    #
    #  Utility Methods
    #

    def save(self):
        log.info('Returning the dictionary of information')
        return {
            'uuid': self._uuid,
            'similarity_type': self._similarity_type,
            'similarity': self._fingerprint_adjacency.tolist(),
            'fingerprint': [fp.save() for fp in self._fingerprints],
            'parameters': {
            }
        }

    def load(self, thedict, db=None):
        log.info('Loading the dictionary of information')
        self._uuid = thedict['uuid']
        self._similarity_type = thedict['similarity_type']
        self._fingerprint_adjacency = np.array(thedict['similarity'])
        self._fingerprints = [Fingerprint.factory(x) for x in thedict['fingerprint']]
        self._parameters = thedict['parameters']

        self._fingerprint_filter_inds = list(range(len(self._fingerprints)))


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

    @property
    def data(self):
        return self._fingerprint_adjacency

    @property
    def data_filtered(self):
        return self._fingerprint_adjacency[:, self._fingerprint_filter_inds][self._fingerprint_filter_inds, :]

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

    def find_similar(self, point, n=9, allow_overlapping_bounding_boxes=True):
        """
        Find fingerprints that are close to the input point.

        Parameters
        ----------
        point : tuple (int, int)
            A point in the plot.
        n : int
            Number to return.

        allow_overlapping_bounding_boxes: bool
            Whether to allow overlapping bb or not.

        Returns
        -------
        list
            List of dicts that describe the closest fingerprints.
        """
        log.info('')

        if self._fingerprint_filter_inds is None:
            self._fingerprint_filter_inds = list(range(len(self._fingerprints)))

        row, col = int(point[0]), int(point[1])

        # The point passed in is based directly off the image and
        # therefore we will need to map the point back to the full set.
        row = self._fingerprint_filter_inds[row]

        # find the Main fingerprint for this point in the image
        distances = self._fingerprint_adjacency[row]
        log.debug('length of dsistances is {}'.format(len(distances)))

        inds = []
        for ind in np.argsort(distances):

            # First, make sure this index is one of the filtered ones.
            if ind in self._fingerprint_filter_inds:

                # Next check to see if we allow overlapping bounding boxes
                # If not, make sure this one doesn't overlap with any in the list so far.
                if(allow_overlapping_bounding_boxes or
                   not any([self.bounding_boxes_overlap(self._fingerprints[ind], self._fingerprints[ii]) for ii in inds])):
                    inds.append(ind)
                    if len(inds) == n:
                        break

        return [{
                    'tsne_point': self._fingerprint_adjacency[ind],
                    'distance': distances[ind],
                    'fingerprint': self._fingerprints[ind]
                } for ind in inds[:n]]

    #
    #  Utility Methods
    #

    def save(self):
        log.info('Returning the dictionary of information')
        return {
            'uuid': self._uuid,
            'similarity_type': self._similarity_type,
            'similarity': self._fingerprint_adjacency.tolist(),
            'fingerprint': [fp.save() for fp in self._fingerprints],
            'parameters': {
                'metric': self._metric
            }
        }

    def load(self, thedict, db=None):
        log.info('Loading the dictionary of information')
        self._uuid = thedict['uuid']
        self._similarity_type = thedict['similarity_type']
        self._fingerprint_adjacency = np.array(thedict['similarity'])
        self._fingerprints = [Fingerprint.factory(x) for x in thedict['fingerprint']]
        self._parameters = thedict['parameters']
        self._metric = self._parameters['metric']

        self._fingerprint_filter_inds = list(range(len(self._fingerprints)))
