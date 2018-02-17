import numpy as np
from sklearn.manifold import TSNE

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("Similarity")
log.setLevel(logging.INFO)

# https://stackoverflow.com/questions/456672/class-factory-in-python


class Similarity:
    _similarity_type = ''

    def __init__(self, similarity_type, *args, **kwargs):
        self.similarity_type = similarity_type

    def _get(domain):
        for cls in Similarity.__subclasses__():
            if cls.is_similarity_for(domain):
                return cls()
        raise ValueError

    def calculate(self, files):
        raise NotImplementedError("Please Implement this method")

    def display(self):
        raise NotImplementedError("Please Implement this method")

    def save(self):
        raise NotImplementedError("Please Implement this method")


class tSNE(Similarity):

    _similarity_type = 'tsne'

    def __init__(self, *args, **kwargs):
        super(tSNE, self).__init__(tSNE._similarity_type, *args, **kwargs)

        log.info('Created {}'.format(__class__._similarity_type))

        # Each line / element in these should correpsond
        self._Y = None
        self._fingerprints = []
        self._filename_index = []
        self._distance_measure = 'l2'

        self._distance_measures = {
            'l2': lambda Y, point:  np.sqrt(np.sum((Y - np.array(point))**2, axis=1)),
            'l1': lambda Y, point: np.sum(np.abs((Y - np.array(point)), axis=1)),
        }

    @classmethod
    def is_similarity_for(cls, similarity_type):
        return cls._similarity_type == similarity_type

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
                except:
                    pass
        else:
            if distance_measure in self._distance_measures:
                self._distance_measure = distance_measure
            else:
                self._distance_measure = self._distance_measures.keys()[0]
                print('ERROR: No definition for {} so using {} instead.'.format(distance_measure, self._distance_measure))

    def calculate(self, fingerprints):
        log.info('Going to calculate tSNE from {} fingerprints'.format(len(fingerprints)))

        # Calculate the unique labels
        self._filename_index = np.array([fp['filename'] for fp in fingerprints])
        labels = []
        values = {}
        for ii, fp in enumerate(fingerprints):
            log.debug('    fingerprint is {}'.format(fp))
            # Add to unique label list
            labels.extend([pred[1] for pred in fp['predictions'] if pred[1] not in labels])

            # Store the predictions for next processing
            values[ii] = fp['predictions']

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

    def display(self, tsne_axis):
        """
        Display the Y values in an axis element.

        :param axes:
        :param args:
        :param kwargs:
        :return:
        """

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for ii, fi in enumerate(set(self._filename_index)):
            inds = np.nonzero(self._filename_index == fi)[0]
            tsne_axis.plot(self._Y[inds, 0], self._Y[inds, 1], '{}.'.format(colors[ii%len(colors)]))
        tsne_axis.grid('on')
        tsne_axis.set_title('tSNE [{}]'.format(self._distance_measure))

    def find_similar(self, point, n=9):

        distances = self._distance_measures[self._distance_measure](self._Y, point)
        inds = np.argsort(distances)
        log.debug('Closest indexes are {}'.format(inds[:n]))
        log.debug('Size of the fingerprint list {}'.format(len(self._fingerprints)))

        return [(distances[ind], self._fingerprints[ind]) for ind in inds[:n]]

class Jaccard(Similarity):

    _similarity_type = 'jaccard'

    def __init__(self, *args, **kwargs):
        super(Jaccard, self).__init__(*args, **kwargs)

        log.info('Created {}'.format(__class__._similarity_type))

        # Each line / element in these should correpsond
        self._fingerprints = []
        self._filename_index = []
        self._fingerprint_adjacency = None
        self._predictions = []

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
        log.info('Going to calculate Jaccard from {} fingerprints'.format(len(fingerprints)))

        from scipy.sparse import csc_matrix, csr_matrix
        import itertools

        self._fingerprints = fingerprints

        self._predictions = [set([tt[1] for tt in x['predictions']]) for x in fingerprints]

        # https://stackoverflow.com/questions/40579415/computing-jaccard-similarity-in-python
        up = list(set(list(itertools.chain(*[list(x) for x in self._predictions]))))

        A = np.zeros((len(self._predictions), len(up)))
        for ii, prediction in enumerate(self._predictions):
            indices = [up.index(x) for x in prediction]
            A[ii, indices] = 1.0

        sparse_adjacency = csc_matrix(A.transpose())
        self._fingerprint_adjacency = self.jaccard_similarities(sparse_adjacency).toarray().T


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
        tsne_axis.set_title('Jaccard')

    def jaccard_similarities(self, mat):
        # https://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html
        log.debug('Going into jaccard_similarities')

        cols_sum = mat.getnnz(axis=0)
        ab = mat.T * mat

        # for rows
        aa = np.repeat(cols_sum, ab.getnnz(axis=0))

        # for columns
        bb = cols_sum[ab.indices]

        similarities = ab.copy()
        similarities.data /= (aa + bb - ab.data)

        return similarities

    def find_similar(self, point, n=9):
        log.debug('Going into find_similar')

        row, col = int(point[0]), int(point[1])
        import time
        # find the Main fingerprint for this point in the image
        start = time.time()
        distances = self._fingerprint_adjacency[row]
        inds = np.argsort(distances)[::-1]
        log.debug('Closest indexes are {}'.format(inds[:n]))
        log.debug('Size of the fingerprint list {}'.format(len(self._fingerprints)))

        return [(distances[ind], self._fingerprints[ind]) for ind in inds[:n]]
