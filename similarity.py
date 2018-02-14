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
        pass

    def display(self):
        pass

    def save(self):
        pass


class tSNE(Similarity):

    _similarity_type = 'tsne'

    def __init__(self, *args, **kwargs):
        super(tSNE, self).__init__(tSNE._similarity_type, *args, **kwargs)

        log.info('Created {}'.format(__class__._similarity_type))

        # Each line / element in these should correpsond
        self._Y = None
        self._fingerprints = []
        self._filename_index = []

    @classmethod
    def is_similarity_for(cls, similarity_type):
        return cls._similarity_type == similarity_type

    def calculate(self, fingerprints):
        log.info('Going to calculate tSNE from {} fingerprints'.format(len(fingerprints)))

        # Calculate the unique labels
        self._filename_index = np.array([fp['filename'] for fp in fingerprints])
        print(self._filename_index)
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

    def displayY(self, axes):
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
            axes.plot(self._Y[inds, 0], self._Y[inds, 1], '{}.'.format(colors[ii%len(colors)]))

    def find_similar(self, point, n=9):

        distances = np.sqrt(np.sum((self._Y - np.array(point)) ** 2, axis=1))
        inds = np.argsort(distances)
        log.debug('Closest indexes are {}'.format(inds[:n]))
        log.debug('Size of the fingerprint list {}'.format(len(self._fingerprints)))

        return [(distances[ind], self._fingerprints[ind]) for ind in inds[:n]]
