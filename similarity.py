import numpy as np
from itertools import chain
from sklearn.manifold import TSNE
from fingerprint import Fingerprint

import logging
logging.basicConfig(format='%(levelname)s: %(name)-10s %(asctime)-15s  %(message)s')
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

class tSNE(Similarity):

    _similarity_type = 'tsne'

    def __init__(self, *args, **kwargs):
        super(tSNE, self).__init__(tSNE._similarity_type, *args, **kwargs)

        log.info('Created {}'.format(__class__._similarity_type))

        # Each line / element in these should correpsond
        self._Y = None
        self._fingerprints = []

    @classmethod
    def is_similarity_for(cls, similarity_type):
        return cls._similarity_type == similarity_type

    def calculate(self, fingerprint_files):

        # Get the unique labels
        labels = []
        for filename in fingerprint_files:
            fp = Fingerprint.load(filename)
            labels.append([x[1] for x in fp.predictions])
        labels = list(set(chain.from_iterable(labels)))
        log.debug('Unique labels: {}'.format(labels))

        X = np.zeros((len(fingerprint_files), len(labels)))

        for ii, filename in enumerate(fingerprint_files):

            fp = Fingerprint.load(filename)

            for prediction in fp.predictions:
                pred_value = prediction[2]
                X[ii, labels.index(prediction[1])] = pred_value

            self._fingerprints.append(fp)

        log.debug('Fingerprint list {}'.format(self._fingerprints))

        # Compute the tSNE of the data.
        self._Y = TSNE(n_components=2).fit_transform(X)

    def displayY(self, axes):
        """
        Display the Y values in an axis element.
        
        :param axes: 
        :param args: 
        :param kwargs: 
        :return: 
        """
        axes.plot(self._Y[:, 0], self._Y[:, 1], '.')

    def find_similar(self, point, n=9):

        distances = np.sqrt(np.sum((self._Y-np.array(point))**2, axis=1))
        inds = np.argsort(distances)
        log.debug('Closest indexes are {}'.format(inds[:n]))
        log.debug('Size of the fingerprint list {}'.format(len(self._fingerprints)))

        log.info('Most similar finger prints {}'.format([self._fingerprints[ind] for ind in inds[:n]]))

        toreturn = []
        for ind in inds[:n]:
            toreturn.append([distances[ind], self._fingerprints[ind]])

        return toreturn


class Jaccard(Similarity):

    _similarity_type = 'jaccard'

    def __init__(self, *args, **kwargs):
        super(Jaccard, self).__init__(Jaccard._similarity_type, *args, **kwargs)

        log.info('Created {}'.format(__class__._similarity_type))

    @classmethod
    def is_similarity_for(cls, similarity_type):
        return cls._similarity_type == similarity_type
