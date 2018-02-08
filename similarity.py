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


        labels = []
        values = {}
        for ii, filename in enumerate(fingerprint_files):
            fp = Fingerprint.load(filename)

            # Add to unique label list
            labels.extend([pred[1] for pred in fp.predictions if pred[1] not in labels])

            # Store the predictions for next processing
            values[ii] = fp.predictions

            self._fingerprints.append(fp)

        log.info('Unique labels {}'.format(labels))

        X = np.zeros((len(fingerprint_files), len(labels)))
        for ii, filename in enumerate(fingerprint_files):
            inds = [labels.index(prediction[1]) for prediction in values[ii]]
            X[ii][inds] = [prediction[2] for prediction in values[ii]]

        log.debug('Fingerprint list {}'.format(self._fingerprints))

        # Compute the tSNE of the data.
        log.info('Calculating the tSNE...')
        self._Y = TSNE(n_components=2).fit_transform(X)
        log.info('Done calculation')

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
