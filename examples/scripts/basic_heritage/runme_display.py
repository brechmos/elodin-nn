import pickle

from transfer_learning.similarity import Similarity
from transfer_learning.similarity.display import SimilarityDisplay

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')

#
# Load up the pickle file.
#

with open('similarity_tsne.pck', 'rb') as fp:
    stsne = pickle.load(fp)
    similarity_tsne = Similarity.factory(stsne)

#
# Run the display program
#

sd = SimilarityDisplay(similarity_tsne)
