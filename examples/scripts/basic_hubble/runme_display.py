import pickle
from transfer_learning.similarity import Similarity
from transfer_learning.similarity.qtdisplay import SimilarityDisplay

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')

#
# Load the data
#

with open('similarity_tsne.pck', 'rb') as fp:
    similarity_tsne_dict = pickle.load(fp)
    similarity_tsne = Similarity.factory(similarity_tsne_dict)

sd = SimilarityDisplay(similarity_tsne)
