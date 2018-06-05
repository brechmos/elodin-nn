import pickle

from transfer_learning.cutout import Cutout
from transfer_learning.database import get_database
from transfer_learning.similarity.display import SimilarityDisplay
from transfer_learning.similarity import Similarity

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')

with open('similarity_tsne.pck', 'rb') as fp:
    stsne = pickle.load(fp)

similarity_tsne = Similarity.factory(stsne)

sd = SimilarityDisplay(similarity_tsne)
