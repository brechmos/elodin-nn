from transfer_learning.cutout import Cutout
from transfer_learning.database import get_database
from transfer_learning.similarity import Similarity
from transfer_learning.similarity.display import SimilarityDisplay
import json

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')

with open('/tmp/orig.json', 'rt') as fp:
    similarity_tsne_dict = json.load(fp)

similarity_tsne = Similarity.factory(similarity_tsne_dict)
sd = SimilarityDisplay(similarity_tsne)
