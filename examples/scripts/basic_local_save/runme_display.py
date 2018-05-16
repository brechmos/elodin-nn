from tldist.cutout import Cutout
from tldist.database import get_database
from tldist.similarity.display import SimilarityDisplay

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')

# Set the database
print('Going to setup the database in {}'.format(config['database']['filename']))
db = get_database(config['database']['type'], config['database']['filename'])

# Load the similarities
print('Loading up the similarity data from the database')
similarities = db.find('similarity')

# Currently the similarities is a list of dictionaries where each dictionary
# represents a similarity.  So, best is to load into a Similarity
# instance.
print(similarities)

# Grab the first on so we can work with it.
similarity_tsne = similarities[1]

sd = SimilarityDisplay(similarity_tsne, db)
