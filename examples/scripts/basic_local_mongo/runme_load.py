from tldist.similarity.similarity import Similarity
from tldist.database import get_database

from configparser import ConfigParser
config = ConfigParser()

config.read('config.ini')

# Create the database
print('Going to setup the database')
db = get_database(config['database']['type'], config['database']['hostname'])

# Load the similarities
similarities = db.find('similarity')
similarities = [Similarity.similarity_factory(s) for s in similarities]

print(similarities)

similarity_tsne = similarities[0]

fingerprints = db.find('fingerprint')
print(fingerprints[:3])
