from transfer_learning.similarity.similarity import Similarity
from transfer_learning.database import get_database

from configparser import ConfigParser
config = ConfigParser()

config.read('config.ini')

# Create the database
print('Going to setup the database')
db = get_database(config['database']['type'], config['database']['hostname'])

# Load the similarities
similarities = db.find('similarity')
print(similarities)
similarity_tsne = similarities[0]

fingerprints = db.find('fingerprint')
print(fingerprints[:3])
