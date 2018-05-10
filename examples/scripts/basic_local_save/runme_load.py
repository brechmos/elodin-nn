from tldist.cutout import Cutout
from tldist.similarity.similarity import Similarity
from tldist.database import get_database

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
similarities = [Similarity.similarity_factory(s) for s in similarities]
print(similarities)

# Grab the first on so we can work with it.
similarity_tsne = similarities[0]

# Load all the fingerprints from the similarity.
fingerprints = db.find('fingerprint', similarity_tsne.fingerprint_uuids)
print(fingerprints[:3])

# If we actually want to have Data instances we can do:
cutouts = []
for f in fingerprints[:5]:
    cutout = db.find('cutout', f['cutout_uuid'])
    cutouts.append(Cutout.cutout_factory(cutout))
print(cutouts[0])
