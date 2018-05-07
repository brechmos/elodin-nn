import uuid
import pickle
import os
import shutil

from tldist.fingerprint.processing import FingerprintCalculatorResnet
from tldist.fingerprint.processing import calculate as fingerprint_calculate
from tldist.similarity.similarity import calculate as similarity_calculate
from tldist.data.data import Data
from tldist.similarity.similarity import Similarity
from tldist.database import get_database



# Create the database
DB_LOC = '/tmp/mydb'
print('Going to setup the database in {}'.format(DB_LOC))

db = get_database('blitzdb', DB_LOC)

# Load the similarities
similarities = db.find('similarity')
similarities = [Similarity.similarity_factory(s) for s in similarities]

print(similarities)

similarity_tsne = similarities[0]

fingerprints = db.find('fingerprint')
similarity_fingerprints = [fuuid for fuuid in similarity_tsne.fingerprint__uuids]

print(fingerprints[:3])
