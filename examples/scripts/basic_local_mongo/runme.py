import uuid
import pickle
import os
import shutil

from tldist.fingerprint.processing import FingerprintCalculatorResnet
from tldist.fingerprint.processing import calculate as fingerprint_calculate
from tldist.similarity.processing import calculate as similarity_calculate
from tldist.data.data import Data
from tldist.database import get_database



# Create the database
print('Going to setup the database')
db = get_database('mongo', 'localhost')

# Load the data
print('Loading the Hubble meta data and location information')
processing_dict = pickle.load(open('../data/hubble_acs.pck', 'rb'))

print('Setting up the data structure required')
data = []
for fileinfo in processing_dict[:20]:
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    data.append(im)
    tt = db.save('data', im.save())
    print(im.uuid, tt)

# Create the fingerprint calculator... fingerprint
print('Creating the info for the fingerprint calculator')
fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

print('Calculating the fingerprints')
fingerprints = fingerprint_calculate(data, fc_save)
print([fp.uuid for fp in fingerprints])
[db.save('fingerprint', x.save()) for x in fingerprints]

print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')
db.save('similarity', similarity_tsne.save())

print('Calculating the Jaccard similarity')
similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
db.save('similarity', similarity_jaccard.save())
