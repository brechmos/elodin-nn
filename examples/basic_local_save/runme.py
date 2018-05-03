import uuid
import pickle
import os
import shutil

from tldist.fingerprint.processing import FingerprintResnet
from tldist.fingerprint.processing import calculate as fingerprint_calculate
from tldist.similarity.processing import calculate as similarity_calculate
from tldist.database import get_database


def stringify(dictionary):
    return {k: str(v) for k, v in dictionary.items()}


# Create the database
DB_LOC = '/tmp/mydb'
print('Going to setup the database in {}'.format(DB_LOC))

if os.path.isdir(DB_LOC):
    shutil.rmtree(DB_LOC)
db = get_database('blitzdb', DB_LOC)


# Load the data
print('Loading the Hubble meta data and location information')
filename_prepend = 'http://18.218.192.161:4123/ACSimages/'
processing_dict = pickle.load(open('../data/hubble_acs.pck', 'rb'))

print('Setting up the data structure required')
data = []
for fileinfo in processing_dict[:20]:
    im = {
        'uuid': str(uuid.uuid4()),
        'location': fileinfo['location'],
        'radec': fileinfo['radec'],
        'meta': stringify(fileinfo['meta'])
    }
    data.append(im)
    db.save('data', im)

# Create the fingerprint calculator... fingerprint
print('Creating the info for the fingerprint calculator')
fresnet = FingerprintResnet()
fc_save = fresnet.save()

print('Calculating the fingerprints')
fingerprints = fingerprint_calculate(data, fc_save)
print([x['uuid'] for x in fingerprints])
[db.save('fingerprint', x) for x in fingerprints]

print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')
db.save('similarity', similarity_tsne)

print('Calculating the Jaccard similarity')
similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
db.save('similarity', similarity_tsne)
