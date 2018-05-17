import pickle

from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.cutout.generators import FullImageCutoutGenerator
from transfer_learning.similarity import calculate as similarity_calculate
from transfer_learning.data import Data
from transfer_learning.database import get_database

from configparser import ConfigParser
config = ConfigParser()

config.read('config.ini')

#
# Create the database
#
print('Going to setup the database')
db = get_database(config['database']['type'], config['database']['hostname'])

#
# Load the data
#
print('Loading the Hubble meta data and location information')
processing_dict = pickle.load(open('../../data/hubble_acs.pck', 'rb'))

print('Setting up the data structure required')
data = []
for fileinfo in processing_dict[:20]:
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    data.append(im)
    tt = db.save('data', im)
    print(im.uuid, tt)

#
#  Create cutouts
#
print('Creating the Full image cutout generator')
full_cutout = FullImageCutoutGenerator(output_size=(224, 224))

print('Going to create the cutouts')
cutouts = []
for datum in data:
    cutout = full_cutout.create_cutouts(datum)
    cutouts.append(cutout)

#
# Create the fingerprint calculator... fingerprint
#
print('Creating the info for the fingerprint calculator')
fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

print('Calculating the fingerprints')
fingerprints = fingerprint_calculate(cutouts, fc_save)
print([fp.uuid for fp in fingerprints])
[db.save('fingerprint', x) for x in fingerprints]

#
#  Comptue the similarity metrics
#
print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')
db.save('similarity', similarity_tsne)

print('Calculating the Jaccard similarity')
similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
db.save('similarity', similarity_jaccard)
