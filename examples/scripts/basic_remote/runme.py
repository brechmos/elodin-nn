import os
import shutil
import pickle

from tldist.fingerprint.processing import FingerprintCalculatorResnet
from tldist.fingerprint.task import calculate_celery as calculate_fingerprints
from tldist.similarity.task import similarity_celery as calculate_similarity
from tldist.data import Data
from tldist.cutout.generators import FullImageCutoutGenerator
from tldist.database import get_database

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')


# Create the database
print('Going to setup the database in {}'.format(config['database']['filename']))

if os.path.isdir(config['database']['filename']):
    shutil.rmtree(config['database']['filename'])
db = get_database(config['database']['type'], config['database']['filename'])


def stringify(dictionary):
    return {k: str(v) for k, v in dictionary.items()}


fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

#
# Load the data
#
print('Loading the processing dict')
processing_dict = pickle.load(open('../../data/hubble_acs.pck', 'rb'))

#
#  Create the datasets
#
print('Creating data objects')
data = []
for fileinfo in processing_dict[:20]:
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    print('image is {}'.format(im.shape))
    if im.shape[0] < 224 or im.shape[1] < 224:
        continue
    db.save('data', im)
    data.append(im)

#
#  Create cutouts
#
print('Creating the cutouts for {} images'.format(len(data)))
full_cutout = FullImageCutoutGenerator(output_size=(224, 224))

cutouts = []
for datum in data:
    cutout = full_cutout.create_cutouts(datum)
    db.save('cutout', cutout)
    cutouts.append(cutout)

#
#  Calculate the fingerprints
#
print('Calculating the fingerprints for {} cutouts'.format(len(cutouts)))
fingerprints = calculate_fingerprints(cutouts, fc_save)
for fingerprint in fingerprints:
    db.save('fingerprint', fingerprint)

#
#  Compute the similarity
#
print('Calculating the similarity')
similarity_tsne = calculate_similarity(fingerprints, 'tsne')
db.save('similarity', similarity_tsne)

similarity_jaccard = calculate_similarity(fingerprints, 'jaccard')
db.save('similarity', similarity_jaccard)
