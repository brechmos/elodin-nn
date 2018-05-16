import pickle
import os
import shutil
import numpy as np

from tldist.fingerprint.processing import FingerprintCalculatorResnet
from tldist.fingerprint.processing import calculate as fingerprint_calculate
from tldist.similarity.similarity import calculate as similarity_calculate
from tldist.data import Data
from tldist.data.processing import GrayScale as DataGrayScale
from tldist.cutout.generators import FullImageCutoutGenerator
from tldist.cutout.processing import Resize as CutoutResize
from tldist.cutout.processing import Crop as CutoutCrop
from tldist.database import get_database

from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

# Create the database
print('Going to setup the database in {}'.format(config['database']['filename']))

if os.path.isdir(config['database']['filename']):
    shutil.rmtree(config['database']['filename'])
db = get_database(config['database']['type'], config['database']['filename'])


# Load the data
print('Loading the Hubble meta data and location information')
processing_dict = pickle.load(open('../../data/hubble_acs.pck', 'rb'))

print('Setting up the data structure required')
gray_scale = DataGrayScale()
data = []
for fileinfo in np.random.choice(processing_dict, 3000, replace=False):
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    im.add_processing(gray_scale.save())
    data.append(im)
    db.save('data', im)

#
#  Create cutouts
#
print('Creating the Full image cutout generator')
full_cutout = FullImageCutoutGenerator(output_size=(224, 224))

cutout_crop = CutoutCrop([15, -15, 15, -15])
cutout_resize = CutoutResize([224, 224])

print('Going to create the cutouts')
cutouts = []
for datum in data:
    cutout = full_cutout.create_cutouts(datum)

    # Add the processing
    cutout.add_processing(cutout_crop) 
    cutout.add_processing(cutout_resize)

    db.save('cutout', cutout)
    cutouts.append(cutout)

# Create the fingerprint calculator... fingerprint
print('Creating the info for the fingerprint calculator')
fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

print('Calculating the fingerprints')
fingerprints = fingerprint_calculate(cutouts, fc_save)
[db.save('fingerprint', x) for x in fingerprints]

print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')
db.save('similarity', similarity_tsne)

print('Calculating the Jaccard similarity')
similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
db.save('similarity', similarity_jaccard)

print('Calculating the Distance similarity')
similarity_distance = similarity_calculate(fingerprints, 'distance')
db.save('similarity', similarity_distance)
