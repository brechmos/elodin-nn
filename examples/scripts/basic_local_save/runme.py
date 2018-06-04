import pickle
import os
import shutil
import numpy as np

from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.data import Data, DataCollection
from transfer_learning.cutout.generators import FullImageCutoutGenerator
from transfer_learning.misc import image_processing
from transfer_learning.similarity import calculate as similarity_calculate
from transfer_learning.database import get_database

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
gray_scale = image_processing.GrayScale()
data_collection = DataCollection()
np.random.seed(12)
for fileinfo in np.random.choice(processing_dict, 20, replace=False):
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    im.add_processing(gray_scale)

    # Add to the data collection
    data_collection.add(im)

#
#  Create cutouts
#
print('Creating the Full image cutout generator')

# Cutout processing
cutout_crop = image_processing.Crop([15, -15, 15, -15])
cutout_resize = image_processing.Resize([224, 224])
cutout_histeq = image_processing.HistogramEqualization()

# Cutout generator
full_cutout = FullImageCutoutGenerator(output_size=(224, 224))

print('Going to create the cutouts')
cutout_processing = [cutout_crop, cutout_resize]
cutouts_orig = full_cutout.create_cutouts(data_collection,
                                          cutout_processing=cutout_processing)

cutout_processing = [cutout_crop, cutout_resize, cutout_histeq]
cutouts_histeq = full_cutout.create_cutouts(data_collection,
                                            cutout_processing=cutout_processing)

cutouts = cutouts_orig + cutouts_histeq
print('{} = {} + {}'.format(len(cutouts), len(cutouts_orig), len(cutouts_histeq)))

# Create the fingerprint calculator... fingerprint
print('Creating the info for the fingerprint calculator')
fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

print('Calculating the fingerprints')
fingerprints = fingerprint_calculate(cutouts, fc_save)

#
# # An example method of filtering fingerprints
# #   This will actually select everything as they should all be listed public.
# f_filter = FingerprintFilter(inclusion_patterns=['PUBLIC'])
# output_fingerprints = f_filter.filter(fingerprints)
# print('output_fingerprints {}'.format(output_fingerprints))

print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')

print('Calculating the Jaccard similarity')
similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
db.save('similarity', similarity_jaccard)

print('Calculating the Distance similarity')
similarity_distance = similarity_calculate(fingerprints, 'distance')
db.save('similarity', similarity_distance)
