import pickle
import os
import shutil
import numpy as np

from transfer_learning.fingerprint import FingerprintFilter
from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.similarity.similarity import calculate as similarity_calculate
from transfer_learning.data import Data
from transfer_learning.data.processing import GrayScale as DataGrayScale
from transfer_learning.cutout.generators import FullImageCutoutGenerator
from transfer_learning.cutout.processing import Resize as CutoutResize
from transfer_learning.cutout.processing import Crop as CutoutCrop
from transfer_learning.cutout.processing import HistogramEqualization as CutoutHistogramEqualization
from transfer_learning.cutout.processing import RescaleIntensity as CutoutRescaleIntensity
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
gray_scale = DataGrayScale()
data = []
np.random.seed(12)
for fileinfo in np.random.choice(processing_dict, 5000, replace=False):
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    im.add_processing(gray_scale)
    data.append(im)
    db.save('data', im)

#
#  Create cutouts
#
print('Creating the Full image cutout generator')
full_cutout = FullImageCutoutGenerator(output_size=(224, 224))

cutout_crop = CutoutCrop([15, -15, 15, -15])
cutout_resize = CutoutResize([224, 224])
cutout_histeq = CutoutHistogramEqualization()

print('Going to create the cutouts')
cutouts = []
for datum in data:

    #
    #  Create base cutout and add in there.
    #

    cutout = full_cutout.create_cutouts(datum)
    processed_cutout = cutout.duplicate_with_processing([cutout_crop, cutout_resize])
    db.save('cutout', processed_cutout)
    cutouts.append(processed_cutout)

    #
    # Add the processing and add in
    #

    processed_cutout = cutout.duplicate_with_processing([cutout_crop, cutout_resize, cutout_histeq])
    db.save('cutout', processed_cutout)
    cutouts.append(processed_cutout)


# Create the fingerprint calculator... fingerprint
print('Creating the info for the fingerprint calculator')
fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

print('Calculating the fingerprints')
fingerprints = fingerprint_calculate(cutouts, fc_save)
[db.save('fingerprint', x) for x in fingerprints]

# An example method of filtering fingerprints
#   This will actually select everything as they should all be listed public.
f_filter = FingerprintFilter(inclusion_patterns=['PUBLIC'])
output_fingerprints = f_filter.filter(fingerprints)
print('output_fingerprints {}'.format(output_fingerprints))

print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(output_fingerprints, 'tsne')
db.save('similarity', similarity_tsne)

#print('Calculating the Jaccard similarity')
#similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
#db.save('similarity', similarity_jaccard)
#
#print('Calculating the Distance similarity')
#similarity_distance = similarity_calculate(fingerprints, 'distance')
#db.save('similarity', similarity_distance)
