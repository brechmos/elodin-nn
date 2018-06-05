import os
import shutil
from configparser import ConfigParser

from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.similarity.similarity import calculate as similarity_calculate
from transfer_learning.data import Data, DataCollection
from transfer_learning.cutout import CutoutCollection
from transfer_learning.fingerprint import FingerprintCollection
from transfer_learning.cutout.generators import BasicCutoutGenerator, BlobCutoutGenerator
from transfer_learning.database import get_database

fc_save = FingerprintCalculatorResnet().save()

config = ConfigParser()
config.read('config.ini')

#
# Load the data
#

# Get the data.
print('Going to load the carina data')
image_data = Data(location='../../data/carina.tiff', radec=(10.7502222, -59.8677778),
                  meta={}, processing=[])
image_data.get_data()

# Add to the data collection
dc = DataCollection()
dc.add(image_data)

#
#  Create the cutouts with a processing step applied
#
print('Going to calculate the sliding window cutouts')
sliding_window_cutouts = BasicCutoutGenerator(output_size=224,
                                              step_size=550)
cc = CutoutCollection()

for cutout in sliding_window_cutouts.create_cutouts(image_data):
    cc.add(cutout)

#
#  Compute the fingerprints for each cutout
#
print('Calculate the fingerprint for each cutout')
fc = FingerprintCollection()
for fingerprint in fingerprint_calculate(cc, fc_save):
    fc.add(fingerprint)

#
#  Compute the similarity metrics
#
print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fc, 'tsne')

# print('Calculating the jaccard similarity')
# similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
