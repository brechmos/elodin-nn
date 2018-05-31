import os
import shutil
from configparser import ConfigParser

from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.similarity.similarity import calculate as similarity_calculate
from transfer_learning.data import Data
from transfer_learning.cutout.generators import BasicCutoutGenerator, BlobCutoutGenerator
from transfer_learning.cutout.processing import HistogramEqualization as CutoutHistogramEqualization
from transfer_learning.database import get_database

fc_save = FingerprintCalculatorResnet().save()

config = ConfigParser()
config.read('config.ini')

# Create the database
print('Going to setup the database in {}'.format(config['database']['filename']))

if os.path.isdir(config['database']['filename']):
    shutil.rmtree(config['database']['filename'])
db = get_database(config['database']['type'], config['database']['filename'])

#
# Load the data
#

print('Going to load the carina data')
image_data = Data(location='../../data/carina.tiff', radec=(-0.8542, 287.6099), meta={})
image_data.get_data()
db.save('data', image_data)

#
#  Create the cutouts
#
print('Going to calculate the sliding window cutouts')
sliding_window_cutouts = BasicCutoutGenerator(output_size=224, step_size=150)
cutouts = sliding_window_cutouts.create_cutouts(image_data)

histogram_equalization = CutoutHistogramEqualization()

for cutout in cutouts:
    processed_cutout = cutout.duplicate_with_processing([histogram_equalization])
    db.save('cutout', processed_cutout)

#
#  Compute the fingerprints for each cutout
#
print('Calculate the fingerprint for each cutout')
fingerprints = fingerprint_calculate(cutouts, fc_save)
print([str(x) for x in fingerprints])
[db.save('fingerprint', fingerprint) for fingerprint in fingerprints]

#
#  Compute the similarity metrics
#
print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')
db.save('similarity', similarity_tsne)

# print('Calculating the jaccard similarity')
# similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
