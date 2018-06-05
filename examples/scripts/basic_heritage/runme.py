import os
import glob
import shutil
import pickle
from configparser import ConfigParser

from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.similarity.similarity import calculate as similarity_calculate
from transfer_learning.data import Data, DataCollection
from transfer_learning.cutout import CutoutCollection
from transfer_learning.cutout.generators import BasicCutoutGenerator
from transfer_learning.database import get_database

fc_save = FingerprintCalculatorResnet().save()

config = ConfigParser()
config.read('config.ini')

#
# Load the data
#

print('Going to calculate the sliding window cutouts')
sliding_window_cutouts = BasicCutoutGenerator(output_size=224, step_size=400)

print('Going to load the HST Heritage data')
data = DataCollection()
cutouts = CutoutCollection()
for filename in glob.glob('../../data/heritage/*.???'):
    print('   processing {}'.format(filename))
    image_data = Data(location=filename, radec=(-32, 12), meta={})
    image_data.get_data()
    data.add(image_data)

    #
    #  Create the cutouts
    #
    cutouts = cutouts + sliding_window_cutouts.create_cutouts(data)
    print('created {} cutouts'.format(len(cutouts)))

#
#  Compute the fingerprints for each cutout
#
print('Calculate the fingerprint for each cutout')
fingerprints = fingerprint_calculate(cutouts, fc_save)

#
#  Compute the similarity metrics
#
print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')

with open('similarity_tsne.pck', 'wb') as fp:
    pickle.dump(similarity_tsne.save(), fp)

# print('Calculating the jaccard similarity')
# similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
