import uuid
import pickle

from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.similarity.similarity import calculate as similarity_calculate
from transfer_learning.data import Data
from transfer_learning.cutout.generators import BasicCutoutGenerator

fc_save = FingerprintCalculatorResnet().save()

#
# Load the data
#

print('Going to load the carina data')
image_data = Data(location='../../data/carina.tiff', radec=(-32, 12), meta={})
image_data.get_data()

#
#  Create the cutouts
#
print('Going to calculate the sliding window cutouts')
sliding_window_cutouts = BasicCutoutGenerator(output_size=224, step_size=424)
cutouts = sliding_window_cutouts.create_cutouts(image_data)

#
#  Compute the fingerprints for each cutout
#
print('Calculate the fingerprint for each cutout')
fingerprints = fingerprint_calculate(cutouts, fc_save)
print([str(x) for x in fingerprints])

#
#  Compute the similarity metrics
#
print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')

print('Calculating the jaccard similarity')
similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
