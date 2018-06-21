from configparser import ConfigParser
import pickle

from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.similarity.similarity import calculate as similarity_calculate
from transfer_learning.data import Data, DataCollection
from transfer_learning.cutout import CutoutCollection
from transfer_learning.fingerprint import FingerprintCollection
from transfer_learning.cutout.generators import BasicCutoutGenerator

fc_save = FingerprintCalculatorResnet().save()

config = ConfigParser()
config.read('config.ini')

#
# Load the data
#

print('Going to load the carina data')
image_data = Data(location='../../data/carina.tiff', radec=(10.7502222, -59.8677778),
                  meta={}, processing=[])
image_data.get_data()

#
# Add to the data collection
#

dc = DataCollection()
dc.add(image_data)

#
#  Create the sliding window cutout generator.
#

print('Creating cutout generator')
sliding_window_cutouts = BasicCutoutGenerator(output_size=224,
                                              step_size=100)

#
#  Create the cutouts using a sliding window cutout generator.
#

print('Creating cutouts')
cutouts = sliding_window_cutouts.create_cutouts(image_data)

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

#
# Save the data to a pickle file.
#

with open('similarity_tsne.pck', 'wb') as fp:
    pickle.dump(similarity_tsne.save(), fp)
