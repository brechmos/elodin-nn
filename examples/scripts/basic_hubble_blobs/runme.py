import pickle
import numpy as np

from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.data import Data, DataCollection
from transfer_learning.cutout.generators import BlobCutoutGenerator
from transfer_learning.misc import image_processing
from transfer_learning.similarity import calculate as similarity_calculate
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

#
# Load the pickle file with the location and meta information.
#

print('Loading the Hubble meta data and location information')
processing_dict = pickle.load(open('../../data/hubble_acs.pck', 'rb'))

#
# Create data pre-processing steps.
#

gray_scale = image_processing.GrayScale()
data_crop = image_processing.Crop([15, -15, 15, -15])

#
# Load up the data
#

print('Setting up the data structure required')
data_collection = DataCollection()
np.random.seed(12)
for fileinfo in np.random.choice(processing_dict, 800, replace=False):
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    im.add_processing(gray_scale)
    im.add_processing(data_crop)

    # Add to the data collection
    data_collection.add(im)

#
# Cutout generator
#

blob_cutout = BlobCutoutGenerator(output_size=(224, 224), mean_threshold=2.0,
                                  gaussian_smoothing_sigma=2, label_padding=10)

#
#  Create cutouts
#

print('Going to create the cutouts')

cutout_resize = image_processing.Resize([224, 224])
cutout_processing = [cutout_resize]
cutouts = blob_cutout.create_cutouts(data_collection,
                                     cutout_processing=cutout_processing)

#
# Create the fingerprint calculator... fingerprint
#

print('Creating the info for the fingerprint calculator')
fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

#
#  Calculate the fingerprints from the cutouts.
#

print('Calculating the fingerprints')
fingerprints = fingerprint_calculate(cutouts, fc_save)

#
# Calculate the fingerprint similarity
#

print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')

#
#  Save all the data.
#

with open('similarity_tsne.pck', 'wb') as fp:
    pickle.dump(similarity_tsne.save(), fp)
