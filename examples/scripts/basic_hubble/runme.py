import pickle
import numpy as np

from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.data import Data, DataCollection
from transfer_learning.cutout.generators import FullImageCutoutGenerator
from transfer_learning.misc import image_processing
from transfer_learning.similarity import calculate as similarity_calculate
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

#
# Load the data from the pickle file.
#   This file has all the location information and meta data we will
#   put into the data for display later on.
#

print('Loading the Hubble meta data and location information')
processing_dict = pickle.load(open('../../data/hubble_acs.pck', 'rb'))

#
# Setup an image processing step on the data. This will convert
# any data that is not gray scale into gray scal.
#

gray_scale = image_processing.GrayScale()

#
#  Now create teh actual data and add to the data collection.
#

print('Setting up the data structure required')
data_collection = DataCollection()
np.random.seed(12)
for fileinfo in np.random.choice(processing_dict, 200, replace=False):
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    im.add_processing(gray_scale)

    # Add to the data collection
    data_collection.add(im)

#
#  Create cutout pre-processing steps, which for this,
#  is just crop and resize.
#

cutout_crop = image_processing.Crop([15, -15, 15, -15])
cutout_resize = image_processing.Resize([224, 224])

#
# Cutout generator
#  This is a full window cutout generator. So, essentially each
#  image will be a cutout.
#

full_cutout = FullImageCutoutGenerator(output_size=(224, 224))

#
#  Now create the cutouts based on the generator and the data.
#

cutout_processing = [cutout_crop, cutout_resize]
cutouts = full_cutout.create_cutouts(data_collection,
                                     cutout_processing=cutout_processing)


#
# Create the fingerprint calculator... fingerprint
#

print('Creating the info for the fingerprint calculator')
fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

#
# Calcualte the fingerprint for each of the cutouts in
# the cutout collection.
#

print('Calculating the fingerprints')
fingerprints = fingerprint_calculate(cutouts, fc_save)

#
# Calcualte the similarity from the fingerprints based
# on three differnt methods.
#

print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')

print('Calculating the Jaccard similarity')
similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')

print('Calculating the Distance similarity')
similarity_distance = similarity_calculate(fingerprints, 'distance')

#
#  Save all the data.
#

with open('similarity_tsne.pck', 'wb') as fp:
    pickle.dump(similarity_tsne.save(), fp)

with open('similarity_jaccard.pck', 'wb') as fp:
    pickle.dump(similarity_jaccard.save(), fp)

with open('similarity_distance.pck', 'wb') as fp:
    pickle.dump(similarity_distance.save(), fp)
