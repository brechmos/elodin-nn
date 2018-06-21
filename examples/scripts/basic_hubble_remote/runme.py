import pickle

from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.fingerprint.task import calculate_celery as calculate_fingerprints
from transfer_learning.similarity.task import similarity_celery as similarity_calculate
from transfer_learning.data import Data, DataCollection
from transfer_learning.misc.image_processing import Resize
from transfer_learning.cutout.generators import FullImageCutoutGenerator

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')


def stringify(dictionary):
    return {k: str(v) for k, v in dictionary.items()}


fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

#
# Load the location and meta information of the data.
#

print('Loading the processing dict')
processing_dict = pickle.load(open('../../data/hubble_acs.pck', 'rb'))

#
# Create the data pre-processing.
#

resize_224 = Resize(output_size=(224, 224))

#
#  Create the datasets
#

print('Creating data objects')
data = DataCollection()
for fileinfo in processing_dict[:20]:
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    im.add_processing(resize_224)
    data.append(im)

#
#  Create cutout generator
#

print('Creating the cutout generator')
full_cutout = FullImageCutoutGenerator(output_size=(224, 224))

#
#  Create the cutouts.
#

cutouts = full_cutout.create_cutouts(data)

#
#  Calculate the fingerprints
#
print('Calculating the fingerprints for {} cutouts'.format(len(cutouts)))
fingerprints = calculate_fingerprints(cutouts, fc_save)

#
#  Compute the similarity metrics
#

print('Calculating the tSNE similarity')
similarity_tsne = similarity_calculate(fingerprints, 'tsne')
with open('similarity_tsne.pck', 'wb') as fp:
    pickle.dump(similarity_tsne.save(), fp)

print('Calculating the Jaccard similarity')
similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
with open('similarity_jaccard.pck', 'wb') as fp:
    pickle.dump(similarity_jaccard.save(), fp)
