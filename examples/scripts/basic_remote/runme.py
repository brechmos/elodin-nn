import uuid
import pickle

from tldist.fingerprint.processing import FingerprintCalculatorResnet
from tldist.fingerprint.task import calculate_celery as calculate_fingerprints
from tldist.similarity.task import similarity_celery as calculate_similarity
from tldist.data import Data
from tldist.cutout.generators import FullImageCutoutGenerator


def stringify(dictionary):
    return {k: str(v) for k, v in dictionary.items()}


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

#
# Load the data
#
print('Loading the processing dict')
processing_dict = pickle.load(open('../../data/hubble_acs.pck', 'rb'))

#
#  Create the datasets
#
print('Creating data objects')
data = []
for fileinfo in processing_dict[:36]:
    im = Data(location= fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    data.append(im)

#
#  Create cutouts
#
print('Creating the cutouts')
full_cutout = FullImageCutoutGenerator(output_size=(224, 224))

cutouts = []
for datum in data:
    cutout = full_cutout.create_cutouts(datum)
    cutouts.append(cutout)

#
#  Calculate the fingerprints
#
print('Calculating the fingerprints')
fingerprints = calculate_fingerprints(cutouts, fc_save)

#
#  Compute the similarity
#
print('Calculating the similarity')
similarity_tsne = calculate_similarity(fingerprints, 'tsne')
similarity_jaccard = calculate_similarity(fingerprints, 'jaccard')
