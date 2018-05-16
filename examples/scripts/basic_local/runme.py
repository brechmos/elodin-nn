import uuid
import pickle

from tldist.fingerprint.processing import FingerprintCalculatorResnet
from tldist.fingerprint.processing import calculate as fingerprint_calculate
from tldist.similarity.similarity import calculate as similarity_calculate
from tldist.cutout.generators import FullImageCutoutGenerator
from tldist.data import Data, Crop, Resize


fc_save = FingerprintCalculatorResnet().save()

#
# Load the data
#
print('Loading the data')
processing_dict = pickle.load(open('../../data/hubble_acs.pck', 'rb'))
preprocessing = [Crop([10, -10, 10, -10]).save(), Resize((224,224)).save()]

data = []
for fileinfo in processing_dict[:10]:
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    [im.add_processing(x) for x in preprocessing]
    data.append(im)

#
#  Create cutouts
#
print('Creating the Full image cutout generator')
full_cutout = FullImageCutoutGenerator(output_size=(224, 224))

print('Going to create the cutouts')
cutouts = []
for datum in data:
    cutout = full_cutout.create_cutouts(datum)
    cutouts.append(cutout)

#
#  Compute the fingerprints for each cutout
#
fingerprints = fingerprint_calculate(cutouts, fc_save)
print([str(x) for x in fingerprints])

#
#  Comptue the similarity metrics
#
similarity_tsne = similarity_calculate(fingerprints, 'tsne')
similarity_jaccard = similarity_calculate(fingerprints, 'jaccard')
