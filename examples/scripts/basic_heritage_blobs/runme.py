import glob
import pickle

from elodin_nn.fingerprint.processing import FingerprintCalculatorResnet
from elodin_nn.fingerprint.processing import calculate as fingerprint_calculate
from elodin_nn.similarity.similarity import calculate as similarity_calculate
from elodin_nn.misc.image_processing import Resize
from elodin_nn.data import Data, DataCollection
from elodin_nn.cutout.generators import BlobCutoutGenerator

#
# Load the data
#

print('Going to calculate the sliding window cutouts')
blob_cutout = BlobCutoutGenerator(output_size=(224, 224), mean_threshold=2.0,
                                  gaussian_smoothing_sigma=2, label_padding=10)

print('Going to load the HST Heritage data')
data = DataCollection()
for filename in glob.glob('../../data/heritage/*.???'):

    #
    # Load the data
    #

    print('   processing {}'.format(filename))
    image_data = Data(location=filename, radec=(-32, 12), meta={})
    data.add(image_data)

#
#  Create the cutouts
#

resize_224 = Resize(output_size=(224, 224))
cutouts = blob_cutout.create_cutouts(data, cutout_processing=[resize_224])

#
#  Compute the fingerprints for each cutout
#

print('Calculate the fingerprint for each cutout')
fc_save = FingerprintCalculatorResnet().save()
fingerprints = fingerprint_calculate(cutouts, fc_save)

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
