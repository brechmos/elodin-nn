import glob
import pickle

from elodin_nn.cutout.generators import BasicCutoutGenerator
from elodin_nn.data import Data, DataCollection
from elodin_nn.fingerprint.processing import FingerprintCalculatorResnet
from elodin_nn.fingerprint.processing import calculate as fingerprint_calculate
from elodin_nn.similarity.similarity import calculate as similarity_calculate

#
# Load the data
#

print('Going to load the HST Heritage data')
data = DataCollection()
for filename in glob.glob('../../data/heritage/*.???'):

    print('   adding data {}'.format(filename))
    image_data = Data(location=filename, radec=(-32, 12), meta={})
    data.add(image_data)

#
# Create the cutout generator.
#

print('Going to calculate the sliding window cutouts')
sliding_window_cutouts = BasicCutoutGenerator(output_size=224, step_size=112)

#
#  Create the cutouts
#
cutouts = sliding_window_cutouts.create_cutouts(data)

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
