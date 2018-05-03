import uuid
import pickle

from tldist.fingerprint.processing import FingerprintCalculatorResnet
from tldist.fingerprint.processing import calculate as fingerprint_calculate
from tldist.similarity.processing import calculate as similarity_calculate
from tldist.data.data import Data


fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

# Load the data
filename_prepend = 'http://18.218.192.161:4123/ACSimages/'
processing_dict = pickle.load(open('../data/hubble_acs.pck', 'rb'))

data = []
for fileinfo in processing_dict[:20]:
    im = Data(location=fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    data.append(im)

fingerprints = fingerprint_calculate(data, fc_save)
print([str(x) for x in fingerprints])

similarity_tsne = similarity_calculate(fingerprints, 'tsne')

similarity_jaccard = similarity_calculate(fingerprints, 'tsne')
