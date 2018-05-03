import uuid
import pickle

from tldist.fingerprint.processing import FingerprintCalculatorResnet
from tldist.fingerprint.task import calculate_celery as calculate_fingerprints
from tldist.similarity.task import similarity_celery as calculate_similarity
from tldist.data.data import Data


def stringify(dictionary):
    return {k: str(v) for k, v in dictionary.items()}


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


fresnet = FingerprintCalculatorResnet()
fc_save = fresnet.save()

# Load the data
processing_dict = pickle.load(open('../data/hubble_acs.pck', 'rb'))

data = []
for fileinfo in processing_dict[:36]:
    im = Data(location= fileinfo['location'], radec=fileinfo['radec'], meta=fileinfo['meta'])
    # data_client.save(im)
    data.append(im)

fingerprints = calculate_fingerprints(data, fc_save)

similarity_tsne = calculate_similarity(fingerprints, 'tsne')
similarity_jaccard = calculate_similarity(fingerprints, 'jaccard')
