import uuid
import pickle

from tldist.fingerprint.processing import FingerprintResnet
from tldist.fingerprint.task import calculate_celery
from tldist.similarity.task import similarity_celery


def stringify(dictionary):
    return {k: str(v) for k, v in dictionary.items()}


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


fresnet = FingerprintResnet()
fc_save = fresnet.save()

# Load the data
processing_dict = pickle.load(open('../data/hubble_acs.pck', 'rb'))

data = []
for fileinfo in processing_dict[:36]:
    im = {
        'uuid': str(uuid.uuid4()),
        'location': fileinfo['location'],
        'radec': fileinfo['radec'],
        'meta': stringify(fileinfo['meta'])
    }
    # data_client.save(im)
    data.append(im)

fingerprints = calculate_celery(data, fc_save)

# fingerprints = fingerprint_client.get()
# print('fingerprint pks {}'.format([str(x[db.key]) for x in fingerprints]))

similarity_tsne = similarity_celery(fingerprints, 'tsne')
similarity_jaccard = similarity_celery(fingerprints, 'jaccard')
