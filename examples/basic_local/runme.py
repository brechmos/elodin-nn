import uuid
import shutil
from collections import OrderedDict
import pickle
import time

from tldist.fingerprint.processing import calculate as fingerprint_calculate
from tldist.similarity.processing import calculate as similarity_calculate
import os.path

def stringify(dictionary):
    return {k: str(v) for k, v in dictionary.items()}

from transfer_learning.fingerprint import FingerprintResnet
fresnet = FingerprintResnet()
fc_save = fresnet.save()

# Load the data
filename_prepend = 'http://18.218.192.161:4123/ACSimages/'
processing_dict = pickle.load(open('hubble_acs.pck', 'rb'))

data = []
for fileinfo in processing_dict[:100]:
    im = {
        'uuid': str(uuid.uuid4()),
        'location': fileinfo['location'],
        'radec': fileinfo['radec'],
        'meta': stringify(fileinfo['meta'])
        }
    #data_client.save(im)
    data.append(im)

fingerprints = fingerprint_calculate(data, fc_save)
print([x['uuid'] for x in fingerprints])

similarity_tsne = similarity_calculate(fingerprints, 'tsne')
 
similarity_jaccard = similarity_calculate(fingerprints, 'tsne')
