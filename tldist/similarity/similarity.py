import uuid

class Similarity:

    def __init__(self, similarity_type=None, similarity=None, fingerprint_uuids=[]):
        self._uuid = str(uuid.uuid4())
        self._similarity_type = similarity_type
        self._similarity = similarity
        self._fingerprint_uuids = fingerprint_uuids

    def __str__(self):
        return 'Similarity {} based on {}...'.format(
                self._similarity_type, self._fingerprint_uuids[:3])

    @property
    def similarity_type(self):
        return self._similarity_type

    @similarity_type.setter
    def similarity_type(self, value):
        self._similarity_type = value

    @property
    def similarity(self):
        return self._similarity

    @similarity.setter
    def similarity(self, value):
        self._similarity = value

    @property
    def fingerprint_uuids(self):
        return self._fingerprint_uuids

    @fingerprint_uuids.setter
    def fingerprint_uuids(self, value):

        if not instance(value, list):
            raise ValueError('Fingerprint_uuids must be a list')
        
        self._fingerprint_uuids = value

    def save(self):
        return {
            'similarity_type': self._similarity_type,
            'similarity': self._similarity,
            'fingerprint_uuids': self._fingerprint_uuids
        }

    def load(self, thedict):
        self._similarity_type = thedict['self._similarity_type']
        self._similarity = thedict['similarity']
        self._fingerprint_uuids = thedict['fingerprint_uuids']
