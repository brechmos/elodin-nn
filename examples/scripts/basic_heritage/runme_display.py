import pickle

from elodin_nn.similarity import Similarity
from elodin_nn.similarity.qtdisplay import SimilarityDisplay

#
# Load up the pickle file.
#

# Or change this to similarity_jaccard.pck
with open('similarity_tsne.pck', 'rb') as fp:
    stsne = pickle.load(fp)
    similarity_tsne = Similarity.factory(stsne)

#
# Run the display program
#

sd = SimilarityDisplay(similarity_tsne)
