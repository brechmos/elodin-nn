import pickle
from elodin_nn.similarity import Similarity
from elodin_nn.similarity.qtdisplay import SimilarityDisplay

with open('similarity_tsne.pck', 'rb') as fp:
    similarity_tsne_dict = pickle.load(fp)
    similarity_tsne = Similarity.factory(similarity_tsne_dict)

sd = SimilarityDisplay(similarity_tsne)
