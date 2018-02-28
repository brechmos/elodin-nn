from similarity import tSNE, Jaccard, Distance
from data_processing import MedianFilterData, ZoomData, RotateData, GrayScaleData
from fingerprint import FingerprintResnet, FingerprintInceptionV3
from transfer_learning import TransferLearning
from transfer_learning_display import TransferLearningDisplay

filename = 'acs.pck'

tl = TransferLearning.load(filename)

# Calculate hte similarity
#similarity = Jaccard()
#similarity = tSNE()
similarity = Distance(metric='euclidean')

# Create the display with the calculated similarity
tld = TransferLearningDisplay(similarity)

# Display the figure
tld.show(tl.fingerprints)

