from similarity import tSNE
from data_processing import MedianFilterData, ZoomData, RotateData, GrayScaleData
from fingerprint import FingerprintResnet, FingerprintInceptionV3
from transfer_learning import TransferLearning
from transfer_learning_display import TransferLearningDisplay

filename = 'test.pck'

tl = TransferLearning.load(filename)

# Calculate hte similarity
#similarity = Jaccard(fingerprints)
similarity = tSNE(tl.fingerprints)

# Create the display with the calculated similarity
tld = TransferLearningDisplay(similarity)

# Display the figure
tld.show(tl.fingerprints)

