from similarity import tSNE
from data_processing import MedianFilterData, ZoomData, RotateData, GrayScaleData
from fingerprint import FingerprintResnet, FingerprintInceptionV3
from transfer_learning import TransferLearning, TransferLearningDisplay
import glob

filename =

tl = TransferLearning()

tl.load(filename)

fingerprints = tl['fingerprints']

#similarity = Jaccard(fingerprints)
similarity = tSNE(fingerprints)

tld = TransferLearningDisplay(similarity)
tld.show(fingerprints)

