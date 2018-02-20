from similarity import tSNE
from data_processing import MedianFilterData, ZoomData, RotateData, GrayScaleData
from fingerprint import FingerprintResnet, FingerprintInceptionV3
from transfer_learning import TransferLearning, TransferLearningDisplay
import glob

input_file_pattern = '/Users/crjones/christmas/hubble/MAGPIS/G371D.tiff'

stepsize = 400

input_filenames = glob.glob(input_file_pattern)

fingerprint_model = FingerprintInceptionV3()

# # calculate fingerpirnts for median filtered

#data_processing = [MedianFilterData((3, 3, 1)), GrayScaleData()]
data_processing = [GrayScaleData(), ZoomData(3)]

tl = TransferLearning(fingerprint_model, data_processing)
tl.set_files(input_filenames)

fingerprints = tl.calculate(stepsize=stepsize, display=True)

tl.save('test_fingerprints.pck')
