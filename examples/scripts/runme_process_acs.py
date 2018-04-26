from transfer_learning.similarity import tSNE
from transfer_learning.data_processing import MedianFilterData, ZoomData, RotateData, GrayScaleData, CropData, GaussianSmoothData
from transfer_learning.cutouts import BasicCutouts, BlobCutouts, FullImageCutout
from transfer_learning.fingerprint import FingerprintResnet, FingerprintInceptionV3
from transfer_learning.transfer_learning import TransferLearning, TransferLearningDisplay
import glob
import pickle

# Load the dictionary of filenames and meta information
# Each element of the list must be a dictionary that contains
# keys of: filename, radec and meta
processing_dict = pickle.load(open('../data/hubble_acs.pck', 'rb'))

fingerprint_model = FingerprintResnet()
basic_cutout = FullImageCutout(output_size=224)

# Added Gray Scale as some were 3 channel gray scale
data_processing = [
            [CropData(), GrayScaleData()],
        ]

tl = TransferLearning(basic_cutout, data_processing, fingerprint_model)
tl.calculate_stream(processing_dict[:200])

tl.save('acs_200.pck')
