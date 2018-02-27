from similarity import tSNE
from data_processing import MedianFilterData, ZoomData, RotateData, GrayScaleData
from cutouts import BasicCutouts, BlobCutouts
from fingerprint import FingerprintResnet, FingerprintInceptionV3
from transfer_learning import TransferLearning, TransferLearningDisplay
import glob

input_file_pattern = '/Users/crjones/christmas/hubble/HSTHeritage/data/[a-d]*.???'

step_size = 200

input_filenames = glob.glob(input_file_pattern)

fingerprint_model = FingerprintResnet()

#basic_cutout = BasicCutouts(output_size=224, step_size=step_size)
basic_cutout = BlobCutouts(output_size=224)

# # calculate fingerpirnts for median filtered

data_processing = [
            [GrayScaleData()],
        ]

tl = TransferLearning(basic_cutout, data_processing, fingerprint_model)
tl.set_files(input_filenames)

tl.calculate()

tl.save('test.pck')
