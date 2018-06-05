import os
import json
import numpy as np

from transfer_learning.data import Data, DataCollection
from transfer_learning.cutout import CutoutCollection
from transfer_learning.fingerprint import FingerprintCollection
from transfer_learning.fingerprint.processing import calculate as fingerprint_calculate
from transfer_learning.fingerprint.processing import FingerprintCalculatorResnet
from transfer_learning.similarity import Similarity
from transfer_learning.similarity import calculate as similarity_calculate
from transfer_learning.cutout.generators import BasicCutoutGenerator


def test_carina():

    # Load the data.
    carina_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/carina.tiff')
    image_data = Data(location=carina_location, radec=(10.7502222, -59.8677778),
                      meta={}, processing=[])
    image_data.get_data()

    # Add to the data collection
    dc = DataCollection()
    dc.add(image_data)

    assert len(dc) == 1

    #
    #  Create the cutouts with a processing step applied
    #
    sliding_window_cutouts = BasicCutoutGenerator(output_size=224,
                                                  step_size=550)

    cc = CutoutCollection()
    for cutout in sliding_window_cutouts.create_cutouts(image_data):
        cc.add(cutout)

    assert len(cc) == 35

    cmp_arr = np.array([[[51, 66, 69], [50, 70, 78]], [[48, 66, 72], [49, 65, 72]]], dtype=np.uint8)
    assert np.allclose(cc[0].get_data()[:2, :2], cmp_arr)

    #
    #  Compute the fingerprints for each cutout
    #
    fc = FingerprintCollection()
    fc_save = FingerprintCalculatorResnet().save()
    for fingerprint in fingerprint_calculate(cc, fc_save):
        fc.add(fingerprint)

    assert [x[1] for x in fc[0].predictions[:3]] == ['hammerhead', 'stingray', 'binder']

    #
    #  Compute the similarity metrics
    #
    similarity_tsne = similarity_calculate(fc, 'tsne')

    assert True


def test_end2end():

    # Load the data.
    carina_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/carina.tiff')
    image_data = Data(location=carina_location, radec=(10.7502222, -59.8677778),
                      meta={}, processing=[])
    image_data.get_data()

    # Add to the data collection
    dc = DataCollection()
    dc.add(image_data)

    #
    #  Create the cutouts with a processing step applied
    #
    sliding_window_cutouts = BasicCutoutGenerator(output_size=224,
                                                  step_size=550)

    cc = CutoutCollection()
    for cutout in sliding_window_cutouts.create_cutouts(image_data):
        cc.add(cutout)

    #
    #  Compute the fingerprints for each cutout
    #
    fc = FingerprintCollection()
    fc_save = FingerprintCalculatorResnet().save()
    for fingerprint in fingerprint_calculate(cc, fc_save):
        fc.add(fingerprint)

    #
    #  Compute the similarity metrics
    #
    similarity_tsne = similarity_calculate(fc, 'tsne')
    new_similarity_tsne = Similarity.factory(similarity_tsne.save())

    assert json.dumps(new_similarity_tsne.save(), sort_keys=True) == json.dumps(similarity_tsne.save(), sort_keys=True)
