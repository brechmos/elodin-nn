import numpy as np
import mahotas

from transfer_learning.fingerprint.fingerprint import Fingerprint, FingerprintCollection

from ..tl_logging import get_logger
log = get_logger('fingerprint processing')


def add_haralick(fingerprint):
    """
    Harlick features from the co-occurence matrix.

    Based on https://github.com/luispedro/mahotas/blob/master/mahotas/features/texture.py
    """
    log.info('')

    # If a FingerprintCollection is passed in
    if isinstance(fingerprint, FingerprintCollection):
        for f in fingerprint:
            _add_haralick(f)

    # If a Fingerprint is passed in
    elif isinstance(fingerprint, Fingerprint):
        _add_haralick(fingerprint)


def _add_haralick(fingerprint):
    """
    Compute Haralick's features on the co-occurrency matrix as defined
    in Mahotas.

    Parameters
    -----------
    fingerprint : Fingerprint object
        The fingerprint object over which we want to calculate the features.

    Notes
    -----
    The calculated features are added directly into the fingerprint
    in the "other_predictors" attribute.
    """

    # Get list of the 13 labels.
    labels = mahotas.features.texture.haralick_labels

    # Get the data, and fix if needed.
    data = fingerprint.cutout.get_data()
    if len(data.shape) == 3:
        data = data[:, :, 0]

    # Compute the features on the data, must be integer input.
    out = mahotas.features.haralick(data.astype(np.int16), return_mean=True)

    # Reformat the output to include the labels and add as another predictor.
    doc_features = [tuple(x) for x in zip(labels[4:], labels[4:], out[4:])]
    fingerprint.add_other_predictor('haralick_co-occurency_matrix', doc_features)


def add_zernike_moment(fingerprint):
    """
    Calculate the fingerprint from a list of data.  The data
    must be of the form
         [ {'uuid': <somtehing>, 'location': <somewhere>, 'meta': {<meta data} }... ]
    """
    log.info('')

    if isinstance(fingerprint, FingerprintCollection):
        for f in fingerprint:
            _add_zernike_moment(f)
    elif isinstance(fingerprint, Fingerprint):
        _add_zernike_moment(fingerprint)


def _add_zernike_moment(fingerprint):
    zm1 = mahotas.features.zernike_moments(fingerprint.cutout.get_data(), 1)

    fingerprint.add_other_predictor('zernike_moment', zm1)
