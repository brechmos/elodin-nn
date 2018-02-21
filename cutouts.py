import uuid
import itertools

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("Cutouts")
log.setLevel(logging.INFO)


class Cutouts:
    def __init__(self):
        # list of data processing elements
        self._uuid = str(uuid.uuid4())

        # number of pixels around the image to include in the display
        self._image_display_margin = 50

    def create_cutouts(self, data):
        raise NotImplemented('Must create cutouts function')


class BasicCutouts:
    def __init__(self, output_size, step_size):

        # list of data processing elements
        self._output_size = output_size
        self._step_size = step_size
        self._uuid = str(uuid.uuid4())

    def calculate(self, data):
        """
        Calculate the fingerprints for each subsection of the image in each file.

        :param stepsize:
        :param display:
        :return:
        """

        N = self._step_size // 2

        # Determine the centers to use for the fingerprint calculation
        nrows, ncols = data.shape[:2]
        rows = range(N, nrows-N, self._stepsize)
        cols = range(N, ncols-N, self._stepsize)

        for ii, (row, col) in enumerate(itertools.product(rows, cols)):
            td = data[row-N:row+N, col-N:col+N]
            yield td

