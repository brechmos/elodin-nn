import itertools
import glob
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(levelname)s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("TransferLearning")
log.setLevel(logging.INFO)


class TransferLearning:

    def __init__(self):
        self._finger_print = None

    def set_fingerprint(self, fp):
        self._finger_print = fp

    def create_fingerprints(self, directory=None):

        # Get the list of rows and columns we need to run over
        nrows, ncols = self._finger_print._data.shape[:2]
        rows = range(112, nrows-224, 112)
        cols = range(112, ncols-224, 112)

        # Run over all combinations of rows and columns
        for row, col in itertools.product(rows, cols):
            self._finger_print.calculate(row, col)
            self._finger_print.save(directory)

if __name__ == "__main__":
    from fingerprint import ResnetFingerprint
    from data import MedianFilterData
    from similarity import tSNE

    filename = '/Users/crjones/christmas/hubble/Carina/data/carina.tiff'
    #data = np.random.rand(1000,1000)

    # tl = TransferLearning()
    # data = MedianFilterData(filename)
    # fp = ResnetFingerprint(data)
    #
    # tl.set_fingerprint(fp)
    #
    # tl.create_fingerprints('/tmp/resnet')

    # Calculate the tSNE similarity
    files = glob.glob('/tmp/resnet/*pck')
    tsne = tSNE()
    tsne.calculate(files)

    plt.figure(1)
    plt.clf()
    axis = plt.axes([0.1, 0.1, 0.5, 0.5])

    sub_windows = []
    for row in range(3):
        for col in range(3):
            tt = plt.axes([0.5 + 0.17 * col, 0.75 - 0.25 * row, 0.15, 0.15])
            tt.set_xticks([])
            tt.set_yticks([])
            sub_windows.append(tt)

    while True:

        tsne.displayY(axis)

        point = plt.ginput(1)

        close_fingerprints = tsne.find_similar(point)
        for ii, (cf_similarity, cf_fingerprint) in enumerate(close_fingerprints):
            cf_fingerprint.display(sub_windows[ii], title=cf_similarity)
