from similarity import tSNE
from matplotlib import pyplot as plt
import os
import utils
import glob

from data import Data

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("TransferLearning")
log.setLevel(logging.INFO)

class TransferLearningCalculate:

    def __init__(self, input_files):

        # fingerprint_model = FingerprintInceptionV3()
        fingerprint_model = FingerprintResnet()

        # # calculate fingerpirnts for median filtered
        log.info('Setting up median filter data')
        data_processing = [MedianFilterData((3, 3, 1)), GrayScaleData()]
        data = Data(fingerprint_model, data_processing)
        data.set_files(input_files)
        data.calculate(stepsize=stepsize, display=True)
        data.save(output_directory)


class TransferLearningDisplay:

    def __init__(self):
        self.tsne_similarity = None
        self.fig = None
        self.axis = None
        self.info_axis = None
        self.info_text = None
        self.sub_windows = None

    def display(self, fingerprints):
        plt.show(block=False)

        self.tsne_similarity = tSNE(fingerprints)
        self.tsne_similarity.calculate(fingerprints)

        self.fig = plt.figure(1)
        plt.gcf()
        self.axis = plt.axes([0.05, 0.05, 0.45, 0.45])

        self.axis_closest = plt.axes([0.5, 0.01, 0.2, 0.2])
        self.axis_closest.set_xticks([])
        self.axis_closest.set_yticks([])
        self.axis_closest.set_xlabel('')
        self.axis_closest.set_ylabel('')

        self.tsne_similarity.displayY(self.axis)

        self.info_axis = plt.axes([0.60, 0.02, 0.3, 0.05])
        self.info_axis.set_axis_off()
        self.info_axis.set_xticks([])
        self.info_axis.set_yticks([])
        self.info_axis.set_xlabel('')
        self.info_axis.set_ylabel('')
        self.info_text = self.info_axis.text(0, 0, '', fontsize=12)

        self._cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._onmove)

        self.sub_windows = []
        for row in range(3):
            for col in range(3):
                # rect = [left, bottom, width, height]
                tt = plt.axes([0.5 + 0.14 * col, 0.75 - 0.25 * row, 0.2, 0.2])
                tt.set_xticks([])
                tt.set_yticks([])
                self.sub_windows.append(tt)
        plt.show(block=False)

    def _update_text(self, thetext):
        self.info_text.set_text(thetext)
        plt.draw()

    def _onmove(self, event):
        if event.inaxes == self.axis:
            point = event.ydata, event.xdata
            close_fingerprint = self.tsne_similarity.find_similar(point, n=1)[0][1]

            self.axis_closest.imshow(utils.rgb2plot(
                close_fingerprint['data'].display(close_fingerprint['filename'],
                                                  close_fingerprint['row_center'],
                                                  close_fingerprint['column_center'])
            ))
            plt.show(block=False)
            plt.pause(0.0001)

    def _onclick(self, event):
        point = event.ydata, event.xdata
        self.axis.cla()
        self.tsne_similarity.displayY(self.axis)

        self._update_text('Loading data...')
        close_fingerprints = self.tsne_similarity.find_similar(point)

        self._update_text('Displaying result...')
        for ii, (distance, fingerprint) in enumerate(close_fingerprints):
            self.sub_windows[ii].imshow(utils.rgb2plot(
                fingerprint['data'].display(fingerprint['filename'],
                                            fingerprint['row_center'],
                                            fingerprint['column_center'])
            ))
            self.sub_windows[ii].set_title('{:0.3f} {} ({}, {})'.format(
                distance,
                os.path.basename(fingerprint['filename']),
                fingerprint['row_center'],
                fingerprint['column_center']), fontsize=8)

        self._update_text('Click in the tSNE plot...')

        plt.show(block=False)
        plt.pause(0.001)


if __name__ == "__main__":
    #input_fingerprints = '/tmp/resnet/data_be633177-2923-423f-bc7e-846d7647cf4d.pck'
    input_fingerprints = '/tmp/hst_heritage_gray/*.pck'

    filenames  = glob.glob(input_fingerprints)

    fingerprints = []
    for filename in filenames:
        data = Data.load(filename)
        fingerprints.extend(data.fingerprints)

    tl = TransferLearning()
    tl.display(fingerprints)
