
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
import os

import utils

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("TransferLearningDisplay")
log.setLevel(logging.WARNING)


class TransferLearningDisplay:
    def __init__(self, similarity_measure):
        self.similarity = similarity_measure
        self.fig = None
        self.axis = None
        self.info_axis = None
        self.info_text = None
        self.sub_windows = None

    def show(self, fingerprints):
        """
        Overall display layout.

        :param fingerprints:
        :return:
        """
        plt.show(block=False)
        plt.ion()

        self.similarity.calculate(fingerprints)

        self.fig = plt.figure(1, figsize=[10, 6])
        plt.gcf()
        self.axis = plt.axes([0.05, 0.05, 0.45, 0.45])

        # Sub window for on move closest fingerprint
        self.axis_closest = plt.axes([0.5, 0.01, 0.2, 0.2])
        self.axis_closest.set_xticks([])
        self.axis_closest.set_yticks([])
        self.axis_closest.set_xlabel('')
        self.axis_closest.set_ylabel('')
        self._data_closest = self.axis_closest.imshow(np.zeros((224, 224)), cmap=plt.gray())
        self._text_closest = self.axis_closest.text(0.68, 0.4, ' ',
                                                    fontsize=6, verticalalignment='top',
                                                    transform=self.axis_closest.transAxes,
                                                    color='w')

        # Display the similarity plot (e.g., tSNE, jaccard etc)
        self.similarity.display(self.axis)

        # Display the information text area
        self.info_axis = plt.axes([0.75, 0.11, 0.3, 0.05])
        self.info_axis.set_axis_off()
        self.info_axis.set_xticks([])
        self.info_axis.set_yticks([])
        self.info_axis.set_xlabel('')
        self.info_axis.set_ylabel('')
        self.info_text = self.info_axis.text(0, 0, 'Loading...', fontsize=8)

        # Display the 9 sub-windows
        self.sub_windows = []
        self.sub_data = []
        self.fingerprint_points = []
        for row in range(3):
            for col in range(3):
                # rect = [left, bottom, width, height]
                tt = plt.axes([0.5 + 0.14 * col, 0.75 - 0.25 * row, 0.2, 0.2])
                tt.set_xticks([])
                tt.set_yticks([])
                sd = tt.imshow(np.zeros((224, 224)), cmap=plt.gray())
                self.sub_windows.append(tt)
                self.sub_data.append(sd)
                self.fingerprint_points.append((None,None))
        plt.show(block=False)

        # Attach the call backs
        self._cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._onmove)


    def _update_text(self, thetext):
        """
        Update the text in the information area

        :param thetext: the text to display
        :return:
        """
        self.info_text.set_text(thetext)
        plt.draw()

    def _onmove(self, event):
        """
        What to do when the mouse moves

        :param event: the event information
        :return:
        """
        log.debug('Moving to {}'.format(event))

        # Do something if in the similarity axis
        if event.inaxes == self.axis:
            point = event.ydata, event.xdata
            close_fingerprint = self.similarity.find_similar(point, n=1)[0][2]

            row = close_fingerprint['row_min'], close_fingerprint['row_max']
            col = close_fingerprint['col_min'], close_fingerprint['col_max']

            self._data_closest.set_data(utils.rgb2plot(
                close_fingerprint['tldp'].display(row, col)
            ))

            # Display image info
            #self._text_closest.set_text('\n'.join(['{} {:.4} '.format(*x[1:]) for x in close_fingerprint['predictions'][:5]]))

            thetitle = close_fingerprint['tldp'].filename.split('/')[-1]
            #thetitle += ' ' + ','.join([repr(x) for x in close_fingerprint['tldp'].data_processing])

            self.axis_closest.set_title(thetitle, fontsize=8)
            self.fig.canvas.blit(self.axis_closest.bbox)
            self.axis_closest.redraw_in_frame()

    def _onclick(self, event):
        """
        Mouse click event in the matplotlib window.

        :param event:
        :return:
        """
        log.debug('Clicked {}'.format(event))

        # Click in the similarity axis
        if event.inaxes == self.axis:
            point = event.ydata, event.xdata

            # Find all the similar data relative to the point that was clicked.
            self._update_text('Loading data...')
            close_fingerprints = self.similarity.find_similar(point)

            # Run through all the close fingerprints and display them in the sub windows
            self._update_text('Displaying result...')
            for ii, (fpoint, distance, fingerprint) in enumerate(close_fingerprints):

                self.fingerprint_points[ii] = fpoint

                # Zero out and show we are loading -- should be fast.3
                self.sub_windows[ii].set_title('Loading...', fontsize=8)
                self.sub_data[ii].set_data(np.zeros((224, 224)))
                self.sub_windows[ii].redraw_in_frame()

                # Display the data information for the fingerprint cutout
                row = fingerprint['row_min'], fingerprint['row_max']
                col = fingerprint['col_min'], fingerprint['col_max']

                # Show new data and set title
                self.sub_data[ii].set_data(utils.rgb2plot(
                    fingerprint['tldp'].display(row, col)
                ))

                thetitle = fingerprint['tldp'].filename.split('/')[-1]

                # Update the title on the window
                self.sub_windows[ii].set_title('{:0.3f} {}'.format(
                    distance, thetitle), fontsize=8)
                self.sub_windows[ii].redraw_in_frame()

            self._update_text('Click in the tSNE plot...')

        # Check to see if one of the 9 was clicked
        elif event.inaxes in self.sub_windows:

            # Get the index number of the image clicked on
            index = self.sub_windows.index(event.inaxes)

            # Create fake event as if we clicked this point
            new_ev = MouseEvent('faked', self.fig.canvas, 
                    self.fingerprint_points[index][0], self.fingerprint_points[index][1])
            new_ev.ydata = self.fingerprint_points[index][0]
            new_ev.xdata = self.fingerprint_points[index][1]
            new_ev.inaxes = self.axis
            self._onclick(new_ev)

    def _display_for_subwindow(self, index, aa):
        """
        Display the data in the subwindow

        :param index:
        :param aa:
        :return:
        """

        distance, fingerprint = aa

        # Zero out and show we are loading -- should be fast.3
        log.debug('Displaying fingerprint {}'.format(index))
        self.sub_windows[index].set_title('Loading...', fontsize=8)
        self.sub_data[index].set_data(np.zeros((224, 224)))
        self.sub_windows[index].redraw_in_frame()

        # Show new data and set title
        self.sub_data[index].set_data(utils.rgb2plot(
            fingerprint['data'].display(fingerprint['filename'],
                                        fingerprint['row'],
                                        fingerprint['col'])
        ))
        self.sub_windows[index].set_title('{:0.3f} {}'.format(
            distance,
            os.path.basename(fingerprint['filename'])), fontsize=8)

        self.sub_windows[index].redraw_in_frame()

