import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
import os
from scipy.spatial import distance_matrix

from astropy import units
from astropy.coordinates import SkyCoord

import utils

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("TransferLearningDisplay")
log.setLevel(logging.WARNING)


class TransferLearningDisplay:
    def __init__(self, similarity_measure, ra_dec={}):
        self.similarity = similarity_measure
        self.fig = None
        self.axis = None
        self.info_axis = None
        self.info_text = None
        self.sub_windows = None

        # Dictionary of RA DEC information 
        # The dictionary should be keyed by filename and will have an 'ra' and 'dec'
        # as sub-keys
        self.ra_dec = ra_dec 

        self._onmove_color = (0.1, 0.6, 0.1)

    def add_axes(self, ax):
        pass 

    def show(self, fingerprints):
        """
        Overall display layout.

        :param fingerprints:
        :return:
        """
        plt.show(block=False)
        plt.ion()

        self.similarity.calculate(fingerprints)

        self.fig = plt.figure(1, figsize=[16, 10])
        plt.gcf()

        # Create the axis for the similarity plot
        self.axis = plt.axes([0.05, 0.05, 0.3, 0.45])

        self._aitoff = Aitoff([0.05, 0.55, 0.3, 0.3], parent=self)
        self.add_axes(self._aitoff.get_axes())
        self._aitoff.onmove_color = self._onmove_color

        # Sub window for on move closest fingerprint
        self.axis_closest = plt.axes([0.35, 0.01, 0.15, 0.15])
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
        self._hexbin = self.similarity.display(self.axis)

        # Display the information text area
        self.info_axis = plt.axes([0.75, 0.05, 0.2, 0.85])
        self.info_axis.set_axis_off()
        self.info_axis.set_xticks([])
        self.info_axis.set_yticks([])
        self.info_axis.set_xlabel('')
        self.info_axis.set_ylabel('')
        self.info_axis.set_ylim(self.info_axis.get_ylim()[::-1])  # invert the axis
        self.info_text = self.info_axis.text(0, 0.83, 'Loading...', fontsize=8, va='bottom')

        # Display the 9 sub-windows
        self.sub_windows = []
        self.sub_windows_fingerprint = [None]*9
        self.sub_window_current = None
        self.sub_data = []
        self.fingerprint_points = []
        for row in range(3):
            for col in range(3):
                # rect = [left, bottom, width, height]
                tt = plt.axes([0.35 + 0.13 * col, 0.55 - 0.17 * row, 0.15, 0.15])
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

    def get_ra_dec(self, fingerprint):
        ra, dec = fingerprint['tldp'].radec
        if ra is not None and dec is not None:
            coords = SkyCoord(ra=ra, dec=dec, unit='degree')
            ra = coords.ra.wrap_at(180 * units.deg).radian
            dec = coords.dec.radian
        else:
            ra,dec = None, None
        return ra, dec


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

            thetitle = close_fingerprint['tldp']._file_meta['filename'].split('/')[-1]
            #thetitle += ' ' + ','.join([repr(x) for x in close_fingerprint['tldp'].data_processing])

            self.axis_closest.set_title(thetitle, fontsize=8)
            self.fig.canvas.blit(self.axis_closest.bbox)
            self.axis_closest.redraw_in_frame()

            self.axis_closest.text(0.02, 1.0, r'$\bullet$', fontsize=24,
                    color=self._onmove_color, transform=self.axis_closest.transAxes)

            self._aitoff.onmove(event, close_fingerprint)

        # Do something based on being in one of the 9 sub windows
        elif event.inaxes in self.sub_windows:
            subwindow_index = self.sub_windows.index(event.inaxes)
            self._display_subwindow_meta_fingerprint(subwindow_index)

    def _add_subwindow_outline(self, index):
        for side in ['top', 'bottom', 'left', 'right']:
            self.sub_windows[index].spines[side].set_visible(True);
            self.sub_windows[index].spines[side].set_lw(2);
            self.sub_windows[index].spines[side].set_color('red');

    def _remove_subwindow_outline(self, index):
        for side in ['top', 'bottom', 'left', 'right']:
            self.sub_windows[index].spines[side].set_visible(False);

    def _display_subwindow_meta_fingerprint(self, index):

        if self.sub_windows[0] is not None and not index == self.sub_window_current:
            self.sub_window_current = index

            to_disp = ''
            # Add the meta information
            for k, v in self.sub_windows_fingerprint[index]['tldp']._file_meta['meta'].items():
                to_disp += '{}: {}\n'.format(k, v)

            # Add the fingerprint similarity
            to_disp += '\nFingerprints\n------------\n'
            for p in self.sub_windows_fingerprint[index]['predictions'][:8]:
                to_disp += '{:18s} {:4.4f}\n'.format(p[1], p[2])

            self._update_text(to_disp)

            # Add outline to hovered subwindow
            self._add_subwindow_outline(index)
            for index in set(range(9)) - set([index]):
                self._remove_subwindow_outline(index)

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
            points = []
            self._update_text('Displaying result...')
            for ii, (fpoint, distance, fingerprint) in enumerate(close_fingerprints):

                self.fingerprint_points[ii] = fpoint

                # Zero out and show we are loading -- shoul3d be fast.3
                self.sub_windows[ii].set_title('Loading...', fontsize=8)
                self.sub_data[ii].set_data(np.zeros((224, 224)))
                self.sub_windows[ii].redraw_in_frame()

                # Associate the fingerprint to the subwindow so when
                # we hover over the window we can get info about it.
                self.sub_windows_fingerprint[ii] = fingerprint

                # Display the data information for the fingerprint cutout
                row = fingerprint['row_min'], fingerprint['row_max']
                col = fingerprint['col_min'], fingerprint['col_max']

                # Show new data and set title
                self.sub_data[ii].set_data(utils.rgb2plot(
                    fingerprint['tldp'].display(row, col)
                ))

                thetitle = fingerprint['tldp']._file_meta['filename'].split('/')[-1]

                # Update the title on the window
                self.sub_windows[ii].set_title('{}) {:0.3f} {}'.format(
                    (ii+1), distance, thetitle), fontsize=8)
                self.sub_windows[ii].redraw_in_frame()


            self._aitoff.onclick(event, close_fingerprints)

            self.sub_window_current = 1
            self._display_subwindow_meta_fingerprint(0)


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


class AxExtra:
    def __init__(self):
        self._axes_limits = None

    def get_axes(self):
        return self._axes_limits

class Aitoff(AxExtra):

    def __init__(self, axes_limits, parent):
        self._parent = parent
        self._axes_limits = axes_limits

        # If we have ra_dec then let's display the Aitoff projection axis
        self.axis_aitoff = plt.axes(self._axes_limits, projection="aitoff")
        self.axis_aitoff.grid('on')
        self.axis_aitoff.set_xlabel('RA')
        self.axis_aitoff.set_ylabel('DEC')
        self._onmove_point = None
        self._onclick_points = {} 
        self._parent.fig.canvas.draw()
        self.axis_aitoff_background = self._parent.fig.canvas.copy_from_bbox(self.axis_aitoff.bbox)
        self._axis_aitoff_text_labels = []

        self._onmove_color = (0.1, 0.6, 0.1)

    @property
    def onmove_color(self):
        return self._onmove_color

    @onmove_color.setter
    def onmove_color(self, value):
        self._onmove_color = value

    def onmove(self, event, close_fingerprint):
        # Update the mbitoff figure as well
        self._parent.fig.canvas.restore_region(self.axis_aitoff_background)

        # Draw the blue dots and numbers
        for tt in self._axis_aitoff_text_labels:
            self.axis_aitoff.draw_artist(tt)

        ra, dec = self._parent.get_ra_dec(close_fingerprint)
        if ra is not None and dec is not None:
            if self._onmove_point is not None:
                self._onmove_point[0].set_data(ra, dec)
            else:
                self._onmove_point = self.axis_aitoff.plot(ra, dec, 
                        'o', color=self._onmove_color)
        self.axis_aitoff.draw_artist(self._onmove_point[0])
        self._parent.fig.canvas.blit(self.axis_aitoff.bbox)

    def onclick(self, event, close_fingerprints):

        # Delete the artists
        for x in self._axis_aitoff_text_labels:
            x.remove()
        self._axis_aitoff_text_labels = []

        points = []
        for ii, (fpoint, distance, fingerprint) in enumerate(close_fingerprints):
        
            # Add point to Aitoff plot
            ra, dec = self._parent.get_ra_dec(fingerprint)
            if self._onclick_points and ii in self._onclick_points:
                self._onclick_points[ii][0].set_data(ra, dec)
            else:
                self._onclick_points[ii] = self.axis_aitoff.plot(ra, dec, 'bo', label=str(ii))
            points.append([ra,dec])

        self._parent.fig.canvas.restore_region(self.axis_aitoff_background)

        # annotate the points in the aitoff plot
        points = np.array(points)
        d = distance_matrix(points, points)

        rows = set(range(points.shape[0]))
        groups = {}

        while len(rows) > 0:
            row = rows.pop()
            close = np.nonzero(d[row] < 0.01)[0]
            rows = rows - set(list(close))
            groups[row] = close

        for k,v in groups.items():
            tt = self.axis_aitoff.text(points[k][0]+0.05, points[k][1]+0.05, 
                    ','.join([str(x+1) for x in v]))
            self._axis_aitoff_text_labels.append(tt)
            self.axis_aitoff.draw_artist(tt)

        self._parent.fig.canvas.blit(self.axis_aitoff.bbox)

        self.axis_aitoff_background = self._parent.fig.canvas.copy_from_bbox(self.axis_aitoff.bbox)

