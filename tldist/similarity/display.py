import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from scipy.spatial import distance_matrix

from tldist.database import get_database
from tldist.fingerprint import Fingerprint
from tldist.cutout import Cutout

from astropy.coordinates import SkyCoord
from astropy import units

from ..tl_logging import get_logger
import logging
log = get_logger('display', '/tmp/mylog.log', level=logging.WARNING)


class SimilarityDisplay(object):

    def __init__(self, similarity, db):

        plt.ion()

        self._similarity = similarity
        self._db = db

        # Setup the figure
        self._figure = plt.figure(2, figsize=[16, 7.5])

        # Left is aitoff and similarity, Right is similar images
        main_grid = gridspec.GridSpec(1, 2)
        left_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_grid[0])
        aitoff_grid = left_grid[0, 0]
        similarity_main_grid = gridspec.GridSpecFromSubplotSpec(2, 4, left_grid[1, 0])

        right_grid = main_grid[1]

        # The NxM set of similar images
        self._similar_images_axis = SimilarImages(right_grid, [3, 3], db)
        # TODO: this should be removed in the future
        self._similar_images_axis.set_db(db)

        # The current image displayed as one moves around the
        # similarity plot (e.g., tSNE)
        self._current_image_axis = Image(similarity_main_grid[1, -1])
        #self._current_image_axis.imshow(np.ones((224, 224)))
        self._current_image_axis_fingerprint_uuid = ''  # caching
        self._current_image_axis_time_update = 0

        # The AITOFF plot
        self._aitoff_axis = Aitoff(aitoff_grid)

        # Display the similarity matrix based on display method in
        # the similarity subclass (e.g,. tSNE, Jaccard etc)
        self._similarity_matrix = Image(similarity_main_grid[:, :-1])
        self._similarity.display(self._similarity_matrix)

        # Connect the callbacks
        self._ccid = self._figure.canvas.mpl_connect('button_press_event', self._click_callback)
        self._mcid = self._figure.canvas.mpl_connect('motion_notify_event', self._move_callback)

        # Initialize the 9 similar images with the first 9 fingerprints
        # TODO: Fix this so it is the first and the actual similar ones
        fingerprints = self._db.find('fingerprint')
        for ii in range(9):
            f = fingerprints[ii]
            cutout_uuid = f.cutout_uuid
            cutout = Cutout.factory(cutout_uuid, self._db)
            d = cutout.data
            self._similar_images_axis.set_image(ii, cutout.get_data(),
                                                str(ii+1) + ') ' + os.path.basename(d.location), fingerprints[ii])

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    def _move_callback(self, event):
        """
        Move callback:  This will essentially just update the hover image
        and the point

        """
        log.debug('---------------------------------------------')

        # If we are hovering over the similarity plot
        if event.inaxes == self._similarity_matrix._axes:
            point = event.ydata, event.xdata
            close_fingerprint_uuids = self._similarity.find_similar(point, 1)

            now = time.time()

            fingerprint_uuid = close_fingerprint_uuids[0]['fingerprint_uuid']
            if (not fingerprint_uuid == self._current_image_axis_fingerprint_uuid and
               (now - self._current_image_axis_time_update) > 0.1):
                try:
                    log.debug('going to show fingerprint {}'.format(fingerprint_uuid))
                    self._current_image_axis_fingerprint_uuid = fingerprint_uuid

                    fingerprint = Fingerprint.factory(fingerprint_uuid, self._db)

                    # Get the data location
                    log.debug('Loading data {}'.format(fingerprint.cutout_uuid))
                    cutout = Cutout.factory(fingerprint.cutout_uuid, self._db)

                    # Display the image
                    log.debug('Imshow on _current_image_axis')
                    self._current_image_axis.imshow(cutout.get_data())

                    # Display the location on the Aitoff plot
                    self._aitoff_axis.on_move(cutout.data.radec)
                except Exception as e:
                    log.debug(e)

        # If we are hovering over one of the NxN similar images
        elif self._similar_images_axis.hover_in(event.inaxes) is not None:
            log.debug('In one of the similar axes')
            self._similar_images_axis.show_fingerprint(event.inaxes)

        self._move_callback_processing = False

    def _click_callback(self, event):

        # Click in the simiarlity matrix axis
        if event.inaxes == self._similarity_matrix._axes:
            try:
                point = event.ydata, event.xdata
                close_fingerprints = self._similarity.find_similar(point, n=9)
                close_fingerprint_uuids = [fp['fingerprint_uuid'] for fp in close_fingerprints]
                self._similar_images_axis.set_images(close_fingerprints)

                aitoff_fingerprints = []
                for cfp in close_fingerprints:
                    fingerprint = Fingerprint.factory(cfp['fingerprint_uuid'], self._db)
                    cutout = Cutout.factory(fingerprint.cutout_uuid, self._db)
                    data = cutout.data
                    a = (cfp['tsne_point'], cfp['distance'], data.radec, fingerprint)
                    aitoff_fingerprints.append(a)
                self._aitoff_axis.on_click(aitoff_fingerprints)
            except Exception as e:
                log.debug('EXCEPTION {}'.format(e), exc_info=True)

        # Click in the aitoff axis
        elif event.inaxes == self._aitoff_axis._axis:
            # TODO: This is wrong, need inverse
            ra, dec = self._aitoff_axis.convert_ra_dec(event.ydata, event.xdata)

        # Click on one of the 9 similar images
        elif event.inaxes in [x._axes for x in self._similar_images_axis.axes]:
            fingerprint_uuid = event.inaxes._imdata.uuid

            index = self._similarity.fingerprint_uuids.index(fingerprint_uuid)

            point = similarity_tsne._Y[index]
            close_fingerprint_uuids = self._similarity.find_similar(point)
            self._similar_images_axis.set_images(close_fingerprint_uuids)


class SimilarImages(object):
    def __init__(self, sub_gridspec, rows_cols, db):

        sim_images_grid = gridspec.GridSpecFromSubplotSpec(rows_cols[0], rows_cols[1]+1, subplot_spec=sub_gridspec)

        self._rows_cols = rows_cols
        self._db = db

        self._fingerprints = []
        self.axes = []

        ii = 0
        for row in range(rows_cols[0]):
            for col in range(rows_cols[1]):
                image = Image(sim_images_grid[row, col])
                image.store({'type': 'similar', 'number': ii})
                self.axes.append(image)
                ii = ii + 1

        self._text_axis = plt.subplot(sim_images_grid[:, -1])
        self._text_axis.set_ylim((1, 0))
        self._text_axis.set_frame_on(False)
        self._text_axis.set_xticks([])
        self._text_axis.set_yticks([])
        self._fingerprint_text = self._text_axis.text(0, 0, '', fontsize=8, va='top')

    def set_db(self, db):
        self._db = db

    def hover_in(self, hover_axes):
        """
        Check to see if the passed in axes is one we are repsonsible for.
        """
        log.info('hover_axes {}'.format(hover_axes))
        if hover_axes is not None and hover_axes in [x._axes for x in self.axes]:
            return [x._axes for x in self.axes].index(hover_axes)
        else:
            return None

    def show_fingerprint(self, hover_axes):
        """
        Check to see if the passed in axes is one we are repsonsible for.
        """
        log.info('hover axis {}'.format(hover_axes))
        index = [x._axes for x in self.axes].index(hover_axes)

        try:
            # Draw the text
            start = time.time()

            cutout = Cutout.factory(self._fingerprints[index].cutout_uuid, self._db)

            # Meta information
            meta_text = os.path.basename(cutout.data.location) + '\n\n'
            for k, v in cutout.data.meta.items():
                meta_text += '{}: {}\n'.format(k, v)

            # Fingerprint information
            meta_text += '\n'
            for ii in range(5):
                meta_text += '{}: {:0.3}\n'.format(
                        self._fingerprints[index].predictions[ii][1],
                        self._fingerprints[index].predictions[ii][2])

            self._fingerprint_text.set_text(meta_text)

            # Unhighlight other axes
            for im in list(set(self.axes) - set([self.axes[index]])):
                im.hide_outline()

            # Highlight the axis
            log.debug('Add outline for subwindow')
            self.axes[index].show_outline()

        except Exception as e:
            log.error('show_fingerprint {}'.format(e))
        log.debug('text drawing took {}'.format(time.time() - start))

    def set_images(self, fingerprints):
        self._fingerprints = fingerprints

        self._fingerprints = []
        for ii, fp in enumerate(fingerprints):
            # Get the fingerprint
            f_uuid = fp['fingerprint_uuid']
            fingerprint = Fingerprint.factory(f_uuid, self._db)
            self._fingerprints.append(fingerprint)

            # Get the data location
            cutout = Cutout.factory(fingerprint.cutout_uuid, self._db)

            self.set_image(ii, cutout.get_data(), fingerprint=fingerprint)

    def set_image(self, num, data, title='', fingerprint={}):
        # Display image
        self.axes[num].imshow(data, title=title)

        self.axes[num].store(fingerprint)
        # Show on Aitoff


class Image(object):
    def __init__(self, grid_spec):
        self._axes = plt.subplot(grid_spec)
        self._axes_data = None
        self._spines_visible = False
        self._outline = None

    def _rgb2plot(self, data):
        """
        Convert the input data to RGB. This is basically clipping and cropping the intensity range for display

        :param data:
        :return:
        """

        mindata, maxdata = np.percentile(data[np.isfinite(data)], (0.01, 99.0))
        return np.clip((data - mindata) / (maxdata-mindata) * 255, 0, 255).astype(np.uint8)

    def store(self, thedict):
        self._axes._imdata = thedict

    def get(self):
        return self._axes._imdata

    def imshow(self, data, title=''):

        data = self._rgb2plot(data)

        if self._axes_data is None:
            self._axes_data = self._axes.imshow(data, cmap=plt.gray())
            self._axes.set_xticks([])
            self._axes.set_yticks([])
            self._axes.set_xlabel('')
            self._axes.set_ylabel('')
            self._axes.get_figure().canvas.draw()
        else:
            self._axes_data.set_data(data)

        if title:
            self._axes.set_title(title, fontsize=8)

        self._axes.get_figure().canvas.blit(self._axes.bbox)

        # This is definitely needed to update the image if we
        # are calling this from a script.
        if not matplotlib.get_backend() == 'nbAgg':
            self._axes.redraw_in_frame()

    def plot(self, x, y, title=''):
        self._axes.plot(x, y, '.')
        if title:
            self._axes.set_title(title, fontsize=8)

    def grid(self, onoff='on'):
        self._axes.grid(onoff)

    def set_title(self, title):
        self._axes.set_title(title)

    def show_outline(self):
        log.info('')
        try:
            if self._outline is None:
                self._outline = matplotlib.patches.Rectangle((0, 0), 225, 225, linewidth=4, edgecolor='#ff7777', facecolor='none')
                self._axes.add_patch(self._outline)
            else:
                self._outline.set_visible(True)

            self._axes.get_figure().canvas.blit(self._axes.bbox)
            if not matplotlib.get_backend() == 'nbAgg':
                self._axes.redraw_in_frame()
            self._axes.draw_artist(self._outline)
        except Exception as e:
            log.error(e)

    def hide_outline(self):
        log.info('')
        try:
            if self._outline is not None:
                self._outline.set_visible(False)
                self._axes.draw_artist(self._outline)
                self._axes.get_figure().canvas.blit(self._axes.bbox)
                if not matplotlib.get_backend() == 'nbAgg':
                    self._axes.redraw_in_frame()
        except Exception as e:
            log.error(e)


class Aitoff(object):

    def __init__(self, grid_spec):
        # If we have ra_dec then let's display the Aitoff projection axis
        self._axis = plt.subplot(grid_spec, projection="aitoff")
        self._axis.grid('on')
        self._axis.set_xlabel('RA')
        self._axis.set_ylabel('DEC')
        self._onmove_point = None
        self._onclick_points = {}
        self._axis_text_labels = []

        self._axis_aitoff_text_labels = []

        self._onmove_point = None
        self._onmove_color = (0.1, 0.6, 0.1)

        self._axis.get_figure().canvas.draw()
        self._axis_background = self._axis.get_figure().canvas.copy_from_bbox(self._axis.bbox)

    def convert_ra_dec(self, ra, dec):
        if ra is not None and dec is not None:
            coords = SkyCoord(ra=ra, dec=dec, unit='degree')
            ra = coords.ra.wrap_at(180 * units.deg).radian
            dec = coords.dec.radian
        else:
            ra, dec = None, None
        return ra, dec

    def on_click(self, close_fingerprints):

        if not isinstance(close_fingerprints, list) or not isinstance(close_fingerprints[0], tuple):
            log.error('Wrong thing passed to Aitoff on_click.  It must be a list of tuples where each tuple contains piont, distnace and fingerprint')
            raise Exception('Wrong thing passed to Aitoff')

        try:
            for x in self._axis_aitoff_text_labels:
                x.remove()
            self._axis_aitoff_text_labels = []

            points = []
            for ii, (fpoint, distance, radec, fingerprint) in enumerate(close_fingerprints):

                # Add point to Aitoff plot
                ra, dec = self.convert_ra_dec(*radec)
                if self._onclick_points and ii in self._onclick_points:
                    self._onclick_points[ii][0].set_data(ra, dec)
                else:
                    self._onclick_points[ii] = self._axis.plot(ra, dec, 'bo', label=str(ii))
                points.append([ra, dec])

            self._axis.get_figure().canvas.restore_region(self._axis_background)

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

            for k, v in groups.items():
                tt = self._axis.text(points[k][0]+0.05, points[k][1]+0.05,
                                     ','.join([str(x+1) for x in v]))
                self._axis_aitoff_text_labels.append(tt)
                self._axis.draw_artist(tt)

            self._axis.get_figure().canvas.blit(self._axis.bbox)

            self._axis.get_figure().canvas.draw()
            self._axis_background = self._axis.get_figure().canvas.copy_from_bbox(self._axis.bbox)
        except Exception as e:
            log.error(e)

    def on_move(self, radec):
        try:
            ra, dec = self.convert_ra_dec(radec[0], radec[1])

            # Update the mbitoff figure as well
            self._axis.get_figure().canvas.restore_region(self._axis_background)

            if self._onmove_point is not None:
                self._onmove_point[0].set_data(ra, dec)
            else:
                self._onmove_point = self._axis.plot(ra, dec, 'o', color=self._onmove_color)

            self._axis.draw_artist(self._onmove_point[0])
            self._axis.get_figure().canvas.blit(self._axis.bbox)
        except Exception as e:
            log.error('display point {}'.format(e))


if __name__ == '__main__':
    db = get_database('blitzdb', '/tmp/basic_notebook.db')
    similarities = db.find('similarity')
    similarity_tsne = similarities[0]

    simdisp = SimilarityDisplay(similarity_tsne, db)
