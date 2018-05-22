import os
import traceback
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from itertools import groupby

from scipy.spatial import distance_matrix
from transfer_learning.database import get_database
from transfer_learning.fingerprint import Fingerprint
from transfer_learning.cutout import Cutout

from astropy.coordinates import SkyCoord
from astropy import units

from ..tl_logging import get_logger
import logging
log = get_logger('display')


class SimilarityDisplay(object):

    def __init__(self, similarity, db):

        plt.ion()

        self._similarity = similarity
        self._db = db

        # Setup the figure
        self._figure = plt.figure(2, figsize=[14, 6.5])

        # Left is aitoff and similarity, Right is similar images
        main_grid = gridspec.GridSpec(1, 2)
        left_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_grid[0])
        aitoff_grid = left_grid[0, 0]
        similarity_main_grid = gridspec.GridSpecFromSubplotSpec(2, 4, left_grid[1, 0])

        right_grid = main_grid[1]

        # The NxM set of similar images
        self._similar_images = SimilarImages(right_grid, db)

        # The current image displayed as one moves around the
        # similarity plot (e.g., tSNE)
        self._current_image_axis = Image(similarity_main_grid[1, -1])
        #self._current_image_axis.imshow(np.ones((224, 224)))
        self._current_image_axis_fingerprint_uuid = ''  # caching
        self._current_image_axis_time_update = 0

        self._current_image_text_axis = plt.subplot(similarity_main_grid[0, -1])
        self._current_image_text_axis.set_ylim((1, 0))
        self._current_image_text_axis.set_frame_on(False)
        self._current_image_text_axis.set_xticks([])
        self._current_image_text_axis.set_yticks([])
        self._current_image_text_axis_text = self._current_image_text_axis.text(0, 0, '', fontsize=8, va='top')
        plt.draw()

        # The AITOFF plot
        self._aitoff_axis = Aitoff(aitoff_grid)

        # Display the similarity matrix based on display method in
        # the similarity subclass (e.g,. tSNE, Jaccard etc)
        self._similarity_matrix = Image(similarity_main_grid[:, :-1])
        self._similarity.display(self._similarity_matrix)

        # Connect the callbacks
        self._ccid = self._figure.canvas.mpl_connect('button_press_event', self._click_callback)
        self._mcid = self._figure.canvas.mpl_connect('motion_notify_event', self._move_callback)

        # Initialize the similar images with the fingerprints closest to the
        # (0,0) of the similarity plot
        close_fingerprints = self._similarity.find_similar((0, 0), 9)
        self._similar_images.set_images([Fingerprint.factory(x['fingerprint_uuid'], db)
                                              for x in close_fingerprints])

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
                    log.debug('Imshow on _current_image_axis with title {}'.format(cutout.data.location))
                    # Set the text above the "on move" image
                    current_text_display = os.path.basename(cutout.data.location) +'\n\n'
                    for p in fingerprint.predictions[:5]:
                        current_text_display += '{} {:.3}\n'.format(p[1], p[2])
                    self._current_image_text_axis_text.set_text(current_text_display)

                    self._current_image_axis.imshow_cutout(cutout)

                    # Display the location on the Aitoff plot
                    self._aitoff_axis.on_move(cutout.data.radec)
                except Exception as e:
                    log.debug(e)

        # If we are hovering over one of the NxN similar images
        elif self._similar_images.hover_in(event.inaxes) is not None:
            log.debug('In one of the similar axes')
            self._similar_images.show_fingerprint(event)

        self._move_callback_processing = False

    def _update_from_similarity(self, point):
        try:
            close_fingerprints = self._similarity.find_similar(point, n=9)
            self._similar_images.set_images([
                    Fingerprint.factory(fp['fingerprint_uuid'], self._db)
                    for fp in close_fingerprints
                    ])

            # Need some logic to choose between showing the whole image with borders
            # or the cutouts
            aitoff_fingerprints = []
            for cfp in close_fingerprints:
                fingerprint = Fingerprint.factory(cfp['fingerprint_uuid'], self._db)
                cutout = Cutout.factory(fingerprint.cutout_uuid, self._db)
                data = cutout.data
                log.debug('{} {}'.format(cfp, data.location))
                a = (cfp['tsne_point'], cfp['distance'], data.radec, fingerprint)
                aitoff_fingerprints.append(a)
            self._aitoff_axis.on_click(aitoff_fingerprints)
        except Exception as e:
            log.debug('EXCEPTION {}'.format(e), exc_info=True)

    def _click_callback(self, event):

        # Click in the simiarlity matrix axis
        if event.inaxes == self._similarity_matrix._axes:
            self._update_from_similarity((event.ydata, event.xdata))

        # Click in the aitoff axis
        elif event.inaxes == self._aitoff_axis._axis:
            # TODO: This is wrong, need inverse
            ra, dec = self._aitoff_axis.convert_ra_dec(event.ydata, event.xdata)

        # Click on one of the "9" similar images
        elif event.inaxes in [image.axis for image in self._similar_images.images]:
            # Determine the cutout which was click on (might be the actual image
            # or might be a sub part of the displayed image)
            cutout = self._similar_images.find_cutout((event.ydata, event.xdata), event.inaxes)
            log.debug('closest cutout is {}'.format(cutout))

            sim_point = self._similarity.cutout_point(cutout)
            log.debug('sim_point is {}'.format(sim_point))

            self._update_from_similarity(sim_point)


class SimilarImages(object):
    """
    Interested in 9 similar

    1. Carina image (1 image, multiple cutouts on image)

       - one large image
       - show all the cutouts on the one image

    2. Color Hubble images (many images, many cutouts from each image)

       - show up to 9 images
       - show cutout on each image

    3. B/W Hubble images (many images, 1 cutout from each image)

       - show 9 images
       - don't really need to show the cutout border on the image


    pass the fingerprints into SimilarImages

    determine the number of unique datasets

    create a grid based on the number of unique datasets

    Loop through the fignerprints and display the underlying image and cutout
    """

    def __init__(self, sub_gridspec, db):
        self._grid = sub_gridspec
        self._db = db

        self._fingerprints = []
        self._images = []

        # Grid for the similar images.
        self._sim_images_and_text_grid = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=sub_gridspec)

        # Text grid.
        self._text_axis = plt.subplot(self._sim_images_and_text_grid[:, -1])
        self._text_axis.set_ylim((1, 0))
        self._text_axis.set_frame_on(False)
        self._text_axis.set_xticks([])
        self._text_axis.set_yticks([])
        self._fingerprint_text = self._text_axis.text(0, 0, '', fontsize=8, va='top')

    @property
    def images(self):
        return self._images

    def set_db(self, db):
        self._db = db

    def hover_in(self, hover_axes):
        """
        Check to see if the passed in axes is one we are repsonsible for.
        """
        log.info('hover_axes {}'.format(hover_axes))
        if hover_axes is not None and hover_axes in [x._axes for x in self._images]:
            return [image.axis for image in self._images].index(hover_axes)
        else:
            return None

    def find_cutout(self, point, axis):
        # find the Image that was clicked on
        image = [image for image in self._images if image.axis == axis][0]
        log.debug('Clicked on {}'.format(image))

        fingerprint = image.find_fingerprint(point, self._db)
        log.debug('cutout was {}'.format(fingerprint.cutout))

        return fingerprint.cutout

    def show_fingerprint(self, event):
        """
        Check to see if the passed in axes is one we are repsonsible for.
        """
        log.info(event)

        hover_axes = event.inaxes

        index = [image.axis for image in self._images].index(hover_axes)
        fingerprint = self._images[index].find_fingerprint((event.xdata, event.ydata), self._db)
        image_number = self._fingerprints.index(fingerprint) + 1
        log.debug('image_number {}'.format(image_number))

        try:
            # Draw the text
            start = time.time()

            # Meta information
            meta_text = '{}) '.format(image_number) + os.path.basename(fingerprint.cutout.data.location) + '\n'
            for k, v in fingerprint.cutout.data.meta.items():
                meta_text += '{}: {}\n'.format(k, v)

            # Fingerprint information
            meta_text += '\n'
            for ii in range(5):
                meta_text += '{}: {:0.3}\n'.format(
                        fingerprint.predictions[ii][1],
                        fingerprint.predictions[ii][2])

            self._fingerprint_text.set_text(meta_text)
            plt.draw()

            # Highlight the axis
            log.debug('Add outline for subwindow')
            self._images[index].show_outline(fingerprint)

        except Exception as e:
            log.error('show_fingerprint {}'.format(e))
            print(traceback.format_exc())

        log.debug('text drawing took {}'.format(time.time() - start))

    def _uniquify(self, thelist):
        seen = set()
        return [x for x in thelist if not (x in seen or seen.add(x))]

    def set_images(self, fingerprints):
        log.info('fingerprints is {}'.format(fingerprints))

        self._fingerprints = fingerprints

        # Determine the number of unqiue data locations and
        # therefore the number of images we need to display
        locations = self._uniquify([x.cutout.data.location for x in fingerprints])
        log.debug('locations {}'.format(locations))

        # TODO:  NEED SOMETHING SMARTER HERE AS IT IS REDRAWING THE IMAGE FOR EVERY CUTOUT

        # Create the grid of axes for each unique data location
        n_unique = len(locations)
        n_sqrt = np.sqrt(n_unique)
        nrows = int(np.ceil(n_sqrt))
        ncols = int(np.ceil(n_unique/nrows))
        log.debug('nrows, ncols {} {}'.format(nrows, ncols))
        self._sim_images_grid = gridspec.GridSpecFromSubplotSpec(
                nrows, ncols,
                subplot_spec=self._sim_images_and_text_grid[0, :-1])

        # Group the fingerprints by location so we are doing one
        # plot and then just adding the fingerprints on top.
        self._images = []
        for location, fs in groupby(fingerprints, lambda x: x.cutout.data.location):
            tfs = list(fs)
            ilocation = locations.index(location)
            row, col = ilocation // ncols, ilocation % ncols
            im = Image(self._sim_images_grid[row, col])
            im.imshow(tfs, cutout_numbers=[(fingerprints.index(f)+1) for f in tfs])
            self._images.append(im)

class Image(object):
    def __init__(self, grid_spec):
        self._axes = plt.subplot(grid_spec)
        self._axes_data = None
        self._spines_visible = False
        self._outlines = None
        self._fingerprints = None

    def _rgb2plot(self, data):
        """
        Convert the input data to RGB. This is basically clipping and cropping the intensity range for display

        :param data:
        :return:
        """

        mindata, maxdata = np.percentile(data[np.isfinite(data)], (0.01, 99.0))
        return np.clip((data - mindata) / (maxdata-mindata) * 255, 0, 255).astype(np.uint8)

    @property
    def axis(self):
        return self._axes

    @property
    def cutout(self):
        return self._fingerprint.cutout

    @property
    def location(self):
        return self._fingerprints[0].cutout.data.location

    def store(self, thedict):
        self._axes._imdata = thedict

    def get(self):
        return self._axes._imdata

    def _dist(self, point, bounding_box, requires_in=True):
        """
        Calculate the distance between the point and the bounding_box.
        """
        bb_center = ((bounding_box[3]+bounding_box[2])/2.0, (bounding_box[0]+bounding_box[1])/2.0)

        if requires_in and bounding_box[2] <= point[0] <= bounding_box[3] and bounding_box[0] <= point[1] <= bounding_box[1]:
            return (point[1]-bb_center[0])**2 + (point[0]-bb_center[1])**2
        elif requires_in:
            return np.Inf
        else:
            return (point[1]-bb_center[0])**2 + (point[0]-bb_center[1])**2

    def find_fingerprint(self, point, db):
        log.info('Find cutout closest to {}'.format(point))

        # Find all cutouts for this data
        cutouts = [fingerprint.cutout for fingerprint in self._fingerprints]
        distances = [self._dist(point, cutout.bounding_box) for cutout in cutouts]
        inds = np.argsort(distances)

        return self._fingerprints[inds[0]]

    def imshow_cutout(self, cutout, title=None, origin='lower'):

        data = cutout.get_data()
        data = self._rgb2plot(data)

        if self._axes_data is None:
            self._axes_data = self._axes.imshow(data, cmap=plt.gray(), origin=origin)
            self._axes.set_xticks([])
            self._axes.set_yticks([])
            self._axes.set_xlabel('')
            self._axes.set_ylabel('')
            self._axes.get_figure().canvas.draw()
        else:
            self._axes_data.set_data(data)

        if title is not None:
            self._axes.set_title(title, fontsize=8)

        self._axes.get_figure().canvas.blit(self._axes.bbox)

        # This is definitely needed to update the image if we
        # are calling this from a script.
        if not matplotlib.get_backend() == 'nbAgg':
            self._axes.redraw_in_frame()

    def imshow(self, fingerprints, title=None, origin='lower', cutout_numbers=None):
        """
        Display the cutouts (one from each fingerprint) onto the image.  It is assumed
        at this point that the fingerprints are all from the same image (but might be
        from different cutouts).
        """
        log.info('fingerprints paramter {}'.format(fingerprints))

        start = time.time()
        if self._fingerprints is None or not fingerprints[0].cutout.data.location == self._fingerprints[0].cutout.data.location:
            log.debug('appears to need to update the image with fingerprints {}'.format(fingerprints))
            self._fingerprints = fingerprints
            if 'Full' in fingerprints[0].cutout.generator_parameters['cutout_type']:
                data = fingerprints[0].cutout.data.get_data()
            else:
                data = fingerprints[0].cutout.data.get_data()

            data = self._rgb2plot(data)

            if self._axes_data is None:
                log.debug('do imshow')
                self._axes_data = self._axes.imshow(data, cmap=plt.gray(), origin=origin)
                self._axes.set_xticks([])
                self._axes.set_yticks([])
                self._axes.set_xlabel('')
                self._axes.set_ylabel('')
                self._axes.get_figure().canvas.draw()
            else:
                log.debug('setting data')
                self._axes_data.set_data(data)

        self._fingerprints = fingerprints

        self.store(fingerprints)
        log.debug('A Took {}s'.format(time.time() - start))

        self._outlines = []
        for ii, fingerprint in enumerate(self._fingerprints):
            border = fingerprint.cutout.bounding_box

            # Draw a border around the cutout in the image if the cutout
            # is a subset of the actual image (as opposed to the full image
            # itself.
            start = time.time()
            if border is not None:
                log.debug('border is {}'.format(border))

                # row, col <-> y, x
                outline = matplotlib.patches.Rectangle(
                        (border[2], border[0]), border[3]-border[2], border[1]-border[0],
                        linewidth=1, edgecolor='#ffff77', facecolor='none')
                self._axes.add_patch(outline)
                self._outlines.append(outline)

                if cutout_numbers is not None:
                    self._axes.text(border[2], border[0], '{}'.format(cutout_numbers[ii]), color='#ffff77')

            log.debug('B Took {}s'.format(time.time() - start))

        start = time.time()
        if title is not None:
            log.debug('setting title to {}'.format(title))
            self._axes.set_title(title, fontsize=8)
        log.debug('C Took {}s'.format(time.time() - start))

        start = time.time()
        self._axes.get_figure().canvas.blit(self._axes.bbox)

        # This is definitely needed to update the image if we
        # are calling this from a script.
        if not matplotlib.get_backend() == 'nbAgg':
            self._axes.redraw_in_frame()
        log.debug('D Took {}s'.format(time.time() - start))

    def plot(self, x, y, title=''):
        self._axes.plot(x, y, '.')
        if title:
            self._axes.set_title(title, fontsize=8)

    def grid(self, onoff='on'):
        self._axes.grid(onoff)

    def set_title(self, title):
        self._axes.set_title(title)

    def show_outline(self, fingerprint):
        """
        We need to show the outline of the cutout referred to by
        the fingerprint.
        """
        log.info('')

        # Determine the outline number from the fingerprint
        index = self._fingerprints.index(fingerprint)

        # Set that one to red
        self._outlines[index].set_edgecolor('#77ff77')

        # Set all others to yellow
        for ind in set(range(len(self._outlines))) - set([index]):
            self._outlines[ind].set_edgecolor('#ffff77')

        try:
            self._axes.get_figure().canvas.blit(self._axes.bbox)
            if not matplotlib.get_backend() == 'nbAgg':
                self._axes.redraw_in_frame()
            for outline in self._outlines:
                self._axes.draw_artist(outline)
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
