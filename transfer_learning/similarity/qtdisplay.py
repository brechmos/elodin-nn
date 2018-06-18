# -*- coding: utf-8 -*-
"""
This example demonstrates the use of pyqtgraph's dock widget system.

The dockarea system allows the design of user interfaces which can be rearranged by
the user at runtime. Docks can be moved, resized, stacked, and torn out of the main
window. This is similar in principle to the docking system built into Qt, but
offers a more deterministic dock placement API (in Qt it is very difficult to
programatically generate complex dock arrangements). Additionally, Qt's docks are
designed to be used as small panels around the outer edge of a window. Pyqtgraph's
docks were created with the notion that the entire window (or any portion of it)
would consist of dockable components.

"""
import os
import pickle
import logging
from collections import OrderedDict

import numpy as np
from scipy.spatial import distance_matrix

import astropy.units as u
from astropy.coordinates import SkyCoord

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyqtgraph.dockarea import Dock, DockArea
from PyQt5.QtCore import pyqtSignal

from transfer_learning.similarity import Similarity

#
# Setup the logging format
#
FORMAT = '%(levelname)-8s %(asctime)-15s %(name)-10s %(funcName)-10s %(lineno)-4d %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


class ColorImageView(pg.ImageView):
    """
    Wrapper around the ImageView to create a color lookup
    table automatically as there seem to be issues with displaying
    color images through pg.ImageView.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lut = None

    def updateImage(self, autoHistogramRange=True):
        super().updateImage(autoHistogramRange)
        self.getImageItem().setLookupTable(self.lut)


class FingerprintFilter(QtGui.QWidget):
    """
    Provides a place to be able to filter fingeprints
    based on a text input.
    """

    #
    # Signals
    #

    filter_entered = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        log.info('')

        super().__init__(*args, **kwargs)

        # Filter input Description
        self._filter_description = QtGui.QLabel()
        self._filter_description.setText('<b>Fingerprint Filter</b>: e.g., "s_ra > 0.1" or "πnematode > 0.1 and s_ra > 0.1"')

        # Filter text input
        self._filter_input = QtGui.QLineEdit()
        self._filter_input.textChanged.connect(self._text_changed)
        self._filter_input.editingFinished.connect(self._enter_pressed)
        self._filter_text = ''

        # Filter text result
        self._filter_results = QtGui.QLabel()

        # Filter overlapping bounding boxes
        self._filter_overlapping_bb = QtGui.QCheckBox('Overlapping Bounding Boxes')
        self._filter_overlapping_bb.setCheckState(True)
        self._filter_overlapping_bb.setTristate(False)
        self._filter_overlapping_bb.stateChanged.connect(self._filter_overlapping_bb_state_change)

        # Add both to this widget
        self.layout = QtGui.QGridLayout(self)
        self.layout.addWidget(self._filter_description, 0, 0, 1, 3)
        self.layout.addWidget(self._filter_input, 1, 0, 1, 2)
        self.layout.addWidget(self._filter_overlapping_bb, 1, 2)
        self.layout.addWidget(self._filter_results, 2, 0, 1, 3)

    def _get_filter(self):
        log.info('')
        return self._filter_input.text()

    @property
    def overlapping_bb(self):
        """
        Returns 2 if checked (assumably 1 if checked in tristate mode).
        """
        return self._filter_overlapping_bb.checkState() > 0

    #
    # Signal Handlers
    #

    def _text_changed(self, text):
        log.info('')

        self._filter_text = text

    def _enter_pressed(self):
        log.info('')

        # Fire update on filter
        self.filter_entered.emit(self._get_filter(), self.overlapping_bb)

    def set_error_background(self):
        self._filter_input.setStyleSheet("background-color: rgb(255, 0, 255);")

    def remove_error_background(self):
        self._filter_input.setStyleSheet("")

    def set_filter_results(self, thetext):
        self._filter_results.setText(thetext)

    def _filter_overlapping_bb_state_change(self, event):
        self.filter_entered.emit(self._get_filter())

class SimilarityPlotDock(Dock):
    """
    Shows the similarity image or scatter plot, the similar image when moving and the
    predictions of the image when moving.
    """

    def __init__(self, *args, **kwargs):
        log.info('')

        if 'parent' in kwargs:
            self._parent = kwargs['parent']
            del kwargs['parent']

        super().__init__(*args, **kwargs)

        #
        # Set up image or scatter plot that displays the similarities
        #
        self.plt = None
        if self._parent._similarity_tsne.data.shape[0] == 2 or self._parent._similarity_tsne.data.shape[1] == 2:
            self.sim_repr = pg.PlotWidget(title="Similarity", size=(150, 150))
            self.proxy_mm = pg.SignalProxy(self.sim_repr.scene().sigMouseMoved, rateLimit=5, slot=self.mouseMoved)
            self.proxy_mc = pg.SignalProxy(self.sim_repr.scene().sigMouseClicked, rateLimit=5, slot=self.mouseClicked)
        else:
            self.sim_repr = ColorImageView()
            self.sim_repr.setMinimumHeight(250)
            self.sim_repr.ui.histogram.hide()
            self.sim_repr.ui.menuBtn.hide()
            self.sim_repr.ui.roiBtn.hide()
            self.proxy_mm = pg.SignalProxy(self.sim_repr.scene.sigMouseMoved, rateLimit=5, slot=self.mouseMoved)
            self.proxy_mc = pg.SignalProxy(self.sim_repr.scene.sigMouseClicked, rateLimit=5, slot=self.mouseClicked)
        self.show_similarity_plot()

        #
        #  Show the image that is similar to where the mouse is
        #

        self.sim_image = ColorImageView()
        self.sim_image.ui.histogram.hide()
        self.sim_image.ui.menuBtn.hide()
        self.sim_image.ui.roiBtn.hide()
        self.sim_image.setMinimumHeight(250)

        #
        #  Show the fingerprint information about the image above.
        #

        self.text_widget = QtGui.QLabel('')
        self.text_widget.setMinimumWidth(150)
        self.text_widget.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        #
        # Fingerprint Filter widget
        #
        self.fingerprint_filter_widget = FingerprintFilter()
        self.proxy_filter_entered = self.fingerprint_filter_widget.filter_entered.connect(slot=self.filter_fingerprints)

        #
        # Add above to this Dock widget
        #

        self.addWidget(self.sim_repr, colspan=2, row=0, col=0)
        self.addWidget(self.sim_image, colspan=2, row=1, col=0)
        self.addWidget(self.fingerprint_filter_widget, colspan=3, row=2, col=0)
        self.addWidget(self.text_widget, row=0, rowspan=2, col=2)

    def filter_fingerprints(self, thefilter):
        """
        A list of strings is passed in and we only want to use fingerprints/data
        that have meta with one of the strings in the list.

        Parameters
        -----------
        thefilter : str
            String representation of the filtering.

        Notes
        -----
        """
        log.info('the filter {}'.format(thefilter))

        try:
            nleft = self._parent._similarity_tsne.set_filter_fingerprints(thefilter)
            self.fingerprint_filter_widget.set_filter_results('There are {} fingerprints left'.format(nleft))

            # Set the background color to white as there are no errors.
            self.fingerprint_filter_widget.remove_error_background()
        except Exception as e:
            #
            # If there was an error, then remove the filters and set
            # the background to red.
            #
            self._parent._similarity_tsne.set_filter_fingerprints('')
            self.fingerprint_filter_widget.set_error_background()
            self.fingerprint_filter_widget.set_filter_results('Error in parsing the filter')

        self.show_similarity_plot()

    def mouseMoved(self, event):
        """
        Mouse moved in the tSNE similarity plot.

        If this happens then we need to update the similar image
         and the similar image text.
        """
        log.info('{}'.format(event))

        #
        # Update the similar image.
        #

        # Get the point in proper coordinates
        if isinstance(self.sim_repr, pg.PlotWidget):
            pt_in_plt = self.sim_repr.plotItem.mapToView(event[0])
            point = (pt_in_plt.x(), pt_in_plt.y())
        else:
            v = self.sim_repr.view
            qpf = v.mapToItem(self.sim_repr.imageItem, event[0])
            point = (qpf.toPoint().x(), qpf.toPoint().y())

            similarity_data = self._parent._similarity_tsne.data

            if point[0] < 0 or point[1] < 0 or point[0] >= similarity_data.shape[0] or point[1] >= similarity_data.shape[1]:
                return

        log.debug('point moved {}'.format(point))
        # Find the closest fingerpint
        log.debug('calling find similar')
        fingerprints = self._parent._similarity_tsne.find_similar(point, n=1)

        # If none are returned (likely the filtering is too restrictive)
        if len(fingerprints) == 0:
            return

        f = fingerprints[0]['fingerprint']

        # Display
        log.debug('set image')
        self.sim_image.setImage(f.cutout.get_data())

        # Text Information
        loc = os.path.basename(f.cutout.data.location)
        self.text_widget.setText(loc + '\n\n' + '\n'.join(['{} {:0.3}'.format(p[1], p[2]) for p in f.predictions[:5]]))

    def mouseClicked(self, event):
        """
        Mouse clicked in the tSNE similarity plot

        In this case, we want ot update all "9" similar images.
        """
        log.info('')

        # Get the point clicked, in the proper units
        if isinstance(self.sim_repr, pg.PlotWidget):
            #pt_in_plt = self.sim_repr.mapFromDevice(event[0].scenePos().toQPoint())
            pt_in_plt = self.sim_repr.plotItem.mapToView(event[0].scenePos())
            point = (pt_in_plt.x(), pt_in_plt.y())
        else:
            v = self.sim_repr.view
            qpf = v.mapToItem(self.sim_repr.imageItem, event[0].pos())
            point = (qpf.toPoint().x(), qpf.toPoint().y())

        log.debug('point clicked {}'.format(point))
        # Now update based on a click on a point.
        self._parent.similarity_click(point)

    def show_similarity_plot(self):
        """
        Display the scatter plot (for tSNE) based on the filtered indices.
        """
        log.info('')

        if self.plt is not None:
            self.sim_repr.removeItem(self.plt)

        #
        # We are only going to display points in the scatter plot
        # that ar ein the filtered_indices list.
        #

        data = self._parent._similarity_tsne.data_filtered

        if data.shape[0] == 2 or data.shape[1] == 2:
            self.plt = self.sim_repr.scatterPlot(data[:, 0], data[:, 1],
                                                 size=5, symbol='o', pen=pg.mkPen(color=(255, 255, 255)), background=None)
        else:
            self.plt = self.sim_repr.setImage(data)


class FingerprintDock(Dock):
    """
    Region to display the information about the fingerprint when
    hovering over in the image similarity area.
    """

    def __init__(self, *args, **kwargs):
        log.info('')
        super().__init__(*args, **kwargs)

        self.text_widget = QtGui.QLabel('')
        self.text_widget.setMinimumWidth(150)
        self.text_widget.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        #
        # This is the TextItem where we will actually set results.
        #

        self.addWidget(self.text_widget)

    def text(self, thetext):
        log.info('')
        self.text_widget.setText(thetext)


class SimilarityImagesDock(Dock):

    def __init__(self, *args, **kwargs):
        log.info('')

        self.parent = kwargs['similarity_display']
        del kwargs['similarity_display']

        self.fingerprint_dock = kwargs['fingerprint_dock']
        del kwargs['fingerprint_dock']

        super().__init__(*args, **kwargs)

        self.signals = []
        self.images = []
        self.regions = []

    def show_fingerprints(self, fingerprints):
        """
        Display the fingerprints in the grid of images. But
        we want to be sure to display only each unique one.
        And then on each unique one, display the cutouts.
        """
        log.info('')

        # It is possible, due to filtering, that no fingerprints
        # will be passed in here, so nothing to do.
        if len(fingerprints) == 0:
            return

        #
        # Delete any regions that currently exist
        # TODO:  Might be better to not delete ALL the images,
        #        just not the ones needed.
        #

        for x in self.regions:
            del x
        self.regions = []

        for im in self.images:
            self.layout.removeWidget(im)
            im.deleteLater()
        self.images = []

        # Group the fingerprints by the location data (so we
        # display one unique image, not the same one multiple times)

        seen = set()
        unique_data = [f.cutout.data for f in fingerprints if f.cutout.data not in seen and not seen.add(f.cutout.data)]

        grouped_fingerprints = OrderedDict((d, [f for f in fingerprints if f.cutout.data == d]) for d in unique_data)
        grouped_fingerprints_keys = list(grouped_fingerprints.keys())

        #
        # Create new image views.
        #

        # Create the grid of axes for each unique data location
        n_unique = len(unique_data)
        n_sqrt = np.sqrt(n_unique)
        nrows = int(np.ceil(n_sqrt))
        ncols = int(np.ceil(n_unique/nrows))

        ii = 0
        for row in range(nrows):
            for col in range(ncols):
                if ii < n_unique:
                    im1 = ImageDisplay(grouped_fingerprints[grouped_fingerprints_keys[ii]],
                                       title='Image {}'.format(ii+1), parent=self,
                                       fingerprint_dock=self.fingerprint_dock)
                    self.addWidget(im1, row=row, col=col)
                    self.images.append(im1)

                    ii = ii + 1


class ImageDisplay(ColorImageView):
    """
    The ImageDisplay class is essentially an ImageView but also
    has a set of fingerprints (and therefore cutouts [and there for data])
    associated with it.

    This was primarily done to encapsulate the MouseClicked signal for an
    image and to be able to have a callback that understands the underlying
    fingerprint.
    """

    def __init__(self, fingerprints, parent, *args, **kwargs):
        log.info('')

        if 'title' in kwargs:
            self.title = kwargs['title']
            del kwargs['title']
        else:
            self.title = ''

        # This will be the place to display text about the cutout
        # when hovering over it.
        self.fingerprint_dock = kwargs['fingerprint_dock']
        del kwargs['fingerprint_dock']

        super().__init__(*args, **kwargs)
        """
        A container for `numpy.ndarray`-based datasets, using the
        `~astropy.nddata.NDDataBase` interface.

        The key distinction from raw `numpy.ndarray` is the presence of
        additional metadata such as uncertainty, mask, unit, a coordinate system
        and/or a dictionary containing further meta information. This class *only*
        provides a container for *storing* such datasets. For further functionality
        take a look at the ``See also`` section.

        Parameters
        -----------
        data : `numpy.ndarray`-like or `NDData`-like
            The dataset.

        mask : any type, optional
            Mask for the dataset. Masks should follow the ``numpy`` convention that
            **valid** data points are marked by ``False`` and **invalid** ones with
            ``True``.
            Defaults to ``None``.

        copy : `bool`, optional
            Indicates whether to save the arguments as copy. ``True`` copies
            every attribute before saving it while ``False`` tries to save every
            parameter as reference.
            Note however that it is not always possible to save the input as
            reference.
            Default is ``False``.

            .. versionadded:: 1.2

        Raises
        ------
        TypeError
            In case ``data`` or ``meta`` don't meet the restrictions.

        Notes
        -----
        Each attribute can be accessed through the homonymous instance attribute:
        ``data`` in a `NDData` object can be accessed through the `data`
        attribute::

            >>> from astropy.nddata import NDData
            >>> nd = NDData([1,2,3])
            >>> nd.data
            array([1, 2, 3])

        Given a conflicting implicit and an explicit parameter during
        initialization, for example the ``data`` is a `~astropy.units.Quantity` and
        the unit parameter is not ``None``, then the implicit parameter is replaced
        (without conversion) by the explicit one and a warning is issued::

            >>> import numpy as np
            >>> import astropy.units as u
            >>> q = np.array([1,2,3,4]) * u.m
            >>> nd2 = NDData(q, unit=u.cm)
            INFO: overwriting Quantity's current unit with specified unit. [astropy.nddata.nddata]
            >>> nd2.data  # doctest: +FLOAT_CMP
            array([1., 2., 3., 4.])
            >>> nd2.unit
            Unit("cm")

        See also
        --------
        NDDataArray
        """


        self.fingerprints = fingerprints
        self.parent = parent
        self.regions = []

        # Hide things we don't want.
        self.ui.histogram.hide()
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()

        # Set the MouseClicked signal
        self.click_signal = pg.SignalProxy(self.scene.sigMouseClicked, rateLimit=10, slot=self.mouseClicked)
        self.moved_signal = pg.SignalProxy(self.scene.sigMouseMoved, rateLimit=10, slot=self.mouseMoved)

        # Display the cutout data.
        imdata = self.fingerprints[0].cutout.data.get_data()
        self.setImage(imdata)

        # Display the bounding boxes.
        for fingerprint in self.fingerprints:
            self.add_bounding_box(fingerprint.cutout.bounding_box)

    def mouseMoved(self, event):
        """
        Mouse moved in the tSNE similarity plot.

        If this happens then we need to update the similar image
         and the similar image text.
        """
        log.info('{}'.format(event))

        # Correct way of getting image information.
        v = self.view
        qpf = v.mapToItem(self.imageItem, event[0])
        point = (qpf.toPoint().x(), qpf.toPoint().y())

        if any([point[0] < 0, point[1] < 0, point[0] >= self.imageItem.width(), point[1] >= self.imageItem.height()]):
            return

        log.debug('move imagePos {}'.format(point))

        #
        # Given they moved, see if we are in a cutout.
        #
        f_in = None
        for f in self.fingerprints:
            log.debug('Checking to see if point is in cutout {}'.format(f.cutout.bounding_box))
            bb = f.cutout.bounding_box

            if point[0] >= bb[2] and point[0] <= bb[3] and point[1] >= bb[0] and point[1] <= bb[1]:
                log.debug('    ... it is')
                f_in = f
                break

        #
        #  Now that we know they are in a cutout, determine
        #  which element in the similarity matrix we are close to.
        #
        if f_in is not None:
            thetext = '{}\n\n'.format(f_in.cutout.data.location)
            thetext += '\n'.join(['{} {:0.3}'.format(p[1], p[2]) for p in f_in.predictions[:5]])
            thetext += '\n\n{}'.format('\n'.join('{}: {}'.format(k, v) for k, v in f_in.cutout.data.meta.items()))

            self.fingerprint_dock.text(thetext)

    def mouseClicked(self, event):
        """
        Mouse moved in the tSNE similarity plot.

        If this happens then we need to update the similar image
         and the similar image text.
        """
        log.info('event {}'.format(event[0]))

        #
        # Given they clicked, see if we are in a cutout.
        #
        v = self.view
        qpf = v.mapToItem(self.imageItem, event[0].pos())
        image_point = (qpf.x(), qpf.y())

        if any([image_point[0] < 0, image_point[1] < 0, image_point[0] >= self.imageItem.width(), image_point[1] >= self.imageItem.height()]):
            return

        log.debug('click imagePos {}'.format(image_point))

        #
        # Given they clicked, find the closest cutout to the image_point
        # in this particular data.
        #
        cutout = self.parent.parent._similarity_tsne.closest_cutout(self.fingerprints[0].cutout.data, image_point)
        log.debug('Closest cutout is {}'.format(cutout))

        #
        #  Now that we know they are in a cutout, determine
        #  which element in the similarity matrix we are close to.
        #
        tsne_point = self.parent.parent._similarity_tsne.cutout_point(cutout)
        log.debug('tsne_point is {}'.format(tsne_point))
        self.parent.parent.similarity_click(tsne_point)

    def add_bounding_box(self, cutout_bounding_box):
        log.info('')

        bb = cutout_bounding_box
        roi = pg.RectROI([bb[2], bb[0]], [bb[1]-bb[0], bb[3]-bb[2]],
                         pen=pg.mkPen((255, 255, 0, 255), width=2), movable=False)

        # Store the region and add to the ImageView
        self.regions.append(roi)
        self.addItem(roi)


class AitoffDock(Dock):
    """
    Aitoff plot for displaying locations.
    """

    def __init__(self, *args, **kwargs):
        log.info('')
        super().__init__(*args, **kwargs)

        self.points = None
        self._axis_aitoff_text_labels = []

        self.mw = MatplotlibWidget()
        self.mw.toolbar.hide()
        self.subplot = self.mw.getFigure().add_subplot(111, projection="aitoff")
        self.subplot.grid('on')

        #
        #  Add some dummy points
        #
        self.mw.draw()
        self.subplot_background = self.subplot.get_figure().canvas.copy_from_bbox(self.subplot.bbox)

        self.addWidget(self.mw)

    def clear_points(self):
        #
        #  Delete the points
        #
        log.info('')

        if self.points is not None:
            self.points.remove()
            self.points = None
            self.mw.draw()

        #
        #  Delete the text labels
        #

        for t in self._axis_aitoff_text_labels:
            t.remove()
        self._axis_aitoff_text_labels = []

    def add_points(self, points, color='b'):
        log.info('')

        self.clear_points()

        converted_points = [self.convert_ra_dec(*radec) for radec in points]

        #
        # Display the points on the Aitoff projection
        #

        xs = np.array([x[0] for x in converted_points])
        ys = np.array([x[1] for x in converted_points])
        self.points = self.subplot.scatter(xs, ys, c='b')

        #
        # Now label the numbers on the plot to correspond to the
        # similar images.
        #

        converted_points = np.array(converted_points)
        d = distance_matrix(converted_points, converted_points)

        rows = set(range(converted_points.shape[0]))
        groups = {}

        while len(rows) > 0:
            row = rows.pop()
            close = np.nonzero(d[row] < 0.01)[0]
            rows = rows - set(list(close))
            groups[row] = close

        for k, v in groups.items():
            tt = self.subplot.text(converted_points[k][0]+0.05, converted_points[k][1]+0.05,
                                   ','.join([str(x+1) for x in v]))
            self._axis_aitoff_text_labels.append(tt)
            self.subplot.draw_artist(tt)

        self.mw.draw()

    def convert_ra_dec(self, ra, dec):
        log.info('')

        if ra is not None and dec is not None:
            coords = SkyCoord(ra=ra, dec=dec, unit='degree')
            ra = coords.ra.wrap_at(180 * u.deg).radian
            dec = coords.dec.radian
        else:
            ra, dec = None, None
        return ra, dec


class SimilarityDisplay(QtGui.QApplication):
    """
    Main window for all the differnet pieces.
    """

    def __init__(self, similarity_tsne):
        log.info('')

        # http://www.pyqtgraph.org/documentation/widgets/imageview.html
        # Need to change to row major order.
        pg.setConfigOptions(imageAxisOrder='row-major')

        self._similarity_tsne = similarity_tsne

        self.app = QtGui.QApplication([])
        self.win = QtGui.QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1500, 800)
        self.win.setWindowTitle('Hubble Transfer Learning')

        # Create docks, place them into the window one at a time.
        # Note that size arguments are only a suggestion; docks will still have to
        # fill the entire dock area and obey the limits of their internal widgets.

        self.fingerprint = FingerprintDock("Fingerprint Info", size=(80, 500))
        self.aitoff = AitoffDock("Aitoff", size=(350, 200))
        self.similarity = SimilarityPlotDock("Similarity Plot", size=(250, 200), parent=self)
        self.similar_images = SimilarityImagesDock("Similar Images", size=(500, 500), similarity_display=self, fingerprint_dock=self.fingerprint)

        self.area.addDock(self.aitoff, 'top')
        self.area.addDock(self.similarity, 'bottom')
        self.area.addDock(self.similar_images, 'right')
        self.area.addDock(self.fingerprint, 'right')

        self.win.show()

        self.similar_images.show_fingerprints(self._similarity_tsne._fingerprints[:9])

        self.instance().exec_()

    def similarity_click(self, point):
        """
        When a point is clicked in the similarity plot
        """
        log.info('point {}'.format(point))

        #
        #  Find the N similar images to this point.
        #

        overlapping_bounding_boxes = self.similarity.fingerprint_filter_widget.overlapping_bb
        fingerprints = self._similarity_tsne.find_similar(point, n=9,
                                                          allow_overlapping_bounding_boxes=overlapping_bounding_boxes)

        # It is possilbe to have nothing come back if we are filtering
        # too much.
        if len(fingerprints) == 0:
            return

        #
        # Display the fingeprint cutouts on the images
        #

        self.similar_images.show_fingerprints([f['fingerprint'] for f in fingerprints])

        #
        # Display on the Aitoff plot.
        #

        # Get the RA/DEC
        radec = [f['fingerprint'].cutout.data.radec for f in fingerprints]
        self.aitoff.add_points(radec, 'b')


# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):

        with open('hubble_similarity_tsne.pck', 'rb') as fp:
            stsne = pickle.load(fp)
            similarity_tsne = Similarity.factory(stsne)

            sd = SimilarityDisplay(similarity_tsne)
            sd.instance().exec_()
