import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import distance_matrix

from tldist.database import get_database
from tldist.similarity.similarity import Similarity
from tldist.data import Data
from tldist.fingerprint import Fingerprint

from astropy.coordinates import SkyCoord
from astropy import units

import sys
import logging
logging.basicConfig(format='%(levelname)-6s: %(asctime)-15s %(name)-10s %(funcName)-10s %(message)s')
log = logging.getLogger("TransferLearningDisplay")
# Can be used for debugging Jupyter Notebooks
fhandler = logging.FileHandler(filename='/tmp/mylog.log', mode='a')
log.addHandler(fhandler)
log.setLevel(logging.WARNING)

class SimilarityDisplay(object):
    
    def __init__(self, similarity, db):
        
        plt.ion()

        self._similarity = similarity
        self._db = db
        
        # Setup the figure
        self._figure = plt.figure(2, figsize=[10,6])
        
        # The NxM set of similar images
        self._similar_images_axis = SimilarImages([0.5, 0.5, 0.4, 0.4], [3, 3])

        # The current image displayed as one moves around the
        # similarity plot (e.g., tSNE)
        self._current_image_axis = Image([0.5, 0.1, 0.2, 0.2])
        self._current_image_axis.imshow(np.zeros((224,224)))
        self._current_image_axis_fingerprint_uuid = '' # caching
        self._current_image_axis_time_update = 0
        self._move_processing_callback = False
        
        # The AITOFF plot
        self._aitoff_axis = Aitoff([0.1, 0.55, 0.4, 0.4])
        
        # Display the similarity matrix based on display method in 
        # the similarity subclass (e.g,. tSNE, Jaccard etc)
        self._similarity_matrix = Image([0.1, 0.1, 0.4, 0.4])
        self._similarity.display(self._similarity_matrix)
        
        # Connect the callbacks
        self._figure.canvas.mpl_connect('button_press_event', self._click_callback)
        self._figure.canvas.mpl_connect('motion_notify_event', self._move_callback)
    
        # Initialize the 9 similar images with the first 9 fingerprints
        # TODO: Fix this so it is the first and the actual similar ones
        fingerprints = db.find('fingerprint')
        for ii in range(9):
            f = Fingerprint.fingerprint_factory(fingerprints[ii])
            data_uuid = f.data_uuid
            data = db.find('data', data_uuid)
            d = Data.data_factory(data)
            self._similar_images_axis.set_image(ii, d.get_data(), os.path.basename(d.location), fingerprints[ii])

        plt.show()
    
    def _move_callback(self, event):
        """
        Move callback:  This will essentially just update the hover image 
        and the point 
        
        """
        log.debug('---------------------------------------------')

        if event.inaxes == self._similarity_matrix._axes:
            point = event.ydata, event.xdata
            close_fingerprint_uuids = self._similarity.find_similar(point, 1)
            
            now = time.time()
            
            fingerprint_uuid = close_fingerprint_uuids[0]['fingerprint_uuid']
            if not fingerprint_uuid == self._current_image_axis_fingerprint_uuid and (now - self._current_image_axis_time_update) > 0.5:
                self._current_image_axis_fingerprint_uuid = fingerprint_uuid
                
                fingerprint = db.find('fingerprint', fingerprint_uuid)
                
                # Get the data location
                data = db.find('data', fingerprint['data_uuid'])
                d = Data.data_factory(data)
            
                # Display the image
                self._current_image_axis.imshow(d.get_data())
                
                # Display the location on the Aitoff plot
                self._aitoff_axis.on_move(d.radec)
                
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
                for fp in close_fingerprints:
                    fingerprint = db.find('fingerprint', fp['fingerprint_uuid'])
                    data = db.find('data', fingerprint['data_uuid'])
                    a = (fp['tsne_point'], fp['distance'], data['radec'], fingerprint)
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
            fingerprint_uuid = event.inaxes._imdata['uuid']

            index = self._similarity.fingerprint_uuids.index(fingerprint_uuid)

            point = similarity_tsne._Y[index]
            close_fingerprint_uuids = self._similarity.find_similar(point)
            self._similar_images_axis.set_images(close_fingerprint_uuids)

class SimilarImages(object):
    def __init__(self, axes_limits, rows_cols):
        self._axes_limits = axes_limits
        self._rows_cols = rows_cols
        
        self.axes = []
        ii = 0
        for ri in range(self._rows_cols[0]):
            for ci in range(self._rows_cols[1]):
                row_size = axes_limits[2] / rows_cols[0]
                col_size = axes_limits[3] / rows_cols[1]
                sub_limits = [axes_limits[0]+(row_size + 0.02)*ri, 
                              axes_limits[1]+(col_size + 0.02)*ci, 
                              row_size, 
                              col_size]
                
                image = Image(sub_limits)
                image.store({'type': 'similar', 'number': ii})
                self.axes.append(image)
                ii = ii + 1

                
    def set_images(self, fingerprints):
        for ii, fp in enumerate(fingerprints):
            # Get the fingerprint
            f_uuid = fp['fingerprint_uuid']
            fingerprint = db.find('fingerprint', f_uuid)
                
            # Get the data location
            data = db.find('data', fingerprint['data_uuid'])
            d = Data.data_factory(data)
            
            self.set_image(ii, d.get_data(), fingerprint=fingerprint)
                
    def set_image(self, num, data, title='', fingerprint={}):
        # Display image
        self.axes[num].imshow(data, title=title)
        
        self.axes[num].store(fingerprint)
        # Show on Aitoff

        
class Image(object):
    def __init__(self, axes_limits):
        self._axes = plt.axes(axes_limits)
                          
    def store(self, thedict):
        self._axes._imdata = thedict
                  
    def get(self):
        return self._axes._imdata

    def imshow(self, data, title=''):
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._axes.set_xlabel('')
        self._axes.set_ylabel('')
        self._axes.imshow(data, cmap=plt.gray())
    
        if title:
            self._axes.set_title(title, fontsize=8)
    
        self._axes.get_figure().canvas.blit(self._axes.bbox)
        self._axes.redraw_in_frame()

    def plot(self, x, y, title=''):
        self._axes.plot(x, y, '.')
        if title:
            self._axes.set_title(title, fontsize=8)
    
    def grid(self, onoff='on'):
        self._axes.grid(onoff)
        
    def set_title(self, title):
        self._axes.set_title(title)
    
class Aitoff(object):

    def __init__(self, axes_limits):
        self._axes_limits = axes_limits

        # If we have ra_dec then let's display the Aitoff projection axis
        self._axis = plt.axes(self._axes_limits, projection="aitoff")
        self._axis.grid('on')
        self._axis.set_xlabel('RA')
        self._axis.set_ylabel('DEC')
        self._onmove_point = None
        self._onclick_points = {}
        self._axis_background = self._axis.get_figure().canvas.copy_from_bbox(self._axis.bbox)
        self._axis_text_labels = []

        self._axis_aitoff_text_labels = []
        
        self._onmove_point = None
        self._onmove_color = (0.1, 0.6, 0.1)    
 
    def convert_ra_dec(self, ra, dec):
        if ra is not None and dec is not None:
            coords = SkyCoord(ra=ra, dec=dec, unit='degree')
            ra = coords.ra.wrap_at(180 * units.deg).radian
            dec = coords.dec.radian
        else:
            ra,dec = None, None
        return ra, dec

    def on_click(self, close_fingerprints):
        
        if not isinstance(close_fingerprints, list) or not isinstance(close_fingerprints[0], tuple):
            log.error('Wrong thing passed to Aitoff on_click.  It must be a list of tuples where each tuple contains piont, distnace and fingerprint')
            raise Exception('Wrong thing passed to Aitoff')
            
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
            points.append([ra,dec])

        self._axis.get_figure().canvas.restore_region(self.axis_background)

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
            tt = self._axis.text(points[k][0]+0.05, points[k][1]+0.05,
                    ','.join([str(x+1) for x in v]))
            self._axis_aitoff_text_labels.append(tt)
            self._axis.draw_artist(tt)

        self._axis.get_figure().canvas.blit(self._axis.bbox)

        self.axis_background = self._axis.get_figure().canvas.copy_from_bbox(self._axis.bbox) 

    def on_move(self, radec):
        try:
            ra, dec = self.convert_ra_dec(radec[0], radec[1])

            # Update the mbitoff figure as well
            self._axis.get_figure().canvas.restore_region(self.axis_background)

            if self._onmove_point is not None:
                self._onmove_point[0].set_data(ra, dec)
            else:
                self._onmove_point = self._axis.plot(ra, dec,
                        'o', color=self._onmove_color)
                
            self._axis.get_figure().canvas.blit(self._axis.bbox)    
        except Exception as e:
            log.error('display point {}'.format(e))
            
db = get_database('blitzdb', '/tmp/basic_notebook.db')
similarities = db.find('similarity')
similarity_tsne = Similarity.similarity_factory(similarities[0])

simdisp = SimilarityDisplay(similarity_tsne, db)

