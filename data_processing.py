import numpy as np
import scipy.ndimage.filters
import scipy.ndimage

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("DataProcessing")
log.setLevel(logging.INFO)


class DataProcessing:
    def __init__(self, *args, **kwargs):
        pass

    def _apply2dfunc(self, input_data, func, *args, **kwargs):
        if len(input_data.shape) == 2:
            return func(input_data, *args, **kwargs)
        else:
            output_data = np.zeros(input_data.shape)
            for ii in range(input_data.shape[2]):
                output_data[:, :, ii] = func(input_data[:, :, ii], *args, **kwargs)
            return output_data

    def save(self):
        raise NotImplementedError("Please Implement this method")

    def process(self):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def load_parameters(parameters):

        for class_ in DataProcessing.__subclasses__():
            if class_.__name__ == parameters['data_processing']:

                # If the class name matches, instantiate, pass in the parameters
                # and return the newly created class.
                tt = class_()
                tt.load(parameters)
                return tt


class CropData(DataProcessing):
    def __init__(self, output_size=224, *args, **kwargs):
        super(CropData, self).__init__(*args, **kwargs)
        self._output_size = output_size

    def __repr__(self):
        return 'CropData (output_size {})'.format(self._output_size)

    def process(self, input_data):
        nrows, ncols = input_data.shape[:2]
        row_center, col_center = nrows//2, ncols//2

        return input_data[row_center-112:row_center+112, col_center-112:col_center+112]

    def save(self):
        return {'data_processing': self.__class__.__name__, 'parameters': {'output_size': self._output_size}}

    def load(self, parameters):
        self._output_size = parameters.get('parameters').get('output_size')


class ZoomData(DataProcessing):
    def __init__(self, zoom_level=1, *args, **kwargs):
        super(ZoomData, self).__init__(*args, **kwargs)
        self._zoom_level = zoom_level

    def __repr__(self):
        return 'ZoomData (level {})'.format(self._zoom_level)

    def process(self, input_data):
        output_data = None

        for ii in range(input_data.shape[2]):
            out = scipy.ndimage.zoom(input_data[:,:,ii], self._zoom_level)
            if output_data is None:
                output_data = np.zeros((out.shape[0], out.shape[1], input_data.shape[2]))

            output_data[:,:,ii] = out

        return output_data

    def save(self):
        return {'data_processing': self.__class__.__name__, 'parameters': {'zoom': self._zoom_level}}

    def load(self, parameters):
        self._zoom_level = parameters.get('parameters').get('zoom')


class MedianFilterData(DataProcessing):
    def __init__(self, kernel_size=(1,1,1)):
        super(MedianFilterData, self).__init__()

        self._kernel_size = kernel_size

    def __repr__(self):
        return 'MedianFilterData (kernel size {})'.format(self._kernel_size)

    def process(self, input_data):
        return scipy.ndimage.filters.median_filter(input_data, size=self._kernel_size)

    def save(self):
        return {'data_processing': self.__class__.__name__, 'parameters': {'kernel_size': self._kernel_size}}

    def load(self, parameters):
        self._kernel_size = parameters.get('parameters').get('kernel_size')


class RotateData(DataProcessing):
    def __init__(self, angle=0.0):
        self._angle = angle

    def __repr__(self):
        return 'RotateData (angle {})'.format(self._angle)

    def process(self, input_data):
        return self._apply2dfunc(input_data, scipy.ndimage.rotate, self._angle, reshape=False)

    def save(self):
        return {'data_processing': self.__class__.__name__, 'parameters': {'angle': self._angle}}

    def load(self, parameters):
        self._angle = parameters.get('parameters').get('angle')


class GrayScaleData(DataProcessing):
    def __init__(self):
        pass

    def __repr__(self):
        return 'GrayScaleData'

    def process(self, input_data):
        return np.dot(input_data[:,:,:3], [0.299, 0.587, 0.114])

    def save(self):
        return {'data_processing': self.__class__.__name__, 'parameters': {}}

    def load(self, parameters):
        pass


class FlipUDData(DataProcessing):
    def __init__(self):
        pass

    def __repr__(self):
        return 'FlipUDData'

    def process(self, input_data):
        return input_data[::-1]

    def save(self):
        return {'data_processing': self.__class__.__name__, 'parameters': {}}

    def load(self, parameters):
        pass
