import numpy as np
import scipy.ndimage.filters
import scipy.ndimage

import logging
logging.basicConfig(format='%(levelname)-6s: %(name)-10s %(asctime)-15s  %(message)s')
log = logging.getLogger("DataProcessing")
log.setLevel(logging.INFO)


class DataProcessingClass(object):
    frametype_class_dict = {}

    def __init__(self, *args, **kwargs):
        self.frame_id = str(args[0].__name__)
        DataProcessingClass.frametype_class_dict[self.frame_id] = args[0]

    def __call__(self, cls):
        if self.frame_id:
            DataProcessingClass.frametype_class_dict[self.frame_id] = cls
        return cls

    @staticmethod
    def get_class_from_frame_identifier(frame_identifier):
        return DataProcessingClass.frametype_class_dict.get(frame_identifier)


class DataProcessing:
    def __init__(self):
        pass

    def apply2dfunc(self, input_data, func, *args, **kwargs):
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
    def load(parameters):
        class_name = parameters.get('data_processing')
        cls = DataProcessingClass.get_class_from_frame_identifier(class_name)
        inst = cls()
        inst.load(parameters)
        return inst

@DataProcessingClass
class ZoomData(DataProcessing):
    def __init__(self, zoom_level=1):
        self._zoom_level = zoom_level

    def __repr__(self):
        return 'ZoomData (level {})'.format(self._zoom_level)

    def process(self, input_data):
        return self._apply2dfunc(input_data, scipy.ndimage.zoom, self._zoom_level)

    def save(self):
        return {'data_processing': self.__class__.__name__, 'parameters': {'zoom': self._zoom_level}}

    def load(self, parameters):
        self._zoom_level = parameters.get('parameters').get('zoom')

@DataProcessingClass
class MedianFilterData(DataProcessing):
    def __init__(self, kernel_size=(1,1,1)):
        self._kernel_size = kernel_size

    def __repr__(self):
        return 'MedianFilterData (kernel size {})'.format(self._kernel_size)

    def process(self, input_data):
        return scipy.ndimage.filters.median_filter(input_data, size=self._kernel_size)

    def save(self):
        return {'data_processing': self.__class__.__name__, 'parameters': {'kernel_size': self._kernel_size}}

    def load(self, parameters):
        self._kernel_size = parameters.get('parameters').get('kernel_size')


@DataProcessingClass
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


@DataProcessingClass
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


@DataProcessingClass
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
