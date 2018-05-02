import imageio
from io import BytesIO
import requests
import logging
import os.path

FORMAT = '%(levelname)-8s %(asctime)-15s %(name)-10s %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger('data')
log.setLevel(logging.DEBUG)

def get(db, pk=None):
    log.info('get with pk {}'.format(pk))

    if pk is None:
        # Get all the data
        data = db.find(table='data')
    else:
        # Get the data based on the pk
        data = db.find(table='data', key=pk)
    return data


def get_array(db, pk=None):
    log.info('get_array with pk {}'.format(pk))

    # Get the data based on the pk
    image_info = db.find(table='data', key=pk)
    location = image_info['location']

    if 'http' in location:
        response = requests.get(location)
        # Convert to numpy array.
        nparray = imageio.imread(BytesIO(response.content))
    elif os.path.isfile(location):
        nparray = imageio.imread(location)
    else:
        log.error('   Could not locate data {}'.format(location))
        return None

    return nparray

def save(db, payload):
    log.info('saving data')

    resp = db.save(table='data', data=payload)
    return resp

