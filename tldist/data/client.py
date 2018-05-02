import logging

from PIL import Image
from io import BytesIO
import requests
from imageio import imread
from configparser import ConfigParser

from tlapi.config import api_server

import numpy as np

FORMAT = '%(levelname)-8s %(asctime)-15s %(name)-10s %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger('data')
log.setLevel(logging.DEBUG)

config = ConfigParser()
config.read('config.ini')
api_server = 'http://' + config['processor']['machine']

def get(pk=None):
    """
    Programmatic access into the end point
    """

    # Get the location info
    if pk is None:
        response = requests.get('{}/data'.format(api_server))
    else:
        log.debug('{}/data/{}'.format(api_server, pk))
        response = requests.get('{}/data/{}'.format(api_server, pk))

    return response.json()


def save(data):
    """
    Programmatic access into the end point
    """

    log.debug('{}/data/save'.format(api_server))
    response = requests.post('{}/data/save'.format(api_server), json=data)
    return response.json()


def data_get_array(pk):
    """
    Get the image from that end point
    """

    # Get the location info
    image_info = get(pk)

    # Get the data.
    location = image_info['location']
    response = requests.get(location)

    # Convert to numpy array.
    nparray = imread(BytesIO(response.content))

    return nparray
