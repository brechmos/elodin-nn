import sys
import logging
import time
import uuid
from collections import OrderedDict
import itertools

from celery import group
import numpy as np
import json
import requests
import imageio
from io import BytesIO

from tlapi.data.api import get as get_data
from tlapi.data.api import get_array as get_array_data
from tldist.celery import app

from tldist.fingerprint.processing import Fingerprint

FORMAT = '%(levelname)-8s %(asctime)-15s %(name)-10s %(funcName)-10s %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger('fingerprint')
log.setLevel(logging.DEBUG)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def calculate_celery(data, fc_save):
    """
    This function will queue up all the jobs and run them using celery.
    """

    # Queue up and run
    job = group([
        calculate.s(tt, fc_save) for tt in chunks(data, len(data)//4)
    ])
    result = job.apply_async()

    # Dispaly progress
    counts = OrderedDict({x.id: 0 for x in result.children})
    while not result.ready():
        time.sleep(0.1)
        for x in result.children:
            if x.state == 'PROGRESS' and hasattr(x, 'info') and 'progress' in x.info:
                counts[x.id] = x.info['progress']

        print('Calculating fingerprints \r{}'.format([v for k,v in counts.items()]), end='')

    # Get the results (is a list of lists so need to compress them)
    r =  result.get()

    return list(itertools.chain(*r))

@app.task
def calculate(data, fc_save):
    """
    Calculate the fingerprint from a list of data.  The data
    must be of the form 
         [ {'uuid': <somtehing>, 'location': <somewhere>, 'meta': {<meta data} }... ]
    """

    if not isinstance(data, list) and not isinstance(data[0], dict):
        log.error('Data must be a list of dictionaries')
        raise Exception('Data must be a list of dictionaries')

    # Load the fingerprint calculator based on dictionary information
    fc = Fingerprint.load_parameters(fc_save)

    # Now run through each datum and calculate the fingerprint
    fingerprints_return = []
    for ii, datum in enumerate(data):

        # Update the progress if we are using the task version of this.
        if hasattr(calculate, 'update_state'):
            calculate.update_state(state='PROGRESS', meta={'progress': ii})

        # Load the data
        if 'location' not in datum:
            log.error('Data does not have a location key {}'.format(datum))
            raise Exception('Data does not have a location key {}'.format(datum))

        response = requests.get(datum['location'])

        if not response.status_code == 200:
            log.error('Problem loading the data {}'.format(datum['location']))
            raise Exception('Problem loading the data {}'.format(datum['location']))

        nparray = np.array(imageio.imread(BytesIO(response.content)))

        # Calculate the predictions
        log.debug('calcuating predictions for  {} data is {}'.format(datum['location'], type(nparray)))
        try:
            predictions = fc.calculate(nparray[:224,:224])
        except:
            predictions = []

        # Clean the predictions so the json conversion is happy
        cleaned_predictions = [(x[0], x[1], float(x[2])) for x in predictions]

        # Load up the return list.
        fingerprints_return.append({
            'uuid': str(uuid.uuid4()), 
            'data_uuid': datum['uuid'], 
            'predictions': cleaned_predictions
            })

    return fingerprints_return
