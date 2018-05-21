from collections import OrderedDict

from transfer_learning.celery import app
from celery import group
import time

from .similarity import calculate as similarity_calculate
from .similarity import tSNE, Jaccard, Distance

from ..tl_logging import get_logger
import logging
log = get_logger('similarity task')


def similarity_celery(fingerprints, sim):
    """
    Similarity calculator using celery for job queue/running.
    """
    log.info('')

    # Create and run the job queue
    job = group([
        calculate.s(fingerprints, sim)
    ])
    celery_result = job.apply_async(serializer='pickle')

    # Show the progress (not really needed)
    counts = OrderedDict({x.id: 0 for x in celery_result.children})
    while not celery_result.ready():

        # We only need to display every 100ms or so, maybe less really
        time.sleep(0.1)
        for x in celery_result.children:
            if x.state == 'PROGRESS' and hasattr(x, 'info') and 'progress' in x.info:
                counts[x.id] = x.info['progress']

        print('\r{}'.format([v for k, v in counts.items()]), end='')

    # Get the results (will be a list of lists)
    r = celery_result.get()

    # In this case, there is a list returned with one dict element. If this
    # changes in the future then we'll have to modify this to be something different.
    return r[0]


@app.task
def calculate(fingerprints, similarity_calculator):
    """
    Similarity calculator.
    """
    log.debug('In the app.task calculate with similarity_calculator = {}'.format(similarity_calculator))
    return similarity_calculate(fingerprints, similarity_calculator, serialize_output=True)
