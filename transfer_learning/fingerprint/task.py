import time
from collections import OrderedDict
import itertools

from celery import group

from transfer_learning.celery import app
from transfer_learning.fingerprint.processing import calculate as processing_calculate

from ..tl_logging import get_logger
import logging
log = get_logger('fingerprint task')


def chunks(l, k):
    n = len(l)
    return [l[i * (n // k) + min(i, n % k):(i+1) * (n // k) + min(i+1, n % k)]   for i in range(k)]


def calculate_celery(cutouts, fc_save):
    """
    This function will queue up all the jobs and run them using celery.
    """

    # Queue up and run
    job = group([
                    calculate_task.s(tt, fc_save)
                    for tt in chunks(cutouts, 3)
                ])
    result = job.apply_async()

    # Dispaly progress
    counts = OrderedDict({x.id: 0 for x in result.children})
    while not result.ready():
        time.sleep(0.1)
        for x in result.children:
            if (x.state == 'PROGRESS' and hasattr(x, 'info') and
                    'progress' in x.info):
                counts[x.id] = x.info['progress']

        states_complete = [int(v) for k, v in counts.items()]
        log.info('\rCalculating fingerprints: {} {:.1f}%'.format(
            states_complete, sum(states_complete)/len(cutouts)*100), end='')

    # Get the results (is a list of lists so need to compress them)
    r = result.get()
    return list(itertools.chain(*r))


@app.task
def calculate_task(cutouts, fc_save):
    log.debug('app.current_task {}'.format(app.current_task))
    return processing_calculate(cutouts, fc_save, task=app.current_task)
