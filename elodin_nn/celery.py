from __future__ import absolute_import, unicode_literals
from celery import Celery
from configparser import ConfigParser

# Load the configuration
c = ConfigParser()
c.read('config.ini')


app = Celery('transfer_learning',
             broker=c['processor']['celery_broker'],
             backend=c['processor']['celery_backend'],
             include=[
                 'transfer_learning.fingerprint.task',
                 'transfer_learning.similarity.task',
                 ]
            )

# Optional configuration, see the application user guide.
app.conf.update(
    CELERY_TASK_RESULT_EXPIRES = 3600,
    CELERY_TASK_SERIALIZER = 'pickle',
    CELERY_RESULT_SERIALIZER = 'pickle',
    CELERY_ACCEPT_CONTENT = ['pickle', 'json']
)

if __name__ == '__main__':
    app.start()
