from configparser import ConfigParser

from celery import Celery

# Load the configuration
c = ConfigParser()
c.read('config.ini')


app = Celery('elodin_nn',
             broker=c['processor']['celery_broker'],
             backend=c['processor']['celery_backend'],
             include=[
                 'elodin_nn.fingerprint.task',
                 'elodin_nn.similarity.task',
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
