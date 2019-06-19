import logging


def get_logger(name, logfile=None, level=logging.WARNING):
    """
    Wrapper to create a logger.

    :param name: name of logger
    :param logfile: file to log to (or None)
    :param level: log level based on Python logging levels.
    :return:
    """
    FORMAT = '%(levelname)-8s %(asctime)-15s %(name)-10s %(funcName)-10s %(lineno)-4d %(message)s'
    logging.basicConfig(format=FORMAT)
    log = logging.getLogger(name)
    if logfile is not None:
        fhandler = logging.FileHandler(filename=logfile, mode='a')
        log.addHandler(fhandler)
    log.setLevel(level)

    return log
