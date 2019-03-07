import logging


def get_logger(name, logfile=None, level=logging.WARNING):
    FORMAT = '%(levelname)-8s %(asctime)-15s %(name)-10s %(funcName)-10s %(lineno)-4d %(message)s'
    logging.basicConfig(format=FORMAT)
    log = logging.getLogger(name)
    if logfile is not None:
        fhandler = logging.FileHandler(filename=logfile, mode='a')
        log.addHandler(fhandler)
    log.setLevel(level)

    return log
