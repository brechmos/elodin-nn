import logging

class MyFormatter(logging.Formatter):
    width = 24
    datefmt='%Y-%m-%d %H:%M:%S'

    def format(self, record):
        cpath = '%s:%s:%s' % (record.module, record.funcName, record.lineno)
        cpath = cpath[-self.width:].ljust(self.width)
        record.message = record.getMessage()
        s = "%-7s %s %s : %s" % (record.levelname, self.formatTime(record, self.datefmt), cpath, record.getMessage())
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        #if record.stack_info:
        #    if s[-1:] != "\n":
        #        s = s + "\n"
        #    s = s + self.formatStack(record.stack_info)
        return s
