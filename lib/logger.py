'''
This module contains a little logger class that allows
printing to the screen and to a file. To use:

import logger
import sys
sys.stdout = logger.Logger('log.log')
'''

import sys

class Logger():
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log      = open(logfile, 'a')

    def write(self, message):
        assert not self.log.closed, "Attempted to write to closed log."\
                 " Here's a neat traceback to help you find the error."

        self.terminal.write(message)
        self.log.     write(message)

    def close(self):
        self.log.close()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
