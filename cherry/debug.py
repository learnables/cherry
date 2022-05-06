#!/usr/bin/env python3

"""
General debugging utilities.
"""

import os
import sys
import logging
import traceback
import pdb
import queue

from logging import handlers
from datetime import datetime

IS_DEBUGGING = False


# Sets up general debugger
logger = logging.getLogger('cherry')
logger.setLevel(logging.INFO)
logger.propagate = False

# Handler for normal printing
fmt = logging.Formatter(fmt='%(message)s', datefmt='')
print_handler = logging.StreamHandler(sys.stdout)
print_handler.setFormatter(fmt)
print_handler.setLevel(logging.INFO)
logger.addHandler(print_handler)


def debug(log_dir='./'):
    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/debug.py" class="source-link">[Source]</a>

    ## Description

    Enables some debugging utilities for logging and pdb.

    Includes:

    * Automatically dropping into a post-mortem pdb debugger session
    whenever an exception is raised.
    * Enables fast DEBUG logging to a logging file via QueueHandler.
    * Copies all stdout output to the logging file. (Experimental)

    ## References

    1. Automatically start the debugger on an exception (Python recipe), Thomas Heller, 2001,
        [Link](http://code.activestate.com/recipes/65287-automatically-start-the-debugger-on-an-exception/)
    2. Dealing with handlers that block, Python Documentation, 2019.
        [Link](https://docs.python.org/3/howto/logging-cookbook.html#dealing-with-handlers-that-block)

    ## Arguments

    * `log_dir` (str, *optional*, Default: './') - Location to store the log files.

    ## Example

    ~~~python
    ch.debug.debug()
    raise Exception('My exception')
    -> raise('My exception')
    (Pdb)
    ~~~

    """
    global IS_DEBUGGING
    if not IS_DEBUGGING:
        # Enable debugging logging.
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = os.path.join(log_dir, 'cherry_debug_' + now + '.log')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # Experimental: forward stdout/print to log_file too
        log_file = open(log_file, mode='a', buffering=1, encoding='utf-8')
        stdout_write = sys.stdout.write
        stderr_write = sys.stderr.write

        def custom_stdout_write(*args, **kwargs):
            stdout_write(*args, **kwargs)
            log_file.write(*args, **kwargs)

        def custom_stderr_write(*args, **kwargs):
            stderr_write(*args, **kwargs)
            log_file.write(*args, **kwargs)

        def custom_newline_stdout(*args, **kwargs):
            custom_stdout_write(*args, **kwargs)
            custom_stdout_write('\n')

        global print
        print = custom_newline_stdout
        sys.stdout.write = custom_stdout_write
        sys.stderr.write = custom_stderr_write

        # Log to file using queue handler and listener
        logger.setLevel(logging.DEBUG)
        debug_queue = queue.Queue(-1)
        queue_handler = handlers.QueueHandler(debug_queue)
        logger.addHandler(queue_handler)
        debug_fmt = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s \n%(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        debug_handler = logging.StreamHandler(log_file)
        debug_handler.setFormatter(debug_fmt)
        debug_handler.setLevel(logging.DEBUG)
        queue_listener = handlers.QueueListener(debug_queue, debug_handler)
        queue_listener.start()
        logger.debug('Debugging started.')

        # Enable automatic post-mortem on Exception.
        def info(type, value, tb):
            if hasattr(sys, 'ps1') or not sys.stderr.isatty():
                sys.__excepthook__(type, value, tb)
            else:
                traceback.print_exception(type, value, tb)
                pdb.pm()
        sys.excepthook = info

        # Turn debug flag on.
        IS_DEBUGGING = True


if __name__ == '__main__':
    print('This is from print.')
    print('This is from print.')
    sys.stdout.write('This is from stdout.')
    logger.debug('debug')
    logger.info('info')
    debug(log_dir='./logs')
    debug()
    logger.info('info')
    logger.debug('debug')
    print('This is from print.')
    print('This is from print.')
    sys.stdout.write('This is from stdout.')
    raise Exception('haha')
