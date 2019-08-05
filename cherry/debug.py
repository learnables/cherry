#!/usr/bin/env python3

"""
General debugging utilities.
"""

import sys
import logging
import traceback
import pdb
from datetime import datetime

IS_DEBUGGING = False


# Sets up general debugger
logger = logging.getLogger('cherry')
logger.setLevel(logging.INFO)

# Handler for normal printing
fmt = logging.Formatter(fmt='%(message)s', datefmt='')
print_handler = logging.StreamHandler(sys.stdout)
print_handler.setFormatter(fmt)
print_handler.setLevel(logging.INFO)
logger.addHandler(print_handler)


def debug():
    """
    Enables some debugging utilities for logging and pdb.

    Includes:

    * Automatically dropping into a post-mortem pdb debugger session
    whenever an exception is raised.
    * Enables DEBUG logging to a cherry logging file of the main logger.

    **Reference**

    1. http://code.activestate.com/recipes/65287-automatically-start-the-debugger-on-an-exception/

    **Example**

    ch.debug.debug()
    raise Exception('My exception')
    -> raise('My exception')
    (Pdb)
    """
    global IS_DEBUGGING
    if not IS_DEBUGGING:
        # Enable debugging logging.
        logger.setLevel(logging.DEBUG)
        debug_fmt = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s \n%(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        debug_handler = logging.FileHandler('cherry_debug_' + now + '.log')
        debug_handler.setFormatter(debug_fmt)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

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
    logger.debug('debug')
    logger.info('info')
    debug()
    debug()
    debug()
    logger.debug('debug')
    logger.info('info')
    raise Exception('haha')
