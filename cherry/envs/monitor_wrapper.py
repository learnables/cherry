#!/usr/bin/env python3

import os
import time

from datetime import datetime

from gym.wrappers import Monitor as GymMonitor

from .base import Wrapper


class Monitor(Wrapper, GymMonitor):

    """
    Sugar coating on top of Gym's Monitor.
    """

    def __init__(self, env, directory, *args, **kwargs):
        Wrapper.__init__(self, env)
        env_name = env.spec.id
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')
        self.directory = os.path.join(directory, env_name)
        self.directory = os.path.join(self.directory, date)
        self.env = env
        GymMonitor.__init__(self, env, self.directory, *args, **kwargs)
