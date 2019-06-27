#!/usr/bin/env python
# -*- coding: utf8 -*-
from gym import logger


class beyond_done(object):

    "You are calling 'step()' even though this environment has already returned done = True.  You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior."

    steps = None
    data = None
    
    def reset(self):
        self.steps = 0
        self.data = None
        
    def step(self, done):
        self.steps += 1
        if done:
            if self.data is None:
                self.data = 0
            else:
                if self.data == 0:
                    logger.warn(self.__doc__)
                self.data += 1

    def is_done(self):
        return self.data is not None


if __name__ == "__main__":
    pass
