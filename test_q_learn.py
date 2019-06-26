#!/usr/bin/env python
# -*- coding: utf8 -*-
from lib.q_learn import q_learn
import models
import gym


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: {:s} <model_name>".format(sys.argv[0]))
        sys.exit(0)
    env = gym.make(sys.argv[1])
    s = q_learn(env)().demo().account()
