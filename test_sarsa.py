#!/usr/bin/env python
# -*- coding: utf8 -*-
from lib.sarsa import sarsa
import models
import gym


if __name__ == "__main__":
    import sys

    argc, argv = len(sys.argv), sys.argv
    if 1 == argc:
        print("Usage: {:s} <model_name> [loop] [play_times]".format(argv[0]))
        sys.exit(0)

    obj = sarsa.live_demo(
        argv[1],
        loop = 0 if argc < 3 else int(argv[2]),
        play_times = 1 if argc < 4 else int(argv[3])
    )
