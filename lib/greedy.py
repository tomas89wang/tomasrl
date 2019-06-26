#!/usr/bin/env python
# -*- coding: utf8 -*-
import random


class greedy(object):

    def __init__(self, Q):
        self.Q = Q
 
    def __call__(self, S):
        QS = self.Q[S]
        maxq = QS[QS.argmax()]
        return random.choice([i for i, q in enumerate(QS) if q == maxq])


class epsilon_greedy(greedy):

    def __init__(self, Q, epsilon=1e-5):
        super(epsilon_greedy, self).__init__(Q)
        self.epsilon = epsilon

    def __call__(self, S):
        if random.random() > self.epsilon:
            return super(epsilon_greedy, self).__call__(S)
        else:
            return random.randint(0, len(self.Q[S]) - 1)
