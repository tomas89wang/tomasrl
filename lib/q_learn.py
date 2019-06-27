#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
from .qtable_algo import qtable_algo


class q_learn(qtable_algo):

    def __call__(self, *, alpha=0.1, gamma=0.9, loop=0):
        env, Q, policy = self.env, self.Q, self.policy
        acc, loss = [], loop or float('inf')
        Q[:, :] = 0
        while loss > 0:
            Q_ = Q.copy()
            S = env.reset()
            times, reward, done, = 0, 0, False
            while not done:
                A = policy(S)
                S_, R, done, _ = env.step(A)
                q = Q[S, A]
                Q[S, A] = q + alpha * (R + gamma * Q[S_].max() - q)
                S = S_
                times += 1
                reward += R
            loss = (loss - 1) if loop else np.power(Q - Q_, 2).sum()
            acc.append((times, reward, loss))
        self.acc = acc


if __name__ == "__main__":
    pass
