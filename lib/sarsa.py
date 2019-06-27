#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
from .qtable_algo import qtable_algo


class sarsa(qtable_algo):

    def __call__(self, *, alpha=0.1, gamma=0.9, loop=0):
        env, Q, policy = self.env, self.Q, self.policy
        acc, loss = [], loop or float('inf')
        Q[:, :] = 0
        while loss > 0:
            Q_ = Q.copy()
            S = env.reset()
            A = policy(S)
            times, reward, done, = 0, 0, False
            while not done:
                S_, R, done, _ = env.step(A)
                A_ = policy(S_)
                q = Q[S, A]
                Q[S, A] = q + alpha * (reward + gamma * Q[S_, A_] - q)
                S, A = S_, A_
                times += 1
                reward += R
            loss = (loss - 1) if loop else np.power(Q - Q_, 2).sum()
            acc.append((times, reward, loss))
        self.acc = acc


if __name__ == "__main__":
    pass
