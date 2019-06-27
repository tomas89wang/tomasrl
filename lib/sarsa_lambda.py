#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
from .qtable_algo import qtable_algo


class sarsa_lambda(qtable_algo):

    def __call__(self, *, alpha=0.1, gamma=0.9, lambda_=0.3, loop=0):
        gamma_lambda = gamma * lambda_
        env, Q, policy, shape = self.env, self.Q, self.policy, self.shape
        acc, loss = [], loop or float('inf')
        Q[:, :] = 0
        while loss > 0:
            Q_ = Q.copy()
            E = np.zeros(shape, dtype=np.float32)
            S = env.reset()
            A = policy(S)
            times, reward, done, = 0, 0, False
            while not done:
                S_, R, done, _ = env.step(A)
                A_ = policy(S_)
                alpha_delta = alpha * (R + gamma * Q[S_, A_] - Q[S, A])
                E[S, A] = E[S, A] + 1
                for s in range(shape[0]):
                    for a in range(shape[1]):
                        Q[S, A] = Q[S, A] + alpha_delta * E[s, a]
                        E[s, a] = gamma_lambda * E[s, a]
                S, A = S_, A_
                times += 1
                reward += R
            loss = (loss - 1) if loop else np.power(Q - Q_, 2).sum()
            acc.append((times, reward, loss))
        self.acc = acc


if __name__ == "__main__":
    pass
