#!/usr/bin/env python
# -*- coding: utf8 -*-
import time
import numpy as np
from .greedy import greedy, epsilon_greedy


class sarsa(object):

    def __init__(self, env, *, Q=None, epsilon=1e-5):
        self.env = env
        self.shape = env.observation_space.n, env.action_space.n
        if Q is None:
            Q = np.zeros(self.shape, dtype=np.float32)
        else:
            assert Q.shape == self.shape
        self.Q = Q
        self.policy = epsilon_greedy(Q, epsilon)

    def __call__(self, *, alpha=0.1, gamma=0.9):
        env, Q, policy = self.env, self.Q, self.policy
        acc, loss = [], float('inf')
        Q[:, :] = 0
        while loss > 0 or not acc:
            Q_ = Q.copy()
            S = env.reset()
            A = policy(S)
            T, R, done, = 0, 0, False
            while not done:
                S_, reward, done, _ = env.step(A)
                A_ = policy(S_)
                q = Q[S, A]
                Q[S, A] = q + alpha * (reward + gamma * Q[S_, A_] - q)
                S, A = S_, A_
                T += 1
                R += reward
            loss = np.power(Q - Q_, 2).sum()
            acc.append((T, R, loss))
        self.acc = acc
        return self

    def demo(self):
        policy = greedy(self.Q)
        done = False
        env = self.env
        S = env.reset()
        while not done:
            print("\033[2J", end="")
            env.render()
            time.sleep(0.2)
            A = policy(S)
            S, reward, done, _ = env.step(A)
        print("\033[2J", end="")
        env.render()
        return self

    def account(self):
        acc = self.acc
        print("迭代次数: {:d}".format(len(acc)))
        print("前十次步数:", [i[0] for i in acc[:10]])
        print("前十次奖励:", [i[1] for i in acc[:10]])
        print("后十次步数:", [i[0] for i in acc[-10:]])
        print("后十次奖励:", [i[1] for i in acc[-10:]])
        return self


if __name__ == "__main__":
    pass
