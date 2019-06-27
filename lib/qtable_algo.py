#!/usr/bin/env python
# -*- coding: utf8 -*-
import time
import numpy as np
import gym
from .greedy import greedy, epsilon_greedy


class qtable_algo(object):

    def __init__(self, env, *, Q=None, epsilon=1e-5):
        self.env = env
        self.shape = env.observation_space.n, env.action_space.n
        if Q is None:
            Q = np.zeros(self.shape, dtype=np.float32)
        else:
            assert Q.shape == self.shape
        self.Q = Q
        self.policy = epsilon_greedy(Q, epsilon)
        self.acc = []

    def __call__(self, *, alpha=0.1, gamma=0.9, times=0):
        raise NotImplementedError

    def demo(self, times=1):
        policy = greedy(self.Q)
        reward = 0
        for i in range(times):
            view = i == times - 1
            done = False
            env = self.env
            S = env.reset()
            while not done:
                if view:
                    print("\033[2J", end="")
                    env.render()
                time.sleep(0.2)
                A = policy(S)
                S, R, done, _ = env.step(A)
                reward += R
            if view:
                print("\033[2J", end="")
                env.render()
        print('-' * 78)
        print('times: {:d}; reward: {:6.2f}'.format(times, reward))

    def account(self):
        acc = self.acc
        print("迭代次数: {:d}".format(len(acc)))
        print("前十次步数:", [i[0] for i in acc[:10]])
        print("前十次奖励:", [i[1] for i in acc[:10]])
        print("后十次步数:", [i[0] for i in acc[-10:]])
        print("后十次奖励:", [i[1] for i in acc[-10:]])

    @classmethod
    def live_demo(self, model_name, *, loop=0, play_times=1):
       env = gym.make(model_name)
       obj = self(env)
       obj(loop=loop)
       obj.demo(play_times)
       obj.account()
       return obj


if __name__ == "__main__":
    pass
