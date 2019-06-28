#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import gym
from gym import spaces

from .getch import getch
from .beyond_done import beyond_done


class PuckWorld(gym.Env):

    steps_move_puck = 50
    steps_max = float('inf')
    speed_delta = 0.03
    speed_limit = 0.5
    speed_keep = True

    steps = None
    state = None

    def config(self, **items):
        conf_k = 'steps_move_puck', 'steps_max', 'peed_delta', 'speed_keep'
        for k, v in items.items():
            assert k in conf_k
            setattr(self, k, v)

    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, (6,), np.float32)
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.steps_beyond_done = beyond_done()
        self.seed()

    def reset(self):
        self.steps_beyond_done.reset()
        self.steps = 0
        a_x, a_y, s_x, s_y, t_x, t_y = tuple(self.observation_space.sample())
        self.state = a_x, a_y, 0, 0, t_x, t_y
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))

        agent_x, agent_y, speed_x, speed_y, target_x, target_y = self.state

        agent_x += speed_x
        while not -1 <= agent_x <= 1:
            if agent_x < -1:
                agent_x, speed_x = -2 - agent_x, -speed_x
            if agent_x > 1:
                agent_x, speed_x = 2 - agent_x, -speed_x

        agent_y += speed_y
        while not -1 <= agent_y <= 1:
            if agent_y < -1:
                agent_y, speed_y = -2 - agent_y, -speed_y
            if agent_y > 1:
                agent_y, speed_y = 2 - agent_y, -speed_y

        steps = self.steps = self.steps + 1
        if 0 == steps % self.steps_move_puck:
            target_x, target_y = tuple(self.observation_space.sample())[-2:]
            if not self.speed_keep:
                speed_x, speed_y = 0

        sl = self.speed_limit
        speed_x = max(-sl, min(sl, speed_x + action[0] * self.speed_delta))
        speed_y = max(-sl, min(sl, speed_y + action[1] * self.speed_delta))

        self.state = agent_x, agent_y, speed_x, speed_y, target_x, target_y
        done = steps > self.steps_max
        reward = -(target_x - agent_x) ** 2 - (target_y - agent_y) ** 2

        self.steps_beyond_done.step(done)

        return self.state, reward, done, {}

    def render(self, mode="human"):
        # 初始化场地
        head, body = '+' + '-' * 63 + '+', '|' + ' ' * 63 + '|'
        s = [list(body if 0 < i < 23 else head) for i in range(24)]

        # 辅助函数与状态
        pos_c = lambda x: int(x * 30 + 30) + 1
        pos_r = lambda y: int(y * 10 + 10) + 1
        a_x, a_y, s_x, s_y, t_x, t_y = self.state
        s[1].append(' A -> agent')
        s[2].append(' S -> speed')
        s[3].append(' T -> target')
        s[6].append(' Ax: {:> 3.3f}'.format(a_x * 100))
        s[7].append(' Ay: {:> 3.3f}'.format(a_y * 100))
        s[9].append(' Sx: {:> 3.3f}'.format(s_x * 100))
        s[10].append(' Sy: {:> 3.3f}'.format(s_y * 100))
        s[12].append(' Tx: {:> 3.3f}'.format(t_x * 100))
        s[13].append(' Ty: {:> 3.3f}'.format(t_y * 100))
        s[16].append(' Press ...')
        s[17].append('   q  -> quit')
        s[18].append('   h  -> left')
        s[19].append('   j  -> down')
        s[20].append('   k  -> up')
        s[21].append('   l  -> right')
        s[22].append(' Enter-> keep')

        # 画上冰球
        r, c = pos_r(t_y), pos_c(t_x)
        s[r][c:c+3] = list('/T\\')
        s[r+1][c:c+3] = list('\\_/')

        # 画上 agent
        r, c = pos_r(a_y), pos_c(a_x)
        s[r][c:c+3] = list('###')
        s[r+1][c:c+3] = list('###')

        # 标注速度
        s_x, s_y = int(s_x * 10 + 0.5), int(s_y * 10 + 0.5)
        sx, sy = str(abs(s_x)), str(abs(s_y))
        if s_x > 0:
            if s_y > 0:
                s[r+1][c+3] = sx
                s[r+2][c+2] = sy
            else:
                s[ r ][c+3] = sx
                s[r-1][c+2] = sy
        else:
            if s_y > 0:
                s[r+1][c-1] = sx
                s[r+2][ c ] = sy
            else:
                s[ r ][c-1] = sx
                s[r-1][ c ] = sy

        print('\n'.join([''.join(i) for i in s]))


if __name__ == "__main__":
    import sys
    obj = gym.make("PuckWorld-v0")
    obj.config(steps_max=1000)
    obj.reset()
    obj.render()
    step, reward, R, done = 0, 0, 0, False
    while True:
        print('{:03d} => Reward({:1.2f}/{:1.2f}) $ '.format(
            step, R, reward
        ), end='')
        sys.stdout.flush()
        c = getch()
        if done or c in (3, 4, 113):
            break
        action = {
            107: [0, -1],   # up
            106: [0, 1],    # down
            104: [-1, 0],   # left
            108: [1, 0],    # right
            13:  [0, 0],    # keep
        }.get(c)
        if action is not None:
            S, R, done, _ = obj.step(action)
            reward += R
            print('\033[2J')
            obj.render()
        step += 1
    print()
