#!/usr/bin/env python
# -*- coding: utf8 -*-
import gym
from gym import spaces

from .beyond_done import beyond_done
from .grid_world import GridWorld, world_play


class CliffGridWorld(gym.Env):

    def __init__(self):
        grid = self.grid = GridWorld(4, 12)
        grid[3, 0] = 1
        grid[3, 11] = 2
        for i in range(1, 11):
            grid[3, i] = 23
        self.observation_space = spaces.Discrete(grid.states)
        self.action_space = spaces.Discrete(4)
        self.steps_beyond_done = beyond_done()

    def reset(self):
        self.steps_beyond_done.reset()
        self.grid.reset()
        return self.grid.state

    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))
        grid = self.grid

        reward, done = 0, False
        if grid.isterminal(grid.state):
            done = True
        else:
            row, col = grid.state_step(grid.state, action)
            grid.set_state(row, col)
            if grid.isterminal(grid.state):
                done = True
            elif row == grid.rows - 1 and col:
                reward = -100
                grid.reset()
            else:
                reward = -1

        self.steps_beyond_done.step(done)

        return grid.state, reward, done, {}

    def render(self, mode="human"):
        return self.grid.render(mode)


if __name__ == "__main__":
    obj = gym.make('CliffGridWorld-v0')
    world_play(obj)
