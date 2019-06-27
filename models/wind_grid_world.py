#!/usr/bin/env python
# -*- coding: utf8 -*-
import gym
from gym import spaces

from .beyond_done import beyond_done
from .grid_world import GridWorld, world_play


class WindGridWorld(gym.Env):

    def __init__(self):
        grid = self.grid = GridWorld(7, 10)
        grid[3, 0] = 1
        grid[3, 7] = 2
        self.observation_space = spaces.Discrete(grid.states)
        self.action_space = spaces.Discrete(4)
        self.steps_beyond_done = beyond_done()
        self.wind_values = 0, 0, 0, 1, 1, 1, 2, 2, 1, 0

    def reset(self):
        self.steps_beyond_done.reset()
        self.grid.reset()
        return self.grid.state

    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))
        grid = self.grid

        if grid.isterminal(grid.state):
            done = True
            reward = 0
        else:
            row, col = grid.state_step(grid.state, action, clip=False)
            col = grid.clip_col(col)
            row -= self.wind_values[col]
            row = grid.clip_row(row)
            done = grid.isterminal(grid.set_state(row, col))
            reward = 0 if done else -1

        self.steps_beyond_done.step(done)

        return grid.state, reward, done, {}

    def render(self, mode="human"):
        return self.grid.render(mode)


if __name__ == "__main__":
    obj = gym.make("WindGridWorld-v0")
    world_play(obj)
