#!/usr/bin/env python
# -*- coding: utf8 -*-
import gym
from gym import spaces, logger

from .grid_world import GridWorld, world_play


class WindGridWorld(gym.Env):

    def __init__(self):
        grid = self.grid = GridWorld(7, 10)
        grid[3, 0] = 1
        grid[3, 7] = 2
        self.observation_space = spaces.Discrete(grid.states)
        self.action_space = spaces.Discrete(4)
        self.viewer = None
        self.steps_beyond_done = None
        self.wind_values = 0, 0, 0, 1, 1, 1, 2, 2, 1, 0

    def reset(self):
        self.grid.reset()
        self.steps_beyond_done = None
        return self.grid.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

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

        if done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1

        return grid.state, reward, done, {}

    def render(self, mode="human"):
        return self.grid.render(mode)


if __name__ == "__main__":
    obj = gym.make("WindGridWorld-v0")
    world_play(obj)
