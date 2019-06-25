#!/usr/bin/env python
# -*- coding: utf8 -*-
import numpy as np
import random
from .getch import getch


class GridWorld(object):

    _action_values = (-1, 0), (0, 1), (0, -1), (1, 0)
    _value2char = ' @$ABCDEFGHIJKLMNOPQUVWXYZ'

    def __init__(self, rows, cols, dtype=np.int32):
        self.rows = rows
        self.cols = cols
        self.states = self.rows * self.cols
        self.table = np.zeros((self.states,), dtype=dtype)
        self.state = None

    def reset(self) -> None:
        states = self.initial_states()
        random.shuffle(states)
        self.set_state(states[0])

    def set_state(self, index, col=None) -> int:
        if col is not None:
            index = self.rc2index(index, col)
        self.state = index
        return index

    def tochar(self, state) -> str:
        return '#' if state == self.state else self._value2char[self[state]]

    def __getitem__(self, index):
        if type(index) in (tuple, list):
            index = self.rc2index(*index)
        return self.table[index]

    def __setitem__(self, index, value):
        if type(index) in (tuple, list):
            index = self.rc2index(*index)
        self.table[index] = value

    def batch_set(self, *args):
        for index, value in args:
            self.__setitem__(index, value)
        return self

    def index2rc(self, i) -> tuple:
        cols = self.cols
        col = i % cols
        row = (i - col) // cols
        return row, col

    def rc2index(self, row, col) -> int:
        return row * self.cols + col

    def inrows(self, row) -> bool:
        return 0 <= row < self.rows

    def incols(self, col) -> bool:
        return 0 <= col < self.cols

    def instates(self, i) -> bool:
        return 0 <= i < self.states

    def initial_states(self) -> list:
        return [i for i, v in enumerate(self.table) if 1 == v]

    def terminal_states(self) -> list:
        return [i for i, v in enumerate(self.table) if 2 == v]

    def isterminal(self, state) -> bool:
        return 2 == self.table[state]

    def clip_row(self, row) -> int:
        return 0 if row < 0 else row if row < self.rows else self.rows - 1

    def clip_col(self, col) -> int:
        return 0 if col < 0 else col if col < self.cols else self.cols - 1

    def clip(self, row, col) -> tuple:
        return self.clip_row(row), self.clip_col(col)

    def state_step(self, state, action, *, clip=True) -> tuple:
        row, col = self.index2rc(state)
        r_, c_ = self._action_values[action]
        row, col = row + r_, col + c_
        return self.clip(row, col) if clip else (row, col)

    def render(self, model=None) -> None:
        cols, tochar, rc2index = self.cols, self.tochar, self.rc2index
        side_line = ''.join(['+', '-' * self.cols, '+'])
        print(side_line)
        for r in range(self.rows):
            line = ''.join([tochar(rc2index(r, c)) for c in range(cols)])
            print(''.join(['|', line, '|']))
        print(side_line)


def world_play(world):
    c, done, R = None, False, 0
    world.reset()
    while c not in (3, 4):
        action = {107: 0, 108: 1, 104: 2, 106: 3}.get(c, -1)
        if action >= 0:
            _, r, done, _ = world.step(action)
            R += r
        print("\033[2J")
        print("R: {:6.2f}".format(R))
        world.render()
        if done:
            break
        c = getch()
