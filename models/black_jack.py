#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys, random
import numpy as np
import gym
from gym import spaces, logger
from .hyperspace import hyperspace
from .getch import getch

##############################################################################
# 扑克牌
##============================================================================
class Cards(object):

    '''一副牌，每张牌是一个元组 (花色，分数)，其中 X代表分数10'''

    _color = ("Spade", "Heart", "Club", "Diamond")
    _point = tuple("A23456789XJQK")

    def __init__(self, king=False):
        self.cards = [(k, s) for k in self._color for s in self._point]
        if king:
            self.cards.append(("Black", "Joker"))
            self.cards.append(("Red", "Joker"))
        self.reset()

    def reset(self):
        '''牌被初始化，并洗牌'''
        self.i = len(self.cards)
        random.shuffle(self.cards)

    def get(self, loop=True):
        '''
        抓取一张牌：返回元组 ((当前牌花色, 当前牌点数)，还剩多少张)
        如果 loop 是假，没有更多牌的时候会抛出 AssertionError 错误
        '''
        i = self.i = self.i - 1
        if i < 0:
            assert loop, "haven't more"
            self.reset()
            i = self.i = self.i - 1
        return self.cards[i], i

##============================================================================
# 游戏者
##============================================================================
class BlackJackGamer(object):

    '''玩家'''

    _name2i = dict([(c, min(i, 9) + 1) for i, c in enumerate(Cards._point)])

    def __init__(self):
        self.reset()

    def reset(self):
        '''初始化'''
        self.cards = []     # 目前手中牌的列表
        self.A = False      # 手中牌里是否有 A
        self.score = 0      # 当前牌构成的分数 (A 记为 1分)

    def __call__(self, card):
        '''抓一张牌'''
        self.cards.append(card)
        _, point = card
        if point == 'A':
            self.A = True
        # 凡是超过 21 点的分数，都记为 22 这样状态空间大小可以确定
        self.score = min(22, self.score + self._name2i[point])

    def info(self):
        score = self.score
        if self.A and score < 12:
            score += 10
        return ''.join([s for _, s in self.cards]), -1 if score > 21 else score

    def bid(self) -> bool:
        '''是否继续抓牌'''
        raise NotImplementedError


class Player(BlackJackGamer):

    '''普通玩家'''

    def bid(self) -> bool:
        '''玩家策略'''
        score = self.score
        return (score < 10 or 11 < score < 20) if self.A else score < 20


class Dealer(BlackJackGamer):

    '''庄家'''

    def bid(self) -> bool:
        '''庄家策略'''
        score = self.score
        return (score < 7 or 11 < score < 17) if self.A else score < 17

##============================================================================
# 赌场环境
##============================================================================
class BlackJackCards(gym.Env):

    def __init__(self):
        '''
        状态包含庄家第一张牌    (1-10)
        玩家的点数              (2-22)
        玩家是否有可用的 A      (0-1)
        一共 10 * 20 * 2 = 400 种不同状态
        动作空间为 (不叫牌，叫牌)
        '''
        self.observation_space = spaces.Discrete(420)
        self.action_space = spaces.Discrete(2)
        self.state = None
        self.viewer = None
        self.steps_beyond_done = None
        self.dealer = Dealer()              # 庄家
        self.player = Player()              # 玩家
        self.cards = Cards()                # 一副扑克牌
        self._hs = hyperspace(10, 21, 2)    # 将状态与索引互相转换

    def reset(self):
        self.steps_beyond_done = None
        # objects reset
        cards, dealer, player = self.cards, self.dealer, self.player
        cards.reset()
        dealer.reset()
        player.reset()
        # 先给庄家两张牌
        dealer(cards.get()[0])
        dealer(cards.get()[0])
        # 再给玩家两张牌
        player(cards.get()[0])
        player(cards.get()[0])
        # 当前状态 (庄家第一张牌，玩家当前分数，玩家是否有 A)
        self.arena = self.make_arena()
        self.state = self.arena.send(None)
        return self.state

    def _state(self, hs, n2i, dealer, player):
        return hs.index(
            n2i[dealer.cards[0][1]] - 1,
            player.score - 2,
            1 if player.A else 0
        )

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def make_arena(self):
        cards, dealer, player = self.cards, self.dealer, self.player
        _state, hs, n2i = self._state, self._hs, BlackJackGamer._name2i
        S = _state(hs, n2i, dealer, player)
        action = yield S
        while action:
            player(cards.get()[0])
            S = _state(hs, n2i, dealer, player)
            action = yield S
        while dealer.bid():
            dealer(cards.get()[0])
        yield _state(hs, n2i, dealer, player)

    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))

        self.state = self.arena.send(action)
        dealer, player = self.dealer, self.player
        if player.score > 21:
            done, reward = True, -1
        elif action:
            done, reward = False, 0
        else:
            # 获取庄家和玩家信息
            dc, ds = dealer.info()
            gc, gs = player.info()
            done, reward = True, np.sign(gs - ds)

        if done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True.  You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1

        return self.state, reward, done, {}

    def render(self, mode="human"):
        dealer, player = self.dealer, self.player
        done = self.steps_beyond_done is not None
        print("Dealer -> ", end='')
        print("(\033[31m{:s}-{:s}\033[0m)".format(*dealer.cards[0]), end=' ')
        if done:
            for i in dealer.cards[1:]:
                print("({:s}-{:s})".format(*i), end=' ')
        else:
            for i in dealer.cards[1:]:
                print("(_____-_)", end=' ')
        print()
        print("Player -> ", end='')
        for i in player.cards:
            print("({:s}-{:s})".format(*i), end=' ')
        print()

##============================================================================
# 生成游戏记录
##============================================================================
def make_play_records(times=1, point=10000):
    '''生成 n局游戏记录，每 point 局输出一个小数点'''
    cards = Cards()
    dealer = Dealer()
    player = Player()

    ret = []
    for i in range(times):
        dealer.reset()
        player.reset()
        # 先给庄家两张牌
        dealer(cards.get()[0])
        dealer(cards.get()[0])
        # 再给玩家两张牌
        player(cards.get()[0])
        player(cards.get()[0])
        # 玩家先叫牌
        while player.bid():
            player(cards.get()[0])
        # 如果玩家没超过21点，庄家叫牌
        if player.score < 22:
            while dealer.bid():
                dealer(cards.get()[0])
        # 获取庄家和玩家信息
        dc, ds = dealer.info()
        gc, gs = player.info()
        # 添加记录 (庄家牌, 玩家牌, 玩家获得奖励)
        ret.append((dc, gc, np.sign(gs - ds)))

        if point and i % point == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
    if point:
        print()

    return ret


def world_play():
    obj = gym.make('BlackJackCards-v0')
    c, R = None, 0
    while True:
        obj.reset()
        done = False
        while True:
            action = 0 if c == 110 else 1 if c == 121 else -1
            if not (done or action < 0):
                _, r, done, _ = obj.step(action)
                R += r
            print("\033[2J")
            print("R: {:6.2f}".format(R))
            print('-' * 78)
            obj.render()
            print('-' * 78)
            print('Continue to bid ([y]es/[n]o)?  ', end='')
            if done:
                print('Press Enter restart a new game...', end='')
            else:
                print('Press q to exit...', end='')
            sys.stdout.flush()
            c = getch()
            if done and c == 13:
                break
            if c in (3, 4, 113):
                return print()

##============================================================================
# main
##============================================================================
if __name__ == "__main__":
    argv, argc = sys.argv, len(sys.argv)
    if 1 == argc:
        world_play()
    elif 3 == argc:
        from pprint import pprint
        pprint(make_play_records(int(argv[1])), stream=open(argv[2], 'w'))
    else:
        print("Usage: {:s} <times> <save_file>".format(argv[0]))
        sys.exit(0)

##============================================================================
# THE END
##############################################################################
