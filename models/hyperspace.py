import numpy as np


class hyperspace(object):

    def __init__(self, *shape):
        self.shape = shape
        self.states = int(np.product(shape))
        x, xs = 1, [1]
        for i in reversed(shape[1:]):
            x *= i
            xs.append(x)
        self._xs = tuple(reversed(xs))

    def vector(self, index) -> tuple:
        ret = []
        for i in self._xs:
            x, index = divmod(index, i)
            ret.append(x)
        return tuple(ret)

    def index(self, *vector) -> int:
        return sum([i * j for i, j in zip(vector, self._xs)])


if __name__ == "__main__":
    pass
