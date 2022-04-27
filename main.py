from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

from gym import Env, spaces, utils


# from gym.envs.toy_text.utils import categorical_sample


def random_generate_map(size=8, probability=0.7):
    valid = False

    def search_path(res, sx, sy, gx, gy):
        queue, visited = [], set()
        queue.append([sx, sy])
        directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        while queue:
            ux, uy = queue.pop()
            if (ux, uy) in visited:
                continue
            visited.add((ux, uy))
            for [dx, dy] in directions:
                next_x, next_y = dx + ux, dy + uy
                if min(next_x, next_y) < 0 or max(next_x, next_y) >= size:
                    continue
                if res[next_x][next_y] == 'H' or (next_x, next_y) in visited:
                    continue
                if [next_x, next_y] == [gx, gy]:
                    return True
                queue.append([next_x, next_y])
        return False

    while not valid:
        probability = min(probability, 1 - probability)
        res = np.random.choice(['F', 'H'], (size, size), p=[probability, 1 - probability])
        sx, sy = np.random.randint(0, size), np.random.randint(0, size)
        gx, gy = np.random.randint(0, size), np.random.randint(0, size)
        while abs(sx - gx) + abs(sy - gy) < size:
            gx, gy = np.random.randint(0, size), np.random.randint(0, size)
        res[sx][sy], res[gx][gy] = 'S', 'G'
        valid = search_path(res, sx, sy, gx, gy)
    return ["".join(x) for x in res]


class solution:

    def __init__(self, size=8):
        env_map = random_generate_map(size)
        for i in env_map:
            print(i)


c = solution()
