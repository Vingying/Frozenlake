from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample


def random_generate_map(size=8):
    valid = False
    def search_path(x, y):
        


    while not valid:
