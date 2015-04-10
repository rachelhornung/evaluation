__author__ = 'korhammer'

import numpy as np


def based_floor(x, base=10):
    """
    rounds down to the last integer of base 10 (or other)
    """
    return np.int(base * np.floor(np.float(x) / base))


def based_ceil(x, base=10):
    """
    rounds up to the next integer of base 10 (or other)
    """
    return np.int(base * np.ceil(np.float(x) / base))


def float(x):
    return np.float(x) if len(x) > 0 else np.nan


def int(x):
    return np.int(x) if len(x) > 0 else 0

