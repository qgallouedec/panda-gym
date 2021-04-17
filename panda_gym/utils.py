import numpy as np


def distance(a, b):
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)
