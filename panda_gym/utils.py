from typing import Union

import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)
