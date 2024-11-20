import numpy as np
from utils import check_shapes

rng = np.random.default_rng(0)

def trajectory_length(data : np.ndarray) -> float:
    return np.sum(np.sqrt(np.sum(np.diff(data[:, 1:4], axis=0)**2, axis=1)))

def ate(data : list) -> float:
    data = check_shapes(data)
    return np.sqrt(np.mean(np.sum(data[0][:, 1:4] - data[1][:, 1:4], axis=1)**2))

def re(data : list, indices : list) -> float:
    data = check_shapes(data)
    return ate([data[0][indices[0]:indices[1]], data[1][indices[0]:indices[1]]])

def re_statistics(data : list, n_samples : int, length : int) -> tuple:
    data = check_shapes(data)
    errors = []
    for _ in range(n_samples):
        # Pick a random starting index for the batch
        start_idx = rng.integers(0, len(data[0]) - length + 1)
        errors.append(re(data, [start_idx, start_idx + length]))
    return np.mean(errors), np.std(errors)