from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def get_velocity_kernel(N: int, sigma: float):
    velocity_kernel = np.zeros((N - 1, N))
    velocity_kernel[:, 1:] = np.eye(N - 1)
    velocity_kernel -= np.eye(N - 1, N)
    VTV = velocity_kernel.T @ velocity_kernel
    score_kernel = -np.linalg.inv(VTV + np.eye(N) / (sigma**2)) @ VTV / (sigma**2)
    return score_kernel


def compute_kinetic_energy_score(trajectory: np.ndarray, sigma):
    return np.einsum(
        "it,tad->iad", get_velocity_kernel(trajectory.shape[0], sigma), trajectory
    )
