"""Generation of timesteps for inference"""

import torch
import numpy as np


def make_weighted_timesteps(num_steps: int, weights: np.ndarray, arguments: np.ndarray) -> torch.Tensor:
    """
    Create non-uniform timesteps in [0, 1] based on given weights.
    Larger weights = smaller step sizes.

    Args:
        num_steps (int): number of steps.
        weights (np.ndarray): 1D array of weights.
        arguments (np.ndarray): 1D array of arguments corresponding to weights (e.g. uniform timesteps).

    Returns:
        timesteps (torch.Tensor): tensor of shape (num_steps + 1,) with timesteps in [0, 1]
        dt (torch.Tensor): tensor of shape (num_steps) with differences between consecutive timesteps.
    """
    assert weights.ndim == 1, "weights must be 1D"
    assert (weights >= 0).all(), "weights must be positive"

    # quantile grid for the inverse
    q = np.linspace(0.0, 1 - (1 / num_steps), num_steps)
    q = np.concatenate([q, np.array([1.0])], axis=0)

    # making sure the weights are normalized
    weights = weights / weights.sum()

    # calculating cumulative function of the weights
    cdf = np.concatenate(([0], np.cumsum(weights, axis=0)))

    # inverse of the cdf
    inv_cdf = np.interp(q, cdf, arguments)
    timesteps = torch.Tensor(inv_cdf)
    dt = timesteps[1:] - timesteps[:-1]

    return timesteps, dt