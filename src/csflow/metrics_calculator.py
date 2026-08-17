"""RSignal, ΔRSignal calculator"""

import numpy as np


class MetricsCalculator:
    """
    Class for evaluating spectral metrics RSignal, ΔRSignal
    from RAPSD.

    Args:
        mean_rapsd_data : np.ndarray
            Mean RAPSD (signal) over timesteps and frequencies.
        mean_rapsd_noise : np.ndarray
            Mean RAPSD (noise) over timesteps and frequencies.
        timesteps : np.ndarray
            Array of diffusion timesteps (0 → 1).
        frequencies : np.ndarray
            Array of spatial frequencies (cycles per degree or pixel).

    Notes
    -----
    This class assumes that `mean_rapsd_data` and `mean_rapsd_noise`
    are aligned with timesteps, i.e. shape (T, F), where
    T = number of timesteps, F = number of frequency bins.

    Default path uses a(t)=t and b(t)=1-t, as in linear flow matching.
    """

    def __init__(
        self,
        mean_rapsd_data: np.ndarray,
        mean_rapsd_noise: np.ndarray,
        timesteps: np.ndarray,
        frequencies: np.ndarray,
    ):
        self.mean_rapsd_data = mean_rapsd_data
        self.mean_rapsd_noise = mean_rapsd_noise
        self.timesteps = np.array(timesteps)
        self.frequencies = frequencies

        self.retained_signal = self._compute_all_retained_signal()
        self.delta_retained_signal = self._compute_delta_retained_signal()

    def calc_retained_signal(self, timestep: float) -> np.ndarray:
        """
        Compute retained signal for a single timestep (fraction of power retained above threshold).
        """
        eps = 1e-6
        alpha_t = np.clip(timestep, eps, 1.0)
        sigma_t = np.clip(1.0 - timestep, eps, 1.0)

        Pn = (alpha_t**2) * self.mean_rapsd_data + (sigma_t**2) * self.mean_rapsd_noise
        retained = (alpha_t**2 * self.mean_rapsd_data) / (Pn + eps)

        return np.clip(retained, 0.0, 1.0)

    def _compute_all_retained_signal(self) -> np.ndarray:
        """
        Compute retained signal for all timesteps (fraction of power retained above threshold).
        """
        self.retained_signal = np.array(
            [self.calc_retained_signal(t) for t in self.timesteps]
        )
        return self.retained_signal

    def _compute_delta_retained_signal(self) -> np.ndarray:
        """Compute Δ retained signal (difference between consecutive timesteps)."""
        if self.retained_signal is None:
            self._compute_all_retained_signal()

        delta = np.diff(self.retained_signal, axis=0)
        delta = np.maximum(delta, 0.0)
        self.delta_retained_signal = delta
        return delta
