"""CSF and Frequency convertion"""

import numpy as np
import math

def csf_barten(u,
               sigma=0.5/60,
               k=3.0, T=0.1, X_0=60, X_max=12, N_max=15,
               n=0.03, p=1.2e6, E=500, phi_0=3e-8, u_0=7):
    """
    Barten (1999) Contrast Sensitivity Function 
    https://github.com/qianglisinoeusa/CSFpy/blob/main/Python/chromatic_spatial_CSFs.py
    """
    M_opt = np.exp(-2 * np.pi**2 * sigma**2 * u**2)
    S = (M_opt / k) / np.sqrt(
        2 / T * (1 / X_0**2 + 1 / X_max**2 + u**2 / N_max**2) *
        (1 / (n * p * E) + phi_0 / (1 - np.exp(-(u / u_0)**2)))
    )
    return S

class FrequencyConverter:
    """
    Args:
        mode: int
            Mode for convertion: 
                0 - cycles per pixel to cycles per degree,
                1 - cycles per degree to cycles per pixel
        frequencies: np.ndarray
            Array of the base frequencies to be converted.
        pixel_size: float
            Physical size of a pixel in meters.
        viewing_distance: float
            Viewing distance to the screen in meters.
    """

    def __init__(
        self,
        mode: int,
        frequencies: np.ndarray,
        pixel_size: float = 0.000114,
        viewing_distance: float = 0.5,
    ):
        self.mode = mode
        self.frequencies = frequencies
        self.pixel_size = pixel_size
        self.viewing_distance = viewing_distance

        self.frequencies_converted = self._convert()

    def _convert(self) -> np.ndarray:
        deg_per_pixel = math.degrees(2 * math.atan(self.pixel_size / (2 * self.viewing_distance)))
        if self.mode == 0:
            self.frequencies_converted = self.frequencies / deg_per_pixel
        elif self.mode == 1:
            self.frequencies_converted = self.frequencies * deg_per_pixel
        else: raise ValueError("mode must be 0 (cpp→cpd) or 1 (cpd→cpp).")
        return self.frequencies_converted
    
        