"""Random spatial fields: smooth noise with a chosen correlation length."""

from __future__ import annotations

import numpy as np


def gaussian_field(
    shape: tuple[int, int], correlation_cells: float, rng: np.random.Generator, *, pad: bool = False
) -> np.ndarray:
    """A random field with mean 0 and standard deviation 1.

    Values stay similar over about ``correlation_cells`` cells (a Gaussian
    covariance, made by filtering white noise in the Fourier domain). The FFT
    wraps around at the edges; with ``pad=True`` the field is made on a larger
    grid and cropped, so opposite edges are not correlated.
    """
    if correlation_cells < 0:
        raise ValueError("correlation_cells must be >= 0")
    margin = int(np.ceil(3 * correlation_cells)) if pad else 0
    full = (shape[0] + 2 * margin, shape[1] + 2 * margin)
    noise = rng.standard_normal(full)
    if correlation_cells > 0:
        ky = np.fft.fftfreq(full[0])[:, None]
        kx = np.fft.rfftfreq(full[1])[None, :]
        kernel = np.exp(-2.0 * (np.pi * correlation_cells) ** 2 * (kx**2 + ky**2))
        noise = np.fft.irfft2(np.fft.rfft2(noise) * kernel, s=full)
    field = noise[margin : margin + shape[0], margin : margin + shape[1]]
    sd = field.std()
    return (field - field.mean()) / (sd if sd > 0 else 1.0)
