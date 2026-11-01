"""Exact results every numerical claim is validated against (hbar = m = 1)."""

from __future__ import annotations

import numpy as np


def harmonic_levels(n_max: int = 10, omega: float = 1.0) -> np.ndarray:
    """E_n = (n + 1/2) hbar omega."""
    return omega * (np.arange(n_max) + 0.5)


def gaussian_packet_momentum_distribution(k, k0: float, sigma: float) -> np.ndarray:
    """|phi(k)|^2 for the Gaussian packet exp(-(x-x0)^2/2sigma^2) e^{ik0 x}."""
    return (
        sigma / np.sqrt(np.pi) * np.exp(-((k - k0) ** 2) * sigma**2)
    )


def barrier_transmission(energy: float, height: float, width: float) -> float:
    """Exact plane-wave transmission through a square barrier.

    T = [1 + V0^2 sinh^2(kappa a) / (4 E (V0 - E))]^{-1},  E < V0,  kappa = sqrt(2(V0-E))
    T = [1 + V0^2 sin^2(k2 a)    / (4 E (E - V0))]^{-1},  E > V0,  k2 = sqrt(2(E-V0))
    T = 1 / (1 + V0 a^2 / 4)                              E = V0 (limit)
    """
    if energy <= 0:
        return 0.0
    if abs(energy - height) < 1e-12:  # E = V0 limit
        return float(1.0 / (1.0 + height * width**2 / 4.0))
    if energy < height:
        kappa = np.sqrt(2.0 * (height - energy))
        numerator = height**2 * np.sinh(kappa * width) ** 2
        return float(1.0 / (1.0 + numerator / (4.0 * energy * (height - energy))))
    k2 = np.sqrt(2.0 * (energy - height))
    numerator = height**2 * np.sin(k2 * width) ** 2
    return float(1.0 / (1.0 + numerator / (4.0 * energy * (energy - height))))


def packet_average_transmission(
    k_grid: np.ndarray, k0: float, sigma: float, height: float, width: float
) -> float:
    """The measurable transmission of a Gaussian packet: the plane-wave T(E)
    averaged over the packet's momentum distribution."""
    weights = gaussian_packet_momentum_distribution(k_grid, k0, sigma)
    energies = k_grid**2 / 2.0
    transmissions = np.array(
        [barrier_transmission(e, height, width) for e in energies]
    )
    normalization = np.trapezoid(weights, k_grid)
    return float(np.trapezoid(weights * transmissions, k_grid) / normalization)


def free_gaussian_width(t: float, sigma0: float) -> float:
    """Width of a FREE Gaussian wavepacket exp(-(x-x0)^2 / (2 sigma0^2)).

    The exponent parameter evolves as sigma(t) = sigma0 sqrt(1 + (t/sigma0^2)^2);
    the RMS width of the probability density is sigma(t)/sqrt(2).
    """
    return sigma0 * np.sqrt(1.0 + (t / sigma0**2) ** 2)


def coherent_state_position(t: float, x0: float, omega: float = 1.0) -> float:
    """Ehrenfest: <x>(t) of a coherent state = classical oscillator."""
    return x0 * np.cos(omega * t)
