"""Spatial grid and standard potentials (units: hbar = m = 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    """Uniform 1D grid on [-length/2, length/2) with N points."""

    n_points: int = 800
    length: float = 16.0

    def __post_init__(self):
        self.dx = self.length / self.n_points
        self.x = (np.arange(self.n_points) - self.n_points / 2 + 0.5) * self.dx

    # -------------------------------------------------------- potentials

    def harmonic(self, omega: float = 1.0) -> np.ndarray:
        return 0.5 * omega**2 * self.x**2

    def square_barrier(self, height: float = 2.0, width: float = 1.0) -> np.ndarray:
        return np.where(np.abs(self.x) <= width / 2, height, 0.0)

    def infinite_well_walls(self, half_width: float) -> np.ndarray:
        """Large finite walls approximating the infinite well (for dynamics);
        for spectra use the Dirichlet solver in ``stationary``."""
        return np.where(np.abs(self.x) <= half_width, 0.0, 1e6)

    # ------------------------------------------------------- wavepackets

    def gaussian_packet(self, x0: float, k0: float, sigma: float) -> np.ndarray:
        """Normalized Gaussian wavepacket with momentum k0."""
        packet = np.exp(-((self.x - x0) ** 2) / (2 * sigma**2)) * np.exp(
            1j * k0 * self.x
        )
        return packet / np.sqrt(np.trapezoid(np.abs(packet) ** 2, self.x))

    def coherent_state(self, x0: float, omega: float = 1.0) -> np.ndarray:
        """Ground state of the oscillator displaced by x0 (a coherent state):
        under harmonic evolution it oscillates classically without spreading."""
        packet = np.exp(-0.5 * omega * (self.x - x0) ** 2)
        return packet / np.sqrt(np.trapezoid(np.abs(packet) ** 2, self.x))
