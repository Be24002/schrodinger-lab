"""Time evolution of the 1D Schrodinger equation by two independent methods.

1. Crank-Nicolson:  (I + iH dt/2) psi' = (I - iH dt/2) psi.
   Unconditionally stable, exactly unitary, tridiagonal solves.
2. Split-operator (Strang splitting):
   psi -> exp(-iV dt/2) FFT exp(-iT dt) IFFT exp(-iV dt/2),
   spectral accuracy for the kinetic part.

The two methods agreeing is itself a validation: they share no code path.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def hamiltonian_fd(x: np.ndarray, potential: np.ndarray) -> sp.csr_matrix:
    """Finite-difference Hamiltonian with Dirichlet boundaries:
    H = -1/2 d^2/dx^2 + V."""
    dx = x[1] - x[0]
    n = len(x)
    diagonal = 1.0 / dx**2 + potential
    off = -0.5 / dx**2 * np.ones(n - 1)
    return sp.diags([off, diagonal, off], [-1, 0, 1], format="csr")


class CrankNicolson:
    """Unitary propagator via pre-factorized sparse solves."""

    def __init__(self, x: np.ndarray, potential: np.ndarray, dt: float):
        self.dt = dt
        hamiltonian = hamiltonian_fd(x, potential)
        n = len(x)
        identity = sp.identity(n, dtype=complex, format="csc")
        self._matrix_b = (identity - 0.5j * dt * hamiltonian).tocsc()
        self._solve_a = spla.splu(
            (identity + 0.5j * dt * hamiltonian).tocsc()
        )

    def step(self, psi: np.ndarray, n_steps: int = 1) -> np.ndarray:
        for _ in range(n_steps):
            psi = self._solve_a.solve(self._matrix_b @ psi)
        return psi


class SplitOperator:
    """Strang-split Fourier propagator (periodic images; keep the packet
    away from the edges or choose a domain where wraparound is harmless)."""

    def __init__(self, x: np.ndarray, potential: np.ndarray, dt: float):
        self.dt = dt
        self.potential = potential
        dx = x[1] - x[0]
        k = 2.0 * np.pi * np.fft.fftfreq(len(x), d=dx)
        self._kinetic_phase = np.exp(-1j * dt * k**2 / 2.0)
        self._half_potential_phase = np.exp(-0.5j * dt * potential)

    def step(self, psi: np.ndarray, n_steps: int = 1) -> np.ndarray:
        for _ in range(n_steps):
            psi = self._half_potential_phase * psi
            psi = np.fft.ifft(np.fft.fft(psi) * self._kinetic_phase)
            psi = self._half_potential_phase * psi
        return psi
