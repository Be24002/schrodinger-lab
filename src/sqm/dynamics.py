"""Observables for wavepacket dynamics (hbar = m = 1)."""

from __future__ import annotations

import numpy as np


def norm(x: np.ndarray, psi: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid(np.abs(psi) ** 2, x)))


def position_expectation(x: np.ndarray, psi: np.ndarray) -> float:
    density = np.abs(psi) ** 2
    return float(np.trapezoid(x * density, x))


def width(x: np.ndarray, psi: np.ndarray) -> float:
    """RMS width of the packet."""
    density = np.abs(psi) ** 2
    x_mean = position_expectation(x, psi)
    variance = np.trapezoid((x - x_mean) ** 2 * density, x)
    return float(np.sqrt(variance))


def momentum_expectation(x: np.ndarray, psi: np.ndarray) -> float:
    dx = x[1] - x[0]
    dpsi = np.gradient(psi, dx)
    return float(np.real(np.trapezoid(np.conj(psi) * (-1j) * dpsi, x)))


def energy_expectation(x: np.ndarray, psi: np.ndarray, potential: np.ndarray) -> float:
    """<H> via the finite-difference kinetic energy."""
    dx = x[1] - x[0]
    d2 = (np.roll(psi, 1) - 2 * psi + np.roll(psi, -1)) / dx**2
    kinetic = np.real(np.trapezoid(np.conj(psi) * (-0.5) * d2, x))
    potential_energy = float(np.trapezoid(potential * np.abs(psi) ** 2, x))
    return float(kinetic + potential_energy)


def transmitted_fraction(x: np.ndarray, psi: np.ndarray, barrier_right_edge: float) -> float:
    """Probability beyond the barrier."""
    mask = x > barrier_right_edge
    density = np.abs(psi) ** 2
    return float(np.trapezoid(density[mask], x[mask]))
