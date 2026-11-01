"""Physics regression tests: every numerical claim vs an exact formula."""

import numpy as np
import pytest

from sqm.analytic import (
    barrier_transmission,
    coherent_state_position,
    free_gaussian_width,
    harmonic_levels,
    packet_average_transmission,
)
from sqm.dynamics import (
    energy_expectation,
    momentum_expectation,
    norm,
    position_expectation,
    transmitted_fraction,
    width,
)
from sqm.grid import Grid
from sqm.solvers import CrankNicolson, SplitOperator
from sqm.stationary import bound_states, infinite_well_levels

DT = 0.005


# --------------------------------------------------------- spectra


def test_harmonic_spectrum_matches_exact():
    grid = Grid(n_points=1200, length=14.0)
    values, _ = bound_states(grid.x, grid.harmonic(), n_levels=10)
    exact = harmonic_levels(10)
    assert np.allclose(values, exact, rtol=1e-3)


def test_ground_state_is_the_gaussian():
    grid = Grid(n_points=1200, length=14.0)
    _, vectors = bound_states(grid.x, grid.harmonic(), n_levels=3)
    analytic = np.pi ** (-0.25) * np.exp(-grid.x**2 / 2)
    assert np.max(np.abs(vectors[:, 0] - analytic)) < 1e-4


def test_infinite_well_spectrum():
    half_width = 2.0
    grid = Grid(n_points=800, length=2 * half_width)
    interior = np.abs(grid.x) < half_width - 1e-9
    values, _ = bound_states(grid.x[interior], np.zeros(interior.sum()), 6)
    exact = infinite_well_levels(half_width, 6)
    # the finite-difference box is effectively (N+1) dx wide: O(dx^2) offset
    assert np.allclose(values, exact, rtol=5e-3)


# ------------------------------------------------------ unitarity


def test_crank_nicolson_preserves_norm():
    grid = Grid(n_points=800, length=20.0)
    psi = grid.coherent_state(x0=2.0)
    propagator = CrankNicolson(grid.x, grid.harmonic(), DT)
    psi = propagator.step(psi, n_steps=600)
    assert norm(grid.x, psi) == pytest.approx(1.0, abs=1e-10)


def test_split_operator_preserves_norm():
    grid = Grid(n_points=800, length=20.0)
    psi = grid.coherent_state(x0=2.0)
    propagator = SplitOperator(grid.x, grid.harmonic(), DT)
    psi = propagator.step(psi, n_steps=600)
    assert norm(grid.x, psi) == pytest.approx(1.0, abs=1e-10)


def test_crank_nicolson_conserves_energy():
    grid = Grid(n_points=800, length=20.0)
    potential = grid.harmonic()
    psi = grid.coherent_state(x0=2.0)
    e0 = energy_expectation(grid.x, psi, potential)
    propagator = CrankNicolson(grid.x, potential, DT)
    psi = propagator.step(psi, n_steps=600)
    e1 = energy_expectation(grid.x, psi, potential)
    assert e1 == pytest.approx(e0, rel=1e-8)


def test_methods_agree_with_each_other():
    grid = Grid(n_points=800, length=30.0)
    psi0 = grid.gaussian_packet(x0=-5.0, k0=3.0, sigma=1.0)
    psi_cn = CrankNicolson(grid.x, np.zeros_like(grid.x), DT).step(psi0, 400)
    psi_sp = SplitOperator(grid.x, np.zeros_like(grid.x), DT).step(psi0, 400)
    overlap = abs(np.trapezoid(np.conj(psi_cn) * psi_sp, grid.x)) ** 2
    # the two schemes carry different O(dt^2) discretization errors
    assert overlap > 1 - 2e-4


# ------------------------------------------------------ dynamics


def test_initial_packet_has_momentum_k0():
    grid = Grid(n_points=800, length=20.0)
    psi = grid.gaussian_packet(x0=-3.0, k0=2.5, sigma=1.0)
    # the finite-difference gradient carries an O((k dx)^2) bias
    assert momentum_expectation(grid.x, psi) == pytest.approx(2.5, rel=2e-3)


def test_ehrenfest_coherent_state_follows_classics():
    grid = Grid(n_points=800, length=20.0)
    psi = grid.coherent_state(x0=3.0)
    propagator = CrankNicolson(grid.x, grid.harmonic(), DT)
    max_error = 0.0
    for step in range(int(2 * np.pi / DT) + 1):
        if step % 10 == 0:
            t = step * DT
            error = abs(
                position_expectation(grid.x, psi) - coherent_state_position(t, 3.0)
            )
            max_error = max(max_error, error)
        psi = propagator.step(psi)
    assert max_error < 0.02


def test_free_gaussian_spreading_matches_analytic():
    grid = Grid(n_points=1200, length=60.0)
    sigma0 = 1.0
    psi = grid.gaussian_packet(x0=0.0, k0=0.0, sigma=sigma0)
    propagator = CrankNicolson(grid.x, np.zeros_like(grid.x), DT)
    for t_check in (2.0, 4.0):
        steps = int(t_check / DT)
        psi_t = propagator.step(psi.copy(), n_steps=steps)
        measured = width(grid.x, psi_t)
        # RMS width = sigma(t)/sqrt(2) for the exponent parameter sigma
        exact = free_gaussian_width(t_check, sigma0) / np.sqrt(2)
        assert measured == pytest.approx(exact, rel=0.02)


# ------------------------------------------------------- tunneling


def _measure_transmission(k0: float, sigma: float = 1.5,
                          height: float = 2.0, barrier_width: float = 1.0):
    grid = Grid(n_points=2400, length=60.0)
    potential = grid.square_barrier(height, barrier_width)
    psi = grid.gaussian_packet(x0=-8.0, k0=k0, sigma=sigma)
    propagator = CrankNicolson(grid.x, potential, DT)
    travel = 14.0 / k0 + 2.0
    psi = propagator.step(psi, n_steps=int(travel / DT))
    return transmitted_fraction(grid.x, psi, barrier_width / 2)


@pytest.mark.parametrize("k0", [1.6, 2.8, 5.0])
def test_tunneling_matches_convolved_formula(k0):
    height, barrier_width, sigma = 2.0, 1.0, 1.5
    measured = _measure_transmission(k0)
    k_grid = np.linspace(max(0.1, k0 - 3 / sigma), k0 + 3 / sigma, 400)
    expected = packet_average_transmission(
        k_grid, k0, sigma, height, barrier_width
    )
    assert measured == pytest.approx(expected, abs=0.03)


def test_plane_wave_formula_limits():
    # E >> V0: transmission -> 1; E -> 0: transmission -> 0.
    assert barrier_transmission(50.0, 2.0, 1.0) > 0.95
    assert barrier_transmission(0.05, 2.0, 1.0) < 0.01


def test_high_energy_packet_passes_through():
    assert _measure_transmission(k0=7.0) > 0.9
