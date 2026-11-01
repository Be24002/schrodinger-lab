"""Bound states by sparse diagonalization of the finite-difference Hamiltonian."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from .solvers import hamiltonian_fd


def bound_states(x: np.ndarray, potential: np.ndarray, n_levels: int = 10):
    """Lowest ``n_levels`` (energies, wavefunctions) of a potential on x.

    Wavefunctions are normalized on the grid; eigenvalues sorted ascending.
    """
    hamiltonian = hamiltonian_fd(x, potential)
    n = len(x)
    # shift-invert-free: SA on a well-conditioned banded matrix is fine
    k = min(n_levels, n - 2)
    values, vectors = eigsh(hamiltonian, k=k, which="SA")
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    dx = x[1] - x[0]
    for i in range(k):
        norm = np.sqrt(np.sum(np.abs(vectors[:, i]) ** 2) * dx)
        vectors[:, i] /= norm
        if vectors[np.argmax(np.abs(vectors[:, i])), i].real < 0:
            vectors[:, i] *= -1
    return values, vectors


def infinite_well_levels(half_width: float, n_levels: int = 8) -> np.ndarray:
    """Exact infinite-well spectrum from the interior Dirichlet matrix."""
    length = 2.0 * half_width
    return (np.arange(1, n_levels + 1) * np.pi / length) ** 2 / 2.0
