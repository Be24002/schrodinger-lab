"""Harmonic oscillator: numerical spectrum and wavefunctions vs exact.

Produces figures/harmonic_oscillator.png and results/ho_spectrum.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_hermite

from sqm.analytic import harmonic_levels
from sqm.grid import Grid
from sqm.stationary import bound_states

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

grid = Grid(n_points=1200, length=14.0)
potential = grid.harmonic()
print("Diagonalizing the finite-difference Hamiltonian ...")
values, vectors = bound_states(grid.x, potential, n_levels=10)
exact = harmonic_levels(10)
deviation = np.abs(values - exact)
for n, (e_num, e_ex) in enumerate(zip(values, exact)):
    print(f"  n = {n}: E = {e_num:.6f}   exact (n+1/2) = {e_ex:.6f}   "
          f"|diff| = {abs(e_num - e_ex):.2e}")
print(f"max relative deviation over 10 levels: {deviation.max() / exact.max():.2e}")

# ---- Figure: spectrum + wavefunctions ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

axes[0].plot(range(10), values, "o", ms=8, label="numerical (finite differences)")
axes[0].plot(range(10), exact, "x", ms=9, label=r"exact $(n+\frac{1}{2})\hbar\omega$")
axes[0].set_xlabel("level $n$")
axes[0].set_ylabel("energy ($\\hbar\\omega$)")
axes[0].set_title("Harmonic oscillator spectrum")
axes[0].legend()
axes[0].grid(alpha=0.3)

for n in range(5):
    axes[1].plot(grid.x, vectors[:, n], label=f"$\\psi_{n}$")
    # exact: Hermite-Gaussian
    from math import factorial

    normalization = 1.0 / np.sqrt(2**n * factorial(n)) * np.pi ** (-0.25)
    exact_wf = normalization * eval_hermite(n, grid.x) * np.exp(-grid.x**2 / 2)
    axes[1].plot(grid.x, exact_wf, "k:", lw=0.8)
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$\\psi_n(x)$")
axes[1].set_title("Eigenfunctions: numerical (colored) vs exact Hermite-Gaussians (dotted)")
axes[1].legend(ncol=2, fontsize=9)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURES / "harmonic_oscillator.png", dpi=150)
plt.close(fig)

np.savetxt(
    RESULTS / "ho_spectrum.csv",
    np.column_stack([np.arange(10), values, exact]),
    delimiter=",",
    header="n,energy_numerical,energy_exact",
    comments="",
)
print("Wrote figures/harmonic_oscillator.png, results/ho_spectrum.csv")
