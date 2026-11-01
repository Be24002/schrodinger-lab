"""Quantum tunneling: wavepacket transmission vs the exact formula.

The honest comparison: a Gaussian packet is not a plane wave, so the
measurable transmission is the plane-wave T(E) convolved with the packet's
momentum distribution -- that is what we compare against.

Produces figures/tunneling.png and results/tunneling.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sqm.analytic import barrier_transmission, packet_average_transmission
from sqm.dynamics import transmitted_fraction
from sqm.grid import Grid
from sqm.solvers import CrankNicolson

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

HEIGHT, WIDTH, SIGMA = 2.0, 1.0, 1.5
DT = 0.005


def measure_transmission(k0: float, snapshot: bool = False):
    grid = Grid(n_points=2400, length=60.0)
    potential = grid.square_barrier(HEIGHT, WIDTH)
    psi = grid.gaussian_packet(x0=-8.0, k0=k0, sigma=SIGMA)
    propagator = CrankNicolson(grid.x, potential, DT)
    travel = 14.0 / k0 + 2.0
    psi = propagator.step(psi, n_steps=int(travel / DT))
    if snapshot:
        return transmitted_fraction(grid.x, psi, WIDTH / 2), grid.x, psi, potential
    return transmitted_fraction(grid.x, psi, WIDTH / 2)


k0_values = [1.6, 2.0, 2.4, 2.8, 3.5, 5.0]
print("Wavepacket transmissions (measured vs convolved exact formula):")
rows = []
for k0 in k0_values:
    measured = measure_transmission(k0)
    k_grid = np.linspace(max(0.1, k0 - 3 / SIGMA), k0 + 3 / SIGMA, 400)
    expected = packet_average_transmission(k_grid, k0, SIGMA, HEIGHT, WIDTH)
    energy = k0**2 / 2.0
    print(
        f"  k0 = {k0:.1f} (E = {energy:5.2f}, V0 = {HEIGHT}): "
        f"T_measured = {measured:.3f}   T_expected = {expected:.3f}   "
        f"|diff| = {abs(measured - expected):.3f}"
    )
    rows.append((k0, energy, measured, expected))

# ---- Figure: T(E) curve + measured points + one snapshot -----------------
energy_grid = np.linspace(0.05, 15, 500)
curve = [barrier_transmission(e, HEIGHT, WIDTH) for e in energy_grid]

_, x_snap, psi_snap, potential_snap = measure_transmission(2.0, snapshot=True)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
axes[0].plot(energy_grid, curve, "-", color="C0",
             label="plane-wave $T(E)$ (exact)")
axes[0].plot([r[1] for r in rows], [r[2] for r in rows], "o", ms=9,
             color="C3", label="wavepacket simulation")
axes[0].axvline(HEIGHT, color="k", ls=":", lw=1, label="$E = V_0$")
axes[0].set_xlabel("energy $E$")
axes[0].set_ylabel("transmission $T$")
axes[0].set_title("Tunneling through a square barrier ($V_0 = 2$, $a = 1$)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].fill_between(x_snap, potential_snap / HEIGHT, alpha=0.2, color="C0",
                     label="barrier (scaled)")
axes[1].plot(x_snap, np.abs(psi_snap) ** 2, color="C3",
             label="$|\\psi|^2$ after scattering ($k_0 = 2$)")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("probability density")
axes[1].set_title("Split of a wavepacket at the barrier")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURES / "tunneling.png", dpi=150)
plt.close(fig)

np.savetxt(
    RESULTS / "tunneling.csv",
    np.array(rows),
    delimiter=",",
    header="k0,energy,transmission_measured,transmission_expected",
    comments="",
)
print("Wrote figures/tunneling.png, results/tunneling.csv")
