"""A coherent state oscillating in a harmonic trap (the quantum version of
a classical pendulum), plus the Ehrenfest check.

Produces figures/coherent_state.gif and figures/ehrenfest.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sqm.dynamics import position_expectation
from sqm.grid import Grid
from sqm.solvers import CrankNicolson

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

DT = 0.005
X0 = 3.0

grid = Grid(n_points=800, length=20.0)
potential = grid.harmonic()
psi = grid.coherent_state(x0=X0)
propagator = CrankNicolson(grid.x, potential, DT)

period = 2 * np.pi
n_frames = 60
steps_per_frame = int(period / DT / n_frames)

times, positions = [], []
frame_data = []
for frame in range(n_frames + 1):
    times.append(frame * steps_per_frame * DT)
    positions.append(position_expectation(grid.x, psi))
    frame_data.append(np.abs(psi) ** 2)
    psi = propagator.step(psi, n_steps=steps_per_frame)

times = np.array(times)
positions = np.array(positions)
classical = X0 * np.cos(times)
max_error = np.abs(positions - classical).max()
print(f"Ehrenfest check over one full period:")
print(f"  max |<x>(t) - x0 cos(t)| = {max_error:.2e}")
print(f"  (the coherent state follows the classical oscillator)")

# ---- Ehrenfest figure -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.plot(times, positions, "o-", ms=4, label=r"quantum $\langle x \rangle(t)$")
ax.plot(times, classical, "k--", label=r"classical $x_0\cos t$")
ax.set_xlabel("time $t$ ($1/\\omega$)")
ax.set_ylabel("$\\langle x \\rangle$")
ax.set_title(f"Coherent state obeys Ehrenfest (max error {max_error:.1e})")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "ehrenfest.png", dpi=150)
plt.close(fig)

# ---- GIF -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.6))
line, = ax.plot([], [], color="C3")
ax.set_xlim(grid.x.min(), grid.x.max())
ax.set_ylim(0, 0.55)
ax.set_xlabel("$x$")
ax.set_ylabel("$|\\psi|^2$")
title = ax.set_title("")
ax.grid(alpha=0.3)
density_scale = frame_data[0].max()
ax.plot(grid.x, potential / potential.max() * 0.4, alpha=0.3, color="C0")

from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


def update(frame: int):
    line.set_data(grid.x, frame_data[frame])
    title.set_text(
        f"coherent state, t = {times[frame]:.2f}   "
        f"$\\langle x \\rangle$ = {positions[frame]:+.2f}"
    )
    return line, title


animation = FuncAnimation(
    fig, update, frames=n_frames + 1, interval=50, blit=True
)
animation.save(
    FIGURES / "coherent_state.gif",
    writer=PillowWriter(fps=20),
    dpi=110,
)
plt.close(fig)
print("Wrote figures/coherent_state.gif, figures/ehrenfest.png")
