"""sqm: numerical quantum mechanics, checked against the exact formulas.

Modules
-------
grid
    Spatial grids and potentials (harmonic oscillator, infinite well,
    barriers).
solvers
    Time evolution: Crank-Nicolson (sparse, unconditionally stable and
    unitary) and the Fourier split-operator (spectral kinetic accuracy).
stationary
    Bound-state spectra by sparse diagonalization of the finite-difference
    Hamiltonian.
dynamics
    Observables: norm, position/momentum expectations, packet width.
analytic
    The exact results everything is validated against: (n+1/2) hbar omega,
    infinite-well levels, barrier transmission T(E), Gaussian spreading.
"""

__version__ = "0.1.0"
