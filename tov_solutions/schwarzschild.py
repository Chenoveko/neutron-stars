"""
TOV Equation Solver — Analytical vs Numerical Schwarzschild Interior Solution
===============================================================================

This script compares the analytical and numerical solutions of the
Tolman–Oppenheimer–Volkoff (TOV) equations for a spherically symmetric,
uniform-density relativistic star (Schwarzschild interior solution).

A canonical neutron star configuration is considered:

    M = 1.4 M_sun
    R = 10 km

Two approaches are implemented:

1. Analytical solution
   The exact Schwarzschild interior solution for a constant-density star
   is evaluated, providing radial profiles of pressure p(r) and enclosed
   mass m(r).

2. Numerical solution
   The TOV equations are integrated numerically using solve_ivp through
   the function solve_tov_schwarzschild, assuming the same total mass
   and radius.

The purpose of this script is to validate the numerical TOV solver by
direct comparison with the exact analytical solution.

Results are presented in two plots:
    1. Interior pressure  p(r)  vs  radius  r (km)
    2. Enclosed mass  m(r)/M_sun  vs  radius  r (km)

Both analytical and numerical profiles are shown for direct comparison.

Units
-----
    - Mass is defined in geometrized units (G = c = 1).
    - Radii are defined in CGS (cm) and converted to kilometres (km).
    - Pressures are converted from geometrized units to CGS (dyn cm^-2).
    - Enclosed mass is expressed in units of the solar mass M_sun.

Dependencies
------------
    - utilities.math_methods        : numerical TOV solver (solve_tov_schwarzschild)
    - utilities.physical_functions  : analytical Schwarzschild solution
    - utilities.physical_data       : unit conversion constants
    - matplotlib                    : plotting
    - numpy                         : radial discretization

Physical Background
-------------------
For a uniform-density star, the Schwarzschild interior solution provides
an exact relativistic solution to Einstein’s field equations. The pressure
profile is finite at the centre and vanishes at the surface (r = R), while
the enclosed mass increases monotonically from zero to the total mass M.

The numerical integration reproduces the same structure by solving the
TOV equations directly, serving as a consistency check of the solver.
"""
# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import linspace

from utilities.tov_solvers import solve_tov_schwarzschild
from utilities.physical_data import mass_cgs_to_geo, pressure_geo_to_cgs, M_sun
from utilities.physical_functions import schwarzschild_solution

# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)
M = 1.4 * M_sun_geo
discretization_points = 10000
R_cgs = 10e5  # 1 Km = 10^5 cm
r_cgs_analytical = linspace(0, R_cgs, discretization_points)

# ========== Analytical solution==========#
analytical_profile = schwarzschild_solution(r_cgs_analytical, M, R_cgs)
p_analytical = analytical_profile[0]
m_analytical = analytical_profile[1] / M_sun_geo  # Convert to M/M_sun
p_analytical = pressure_geo_to_cgs(p_analytical)  # Convert to CGS system
r_km_analytical = r_cgs_analytical / 1e5  # Convert to Km

# Plot of interior pressure vs radius
fig_pressure, ax_pressure = plt.subplots(figsize=(7.5, 4.5))
ax_pressure.plot(r_km_analytical, p_analytical, color='blue', linewidth=1.5, label="Analytical solution")
ax_pressure.set_xlabel(r'$r\ [km]$')
ax_pressure.set_ylabel(r'$p\ [dyn\,cm^{-2}]$')
ax_pressure.set_title(r'Interior pressure')
ax_pressure.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)

# Plot of enclosed mass vs radius
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r_km_analytical, m_analytical, color='blue', linewidth=1.5, label="Analytical solution")
ax_mass.set_xlabel(r'$r\ [km]$')
ax_mass.set_ylabel(r'$M / M_{\odot}$')
ax_mass.set_title(r'Enclosed Mass')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)

# ==========Integration==========#
numerical_profile = solve_tov_schwarzschild(M, R_cgs)
r_cgs_numerical = numerical_profile[0]
p_numerical = numerical_profile[1]
m_numerical = numerical_profile[2] / M_sun_geo  # Convert to M/M_sun
p_numerical = pressure_geo_to_cgs(p_numerical)  # Convert to CGS system
r_km_numerical = r_cgs_numerical / 1e5  # Convert to Km

# ==========Plot 1: Interior Pressure vs Radius==========#
fig_pressure, ax_pressure = plt.subplots(figsize=(7.5, 4.5))
ax_pressure.plot(r_km_analytical, p_analytical, color='blue', linewidth=1.5, label="Analytical solution")
ax_pressure.plot(r_km_numerical, p_numerical, color='red', linestyle='--', linewidth=2.0, label="Numerical solution")
ax_pressure.set_xlabel(r'$r\ [km]$')
ax_pressure.set_ylabel(r'$p\ [dyn\,cm^{-2}]$')
ax_pressure.set_title(r'Interior pressure')
ax_pressure.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_pressure.legend(loc="upper right")
# fig_pressure.savefig("schwarzschild_pressure.png", dpi=600, bbox_inches="tight")

# ==========Plot 1: Enclosed Mass vs Radius==========#
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r_km_analytical, m_analytical, color='blue', linewidth=1.5, label="Analytical solution")
ax_mass.plot(r_km_numerical, m_numerical, color='red', linestyle='--', linewidth=2.0, label="Numerical solution")
ax_mass.set_xlabel(r'$r\ [km]$')
ax_mass.set_ylabel(r'$M / M_{\odot}$')
ax_mass.set_title(r'Enclosed Mass')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_mass.legend(loc="upper left")
# fig_mass.savefig("schwarzschild_mass.png", dpi=600, bbox_inches="tight")
plt.show()
