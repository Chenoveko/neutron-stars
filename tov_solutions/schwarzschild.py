# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import linspace

from utilities.math_methods import solve_tov_schwarzschild
from utilities.physical_data import mass_cgs_to_geo, pressure_geo_to_cgs, M_sun
from utilities.physical_functions import schwarzschild_solution

"""
Analytical solution of TOV equations spherically symmetric relativistic stars with uniform-density.
-----------------------------------------
Typical neutron star dimensions
M = 1.5 M_sun
R = 10 Km
"""

# Parameters in geometrized units
M_sun_geo = mass_cgs_to_geo(M_sun)
M = 1.5 * M_sun_geo
discretization_points = 10000
R_cgs = 10e5  # 1 Km = 10^5 cm
r_cgs_analytical = linspace(0, R_cgs, discretization_points)

# Analytical solution function (parameters in geometrized units)
analytical_profile = schwarzschild_solution(r_cgs_analytical, M, R_cgs)
p_analytical = analytical_profile[0]
m_analytical = analytical_profile[1] / M_sun_geo  # Convert to M/M_sun
p_analytical = pressure_geo_to_cgs(p_analytical)  # Convert to CGS system
r_km_analytical = r_cgs_analytical / 1e5  # Convert to Km

# Plot of interior pressure vs radius
fig_pressure, ax_pressure = plt.subplots(figsize=(7.5, 4.5))
ax_pressure.plot(r_km_analytical, p_analytical, color='blue', linewidth=2.0, label="Analytical solution")
ax_pressure.set_xlabel(r'$r\ (km)$')
ax_pressure.set_ylabel(r'$p\ (dyn\,cm^{-2})$')
ax_pressure.set_title(r'Interior pressure')
ax_pressure.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)

# Plot of enclosed mass vs radius
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r_km_analytical, m_analytical, color='blue', linewidth=2.0, label="Analytical solution")
ax_mass.set_xlabel(r'$r\ (km)$')
ax_mass.set_ylabel(r'$M / M_{\odot}$')
ax_mass.set_title(r'Enclosed Mass')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)

"""
Numerical solution of TOV equations for a spherically symmetric,
uniform-density relativistic star using solve_ivp from SciPy.
-----------------------------------------
same parameters as before
"""

# Numerical solution (parameters in geometrized units)
numerical_profile = solve_tov_schwarzschild(M, R_cgs)
r_cgs_numerical = numerical_profile[0]
p_numerical = numerical_profile[1]
m_numerical = numerical_profile[2] / M_sun_geo  # Convert to M/M_sun
p_numerical = pressure_geo_to_cgs(p_numerical)  # Convert to CGS system
r_km_numerical = r_cgs_numerical / 1e5  # Convert to Km

# Plot of interior pressure vs radius
ax_pressure.plot(r_km_numerical, p_numerical, color='red', linestyle='--', linewidth=2.0, label="Numerical solution")
ax_pressure.legend(loc="upper right")

# Plot of enclosed mass vs radius
ax_mass.plot(r_km_numerical, m_numerical, color='red', linestyle='--', linewidth=2.0, label="Numerical solution")
ax_mass.legend(loc="upper left")

plt.show()
