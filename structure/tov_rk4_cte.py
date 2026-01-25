# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import linspace
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
R = 10e5  # 1 Km = 10^5 cm
r = linspace(0, R, 10)

# Analytical solution function (parameters in geometrized units)
analytical_pressure_profile = schwarzschild_solution(r, M, R)
p = analytical_pressure_profile[0]
m = analytical_pressure_profile[1] / M_sun_geo  # Convert to M/M_sun
p = pressure_geo_to_cgs(p)  # Convert to CGS system
r = r / 1e5  # Convert to Km

# Plot of interior pressure vs radius
fig_pressure, ax_pressure = plt.subplots(figsize=(7.5, 4.5))
ax_pressure.plot(r, p, color='blue', linewidth=2.0, label="Analytical solution")
ax_pressure.set_xlabel(r'$r\ (km)$')
ax_pressure.set_ylabel(r'$p\ (dyn\,cm^{-2})$')
ax_pressure.set_title(r'Interior pressure')
ax_pressure.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_pressure.legend()
plt.show()

# Plot of enclosed mass vs radius
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r, m, color='blue', linewidth=2.0, label="Analytical solution")
ax_mass.set_xlabel(r'$r\ (km)$')
ax_mass.set_ylabel(r'$M / M_{\odot}$')
ax_mass.set_title(r'Enclosed Mass')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_mass.legend()
plt.show()

"""
Numerical solution of TOV equations for a spherically symmetric,
uniform-density relativistic star using the RK4 method.
-----------------------------------------
Typical neutron-star parameters:
M = 1.5 M_sun
R = 10 Km
"""
