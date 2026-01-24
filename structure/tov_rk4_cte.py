# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from functions import tov_equations, schwarzschild_solution
from constants_units import mass_cgs_to_geo, pressure_geo_to_cgs, M_sun
from numpy import linspace

"""
Analytical solution to TOV equations spherically symmetric relativistic stars with uniform-density.
-----------------------------------------
Typical neutron star dimensions
M = 1.5 M_sun
R = 10 Km
"""

# Parameters in geometrized units
M_sun_geo = mass_cgs_to_geo(M_sun)
M = 1.5*M_sun_geo
R = 1e6
r = linspace(0, R, 10)

# Analytical solution function (parameters in geometrized units)
analytical_pressure_profile = schwarzschild_solution(r,M,R)
p = analytical_pressure_profile[0]
m = analytical_pressure_profile[1]/M_sun_geo # Convert to M/M_sun
p = pressure_geo_to_cgs(p) # Convert to CGS system
r = r /1e6 # Convert to Km

# Plot of interior pressure vs radius
fig_pressure, ax_pressure = plt.subplots(figsize=(7.5, 4.5))
ax_pressure.plot(r, p, color = 'blue', linewidth = 2.0, label="Analytical solution")
ax_pressure.set_xlabel(r'$r\ \mathrm{(Km)}$')
ax_pressure.set_ylabel(r'$p\ \mathrm{(dyn\,cm^{-2})}$')
ax_pressure.set_title(r'Interior pressure')
ax_pressure.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_pressure.legend()
plt.figure(fig_pressure)
plt.show()

# Plot of interior mass vs radius
fig_mass, ax_mass = plt.subplots(figsize=(7.5, 4.5))
ax_mass.plot(r,m,color = 'blue', linewidth = 2.0, label="Analytical solution")
ax_mass.set_xlabel(r'$r\ \mathrm{(Km)}$')
ax_mass.set_ylabel(r'$M / M_{\odot}$')
ax_mass.set_title(r'Interior mass')
ax_mass.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_mass.legend()
plt.figure(fig_mass)
plt.show()

"""
Numerical solution of the TOV equations for a spherically symmetric,
uniform-density relativistic star using the RK4 method.
-----------------------------------------
Typical neutron-star parameters:
M = 1.5 M_sun
R = 10 Km
"""



