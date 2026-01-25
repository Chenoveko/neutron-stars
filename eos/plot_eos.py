# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import log10

from utilities.extract_data_eos import extract_mass_density_from_eos_txt, extract_pressure_from_eos_txt

"""
Equations of Structure (EoS)
-----------------------------------------
- SLy4 - ’Skryme Lyon’
- APR4 - A. Akmal, V. R. Pandharipande, D.G. Ravenhall
- GNH3 - N. K. Glendenning.


From:
    Akmal, A., V. R. Pandharipande y D. G. Ravenhall (1998). “Equation of state of nucleon
    matter and neutron star structure”. Physical Review C 58.3, p ags. 1804-1828

Units:
    n_B [fm^{-3}]  rho [g/cm^3]   p [dyn/cm^2]
"""
# Mass density from Eos on log10
log10_rho_akmalpr = log10(extract_mass_density_from_eos_txt("eos_akmalpr.txt"))
log10_rho_glendnh3 = log10(extract_mass_density_from_eos_txt("eos_glendnh3.txt"))
log10_rho_sly4 = log10(extract_mass_density_from_eos_txt("eos_sly4.txt"))
# Pressure from EoS on log10
log10_p_akmalpr = log10(extract_pressure_from_eos_txt("eos_akmalpr.txt"))
log10_p_glendnh3 = log10(extract_pressure_from_eos_txt("eos_glendnh3.txt"))
log10_p_sly4 = log10(extract_pressure_from_eos_txt("eos_sly4.txt"))

# Plot of EoS, log10(mass density) vs log10(pressure)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(log10_p_akmalpr, log10_rho_akmalpr, color='goldenrod', linewidth=1.5, label="APR")
ax.plot(log10_p_glendnh3, log10_rho_glendnh3, color='blue', linewidth=1.5, label="Glendenning NH3")
ax.plot(log10_p_sly4, log10_rho_sly4, color='red', linewidth=1.5, label="SLy4")
ax.set_xlabel(r'$\log_{10}(p)$')
ax.set_ylabel(r'$\log_{10}(\rho)$')
ax.set_title(r'EoS Profile')
ax.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax.legend()
plt.show()

# Zoom from previous plot
fig_zoom, ax_zoom = plt.subplots(figsize=(7.5, 4.5))
ax_zoom.plot(log10_p_akmalpr, log10_rho_akmalpr, color='goldenrod', linewidth=1.5, label="APR")
ax_zoom.plot(log10_p_glendnh3, log10_rho_glendnh3, color='blue', linewidth=1.5, label="Glendenning NH3")
ax_zoom.plot(log10_p_sly4, log10_rho_sly4, color='red', linewidth=1.5, label="SLy4")
ax_zoom.set_xlabel(r'$\log_{10}(p)$')
ax_zoom.set_ylabel(r'$\log_{10}(\rho)$')
ax_zoom.set_title(r'EoS Zoom Profile')
ax_zoom.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_zoom.legend()
ax_zoom.set_xlim(xmin=32, xmax=36.7)
ax_zoom.set_ylim(ymin=13.5, ymax=15.8)
plt.show()
