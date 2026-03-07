# =====================
# Imports
# =====================

import matplotlib.pyplot as plt
from utilities.physical_functions import  p_degenerate_fermions_ur, p_degenerate_fermions_nr, eos_polytropic
from utilities.physical_data import c
from numpy import logspace, log10

# ==========Parameters==========#
rho_energy_cgs = logspace(20, 35, 200)
rho_mass_cgs = rho_energy_cgs / c**2
log10_rho_energy_cgs = log10(rho_energy_cgs)

# ==========UR Fermions==========#
p_ur_cgs = p_degenerate_fermions_ur(rho_mass_cgs)
log10_p_ur_cgs = log10(p_ur_cgs)

# ==========NR Fermions==========#
p_nr_cgs = p_degenerate_fermions_nr(rho_mass_cgs)
log10_p_nr_cgs = log10(p_nr_cgs)

# ==========PT UR Fermions==========#
p_pt_ur_cgs = eos_polytropic(rho_mass_cgs)
log10_p_pt_ur_cgs = log10(p_pt_ur_cgs)

# ==========PT NR Fermions==========#
p_pt_nr_cgs = eos_polytropic(rho_mass_cgs, ur=False)
log10_p_pt_nr_cgs = log10(p_pt_nr_cgs)

# ==========Plot: log10(pressure) vs log10(energy density)==========#
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(log10_rho_energy_cgs, log10_p_ur_cgs, color='goldenrod', linewidth=1.5, label="UR")
ax.plot(log10_rho_energy_cgs, log10_p_nr_cgs, color='blue', linewidth=1.5, label="NR")
#ax.plot(log10_rho_energy_cgs, log10_p_pt_ur_cgs, color='green', linewidth=1.5, label="PT UR")
#ax.plot(log10_rho_energy_cgs, log10_p_pt_nr_cgs, color='red', linewidth=1.5, label="PT NR")
ax.set_xlabel(r'$\log_{10}([\rho(erg/cm^3)])$')
ax.set_ylabel(r'$\log_{10}[p(dyn/cm^2)]$')
ax.set_title(r'EoS Fermi')
ax.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax.legend(loc="upper left")
plt.show()