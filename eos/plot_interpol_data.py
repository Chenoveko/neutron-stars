# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import linspace,log10
from interpol_data import log10_p_akmalpr, log10_p_glendnh3, log10_p_sly4
from interpol_data import log10_rho_akmalpr, log10_rho_glendnh3, log10_rho_sly4
from interpol_data import rho_cs_log10_akmalpr, rho_cs_log10_glendnh3, rho_cs_log10_sly4
"""
Values of pressure to plot CS functions
-----------------------------------------------------------------------------------------
"""
p_plot = linspace(1e15, 1e20, int(1e7))
log10_p_plot = log10(p_plot)

"""
Plot profile of APR4 EoS with data and CS interpolation
-----------------------------------------------------------------------------------------
"""
# log10(energy density) vs log10(pressure)
fig_akmalpr, ax_akmalpr = plt.subplots(figsize=(7.5, 4.5))
ax_akmalpr.plot(log10_p_akmalpr, log10_rho_akmalpr, color='blue', linewidth=1.5, label="Data")
ax_akmalpr.plot(log10_p_plot, rho_cs_log10_akmalpr(log10_p_plot), color='red', linestyle='--', linewidth=1.5,
                label="CS interpol")
ax_akmalpr.set_xlabel(r'$\log_{10}(p)$')
ax_akmalpr.set_ylabel(r'$\log_{10}(\rho)$')
ax_akmalpr.set_title(r'EoS Profile Zoom APR4 ')
ax_akmalpr.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_akmalpr.legend(loc="upper left")

"""
Plot profile of SLy4 EoS with data and CS interpolation
-----------------------------------------------------------------------------------------
"""

# log10(energy density) vs log10(pressure)
fig_sly4, ax_sly4 = plt.subplots(figsize=(7.5, 4.5))
ax_sly4.plot(log10_p_sly4, log10_rho_sly4, color='blue', linewidth=1.5, label="Data")
ax_sly4.plot(log10_p_plot, rho_cs_log10_sly4(log10_p_plot), color='red', linestyle='--', linewidth=1.5,
             label="CS interpol")
ax_sly4.set_xlabel(r'$\log_{10}(p)$')
ax_sly4.set_ylabel(r'$\log_{10}(\rho)$')
ax_sly4.set_title(r'EoS Profile Zoom SLY4 ')
ax_sly4.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_sly4.legend(loc="upper left")

"""
Plot profile of GNH3 EoS with data and CS interpolation
-----------------------------------------------------------------------------------------
"""

# log10(energy density) vs log10(pressure)
fig_glendnh3, ax_glendnh3 = plt.subplots(figsize=(7.5, 4.5))
ax_glendnh3.plot(log10_p_glendnh3, log10_rho_glendnh3, color='blue', linewidth=1.5, label="Data")
ax_glendnh3.plot(log10_p_plot, rho_cs_log10_glendnh3(log10_p_plot), color='red', linestyle='--', linewidth=1.5,
                 label="CS interpol")
ax_glendnh3.set_xlabel(r'$\log_{10}(p)$')
ax_glendnh3.set_ylabel(r'$\log_{10}(\rho)$')
ax_glendnh3.set_title(r'EoS Profile Zoom GNH3 ')
ax_glendnh3.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_glendnh3.legend(loc="upper left")

plt.show()
