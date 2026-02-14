# =====================
# Imports
# =====================
import matplotlib.pyplot as plt
from numpy import linspace

from interpol_data import log10_p_apr, log10_p_gnh3, log10_p_sly4
from interpol_data import log10_rho_apr, log10_rho_gnh3, log10_rho_sly4
from interpol_data import rho_pchip_log10_apr, rho_pchip_log10_gnh3, rho_pchip_log10_sly4

"""
Values of pressure to plot PCHIP functions
-----------------------------------------------------------------------------------------
"""
log10_p_plot = linspace(0, 50, int(1e6))
"""
Plot profile of APR4 EoS with data and PCHIP interpolation
-----------------------------------------------------------------------------------------
"""
# log10(energy density) vs log10(pressure)
fig_apr, ax_apr = plt.subplots(figsize=(7.5, 4.5))
ax_apr.plot(log10_p_apr, log10_rho_apr, color='blue', linewidth=1.5, label="Data")
ax_apr.plot(log10_p_plot, rho_pchip_log10_apr(log10_p_plot), color='red', linestyle='--', linewidth=1.5,
            label="PCHIP interpol")
ax_apr.set_xlabel(r'$\log_{10}(p)$')
ax_apr.set_ylabel(r'$\log_{10}(\rho)$')
ax_apr.set_title(r'EoS Profile APR')
ax_apr.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_apr.legend(loc="upper left")

"""
Plot profile of SLy4 EoS with data and CS interpolation
-----------------------------------------------------------------------------------------
"""

# log10(energy density) vs log10(pressure)
fig_sly4, ax_sly4 = plt.subplots(figsize=(7.5, 4.5))
ax_sly4.plot(log10_p_sly4, log10_rho_sly4, color='blue', linewidth=1.5, label="Data")
ax_sly4.plot(log10_p_plot, rho_pchip_log10_sly4(log10_p_plot), color='red', linestyle='--', linewidth=1.5,
             label="PCHIP interpol")
ax_sly4.set_xlabel(r'$\log_{10}(p)$')
ax_sly4.set_ylabel(r'$\log_{10}(\rho)$')
ax_sly4.set_title(r'EoS Profile SLY4 ')
ax_sly4.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_sly4.legend(loc="upper left")

"""
Plot profile of GNH3 EoS with data and CS interpolation
-----------------------------------------------------------------------------------------
"""

# log10(energy density) vs log10(pressure)
fig_gnh3, ax_gnh3 = plt.subplots(figsize=(7.5, 4.5))
ax_gnh3.plot(log10_p_gnh3, log10_rho_gnh3, color='blue', linewidth=1.5, label="Data")
ax_gnh3.plot(log10_p_plot, rho_pchip_log10_gnh3(log10_p_plot), color='red', linestyle='--', linewidth=1.5,
             label="PCHIP interpol")
ax_gnh3.set_xlabel(r'$\log_{10}(p)$')
ax_gnh3.set_ylabel(r'$\log_{10}(\rho)$')
ax_gnh3.set_title(r'EoS Profile GNH3 ')
ax_gnh3.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_gnh3.legend(loc="upper left")

plt.show()
