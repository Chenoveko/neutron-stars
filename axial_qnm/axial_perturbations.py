# =====================
# Imports
# =====================
from numpy import load,array
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import subprocess

# =====================
# Multiprocessing
# =====================
def run(script):
    subprocess.run(["python", "-m", script], check=True)

if __name__ == "__main__":

    scripts = [
        "axial_qnm.axial_perturbations_gnh3",
        "axial_qnm.axial_perturbations_sly4",
        "axial_qnm.axial_perturbations_apr",
    ]

    with ProcessPoolExecutor(max_workers=10) as ex:
        list(ex.map(run, scripts))

    gnh3 = load("gnh3_results.npz")
    sly4 = load("sly4_results.npz")
    apr = load("apr_results.npz")

    # =====================
    # Ploting
    # =====================
    # ==========Plot 1: Frequency vs Central Pressure==========#
    fig_f, ax_f = plt.subplots(figsize=(7.5, 4.5))
    ax_f.plot(apr["log_p"], apr["f"], color="goldenrod", linewidth=1.5, label="APR")
    ax_f.plot(gnh3["log_p"], gnh3["f"], color="blue", linewidth=1.5, label="GNH3")
    ax_f.plot(sly4["log_p"], sly4["f"], color="red", linewidth=1.5, label="SLy4")
    ax_f.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
    ax_f.set_ylabel(r'$\nu$ [KHz]')
    #ax_f.set_title(r'Frequency')
    ax_f.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
    ax_f.legend(loc="upper right")
    fig_f.savefig("frequency.png", dpi=600, bbox_inches="tight")

    # ==========Plot 2: Damping time vs Central Pressure==========#
    fig_tau, ax_tau = plt.subplots(figsize=(7.5, 4.5))
    ax_tau.plot(apr["log_p"], apr["tau"], color="goldenrod", linewidth=1.5, label="APR")
    ax_tau.plot(gnh3["log_p"], gnh3["tau"], color="blue", linewidth=1.5, label="GNH3")
    ax_tau.plot(sly4["log_p"], sly4["tau"], color="red", linewidth=1.5, label="SLy4")
    ax_tau.set_xlabel(r'$\log_{10}(p_c)\, [\mathrm{dyn\,cm^{-2}}]$')
    ax_tau.set_ylabel(r'$\tau$ [$\mu$s]')
    #ax_tau.set_title(r'Damping time SLy4')
    ax_tau.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
    ax_tau.legend(loc="upper left")
    fig_tau.savefig("damping_time.png", dpi=600, bbox_inches="tight")
    plt.show()