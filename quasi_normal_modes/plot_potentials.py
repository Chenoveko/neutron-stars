# =====================
# Imports
# =====================
from numpy import ndarray, log, geomspace
import matplotlib.pyplot as plt


def axial_potential(r: ndarray, l: int, M: float) -> ndarray:
    """
    Axial potential in Regge-Wheeler equation in GEO units
    :param r: turtle coordinate
    :param l: spherical harmonic index
    :param M: total mass
    :return: potential
    """
    return (1.0 - 2.0 * M / r) * (l * (l + 1) / r ** 2 - 6.0 * M / r ** 3)


def polar_potential(r: ndarray, l: int, M: float) -> ndarray:
    """
    Polar potential in Zerilli equation in GEO units
    :param r: turtle coordinate
    :param l: spherical harmonic index
    :param M: total mass
    :return: potential
    """
    return (
            (1.0 - 2.0 * M / r)
            * (
                    ((l - 1) * (l + 2) / 3.0)
                    * (
                            1.0 / r ** 2
                            + 2.0 * (l - 1) * (l + 2) * (l ** 2 + l + 1)
                            / (6.0 * M + r * (l - 1) * (l + 2)) ** 2
                    )
                    + 2.0 * M / r ** 3
            )
    )


def turtle_coordinate(r: ndarray, M: float) -> ndarray:
    """
    Turtle coordinate in GEO units
    :param r: radial coordinate
    :param M: total mass
    :return: turtle coordinate
    """
    return r + 2.0 * M * log(r / (2.0 * M) - 1.0)


# ==========Parameters==========#
M = 1
l = 2
n = 10000000
r = geomspace(2.0 * M * (1.0 + 1e-10), 100 * M, n)
r_star = turtle_coordinate(r, M)

# ==========Compute potentials==========#
v_polar = polar_potential(r, l, M)
v_axial = axial_potential(r, l, M)

# ==========Plot 1: Potentials==========#
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(r_star, v_polar, color='goldenrod', linewidth=1.5, label="Polar potential")
ax.plot(r_star, v_axial, color='blue', linewidth=1.5, label="Axial potential")
ax.set_xlabel(r"$r_*$")
ax.set_ylabel(r"$V(r_*)$")
ax.set_title(fr"Potentials for $l={l}$ and $M={M}$")
ax.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax.legend(loc="upper right")

# ==========Plot 2: Zoom Potentials==========#
fig_zoom, ax_zoom = plt.subplots(figsize=(7.5, 4.5))
ax_zoom.plot(r_star, v_polar, color='goldenrod', linewidth=1.5, label="Polar potential")
ax_zoom.plot(r_star, v_axial, color='blue', linewidth=1.5, label="Axial potential")
ax_zoom.set_xlabel(r"$r_*$")
ax_zoom.set_ylabel(r"$V(r_*)$")
ax_zoom.set_title(fr"Potentials for $l={l}$ and $M={M}$")
ax_zoom.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
ax_zoom.legend(loc="upper right")
ax_zoom.set_xlim(xmin=-20, xmax=42)

plt.show()
