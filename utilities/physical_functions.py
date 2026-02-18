# =====================
# Imports
# =====================
from numpy import pi, sqrt, array, ndarray, maximum, isfinite, nan
from scipy.constants import m_n, hbar, c, m_u
from utilities.physical_data import pressure_geo_to_cgs, energy_density_cgs_to_geo
"""
Tolman–Oppenheimer–Volkoff (TOV) Equations.
-----------------------------------------
Equations of structure for spherically symmetric relativistic stars.

From:
    Gravity: An Introduction to Einstein's General Relativity
    James B. Hartle

Units:
    Geometrized units in CGS
"""


def tov_equations(var: ndarray, r: float, rho: float) -> ndarray:
    """
    TOV equations in GEO units
    :param var: state variables
    :param r: radial coordinate
    :param rho: energy density
    :return: derivatives
    """
    # Unpack variables
    p, m = var
    # Interior pressure equation
    dp_dr = - (rho + p) * (m + 4 * pi * r ** 3 * p) / (r * (r - 2 * m))
    # Enclosed mass equation
    dm_dr = 4.0 * pi * r ** 2 * rho
    return array([dp_dr, dm_dr], float)


"""
Tolman–Oppenheimer–Volkoff (TOV) Equations, Schwarzschild interior solution.
-----------------------------------------
Interior solution for spherically symmetric relativistic stars with uniform-density.

From:
    A first course in general relativity
    Bernard Schutz

Units:
    Geometrized units in CGS
"""


def schwarzschild_solution(r: ndarray, M: float, R: float) -> ndarray:
    """
    :param r: radial coordinate
    :param M: total mass
    :param R: total radius
    :return: pressure and enclosed mass
    """
    if M / R >= 4 / 9:
        raise ValueError("Schwarzschild interior solution requires M/R < 4/9")

    # Uniform energy density
    rho = 3 * M / (4 * pi * R ** 3)
    # Interior pressure
    p = rho * (
            (sqrt(1 - 2 * M * r ** 2 / R ** 3) - sqrt(1 - 2 * M / R)) /
            (3 * sqrt(1 - 2 * M / R) - sqrt(1 - 2 * M * r ** 2 / R ** 3))
    )
    # Enclosed mass
    m = (4.0 / 3.0) * pi * rho * r ** 3
    return array([p, m], float)


"""
Equation of State for Non-Relativistic Degenerate Fermions
-----------------------------------------
"""


def p_degenerate_fermions_nr(rho: ndarray, m_f: float = m_n, Z: int = 8, A: int = 16) -> ndarray:
    """
    Star primarly composed of 12C and 16O -> A/Z = 2
    :param rho: mass density in CGS
    :param m_f: fermion mass in SI
    :param Z: Atomic number
    :param A: Mass number
    :return: pressure
    """
    # Convert mass to CGS
    m_f = m_f * 1000
    m_N = m_u * 1000
    # Convert hbar to CGS
    hbar_cgs = hbar * 1e7
    # Compute k_nr
    k_nr = (hbar_cgs ** 2 / (15 * pi ** 2 * m_f)) * (3 * pi ** 2 * Z / (m_N * A)) ** (5 / 3)
    return k_nr * rho ** (5 / 3)

def rho_degenerate_fermions_nr(p: ndarray, m_f: float = m_n, Z: int = 8, A: int = 16) -> ndarray:
    """
    Star primarly composed of 12C and 16O -> A/Z = 2
    :param rho: pressure in CGS
    :param m_f: fermion mass in SI
    :param Z: Atomic number
    :param A: Mass number
    :return: energy density in CGS
    """
    # Convert c to CGS
    c_cgs = c * 100
    # Convert mass to CGS
    m_f = m_f * 1000
    m_N = m_u * 1000
    # Convert hbar to CGS
    hbar_cgs = hbar * 1e7
    # Compute k
    k = (hbar_cgs ** 2 / (15 * pi ** 2 * m_f)) * (3 * pi ** 2 * Z / (m_N * A)) ** (5 / 3)
    # Mass density
    p = maximum(p, 0.0)
    rho_m = (p / k) ** (3 / 5)
    # Energy density
    epsilon = rho_m * c_cgs ** 2
    return epsilon

def rho_degenerate_fermions_nr_geo(p_geo):
    p_cgs = pressure_geo_to_cgs(p_geo)
    if (not isfinite(p_cgs)) or (p_cgs <= 0):
        return nan
    eps_cgs = rho_degenerate_fermions_nr(p_cgs)   # ya es energía/volumen (erg/cm^3)
    return energy_density_cgs_to_geo(eps_cgs)

"""
Equation of State for Ultra-Relativistic Degenerate Fermions
"""

def p_degenerate_fermions_ur(rho: ndarray, Z: int = 8, A: int = 16) -> ndarray:
    """
    Star primarly composed of 12C and 16O -> A/Z = 2
    :param rho: mass density in CGS
    :param Z: Atomic number
    :param A: Mass number
    :return: pressure in CGS
    """
    # Convert c to CGS
    c_cgs = c * 100
    # Convert mass to CGS
    m_N = m_u * 1000
    # Convert hbar to CGS
    hbar_cgs = hbar * 1e7
    # Polytropic value
    k = (hbar_cgs * c_cgs / (12 * pi ** 2)) * (3 * pi ** 2 * Z / (m_N * A)) ** (4 / 3)
    return k * rho**(4.0/3.0)

def rho_degenerate_fermions_ur(p: float, Z: int = 8, A: int = 16) -> float:
    """
    Star primarly composed of 12C and 16O -> A/Z = 2
    :param p: mass density in CGS
    :param Z: Atomic number
    :param A: Mass number
    :return: energy density in CGS
    """
    # Convert c to CGS
    c_cgs = c * 100
    # Convert mass to CGS
    m_N = m_u * 1000
    # Convert hbar to CGS
    hbar_cgs = hbar * 1e7
    # Polytropic value
    k = (hbar_cgs * c_cgs / (12 * pi ** 2)) * (3 * pi ** 2 * Z / (m_N * A)) ** (4 / 3)
    # Mass density
    rho_m = (p / k) ** (3.0 / 4.0)
    # Energy density
    epsilon = rho_m * c_cgs ** 2
    return epsilon

def rho_degenerate_fermions_ur_geo(p_geo):
    p_cgs = pressure_geo_to_cgs(p_geo)
    if (not isfinite(p_cgs)) or (p_cgs <= 0):
        return nan
    eps_cgs = rho_degenerate_fermions_ur(p_cgs)
    return energy_density_cgs_to_geo(eps_cgs)



"""
Polytropic Equation of State
-----------------------------------------
"""


def eos_polytropic(rho: ndarray, m_f: float = m_n, Z: int = 8, A: int = 16, ur: bool = True) -> ndarray:
    """
    Star primarly composed of 12C and 16O -> A/Z = 2
    :param rho: mass density in CGS
    :param Z: Atomic number
    :param A: Mass number
    :return: pressure
    """
    # Convert c to CGS
    c_cgs = c * 100
    # Convert mass to CGS
    m_f = m_f * 1000
    m_N = m_u * 1000
    # Convert hbar to CGS
    hbar_cgs = hbar * 1e7
    # NR or UR
    if ur:
        n = 3
        k = (hbar_cgs * c / (12 * pi ** 2)) * (3 * pi ** 2 * Z / (m_N * A)) ** (4 / 3)
    else:
        n = 3 / 2
        k = (hbar_cgs ** 2 / (15 * pi ** 2 * m_f)) * (3 * pi ** 2 * Z / (m_N * A)) ** (5 / 3)
    # Adiabatic exponent
    gamma = 1 + 1 / n

    return k * rho ** gamma
