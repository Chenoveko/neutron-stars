# =====================
# Imports
# =====================
from typing import Callable

from numpy import ndarray, linspace, array, empty, ndenumerate, pi, sqrt, log10

from utilities.physical_functions import tov_equations
from utilities.physical_data import pressure_geo_to_cgs, energy_density_cgs_to_geo, pressure_cgs_to_geo

"""
RK4 method to solve TOV equations with uniform density (Schwarzschild solution)
-----------------------------------------------------------------------------------------
Interior solution for spherically symmetric relativistic stars with uniform-density.

From:
    Computational Physics notes, course 24/25
    Jacobo Ruiz de Elvira Carrascal

Units:
    Geometrized units in CGS
"""


def rk4_schwarzschild_solution(M: float, R: float, n: int, r0: float = 1e-8) -> ndarray:
    """
    :param r0: Initial point near center
    :param M: total mass
    :param R: total radius
    :param n: Number of points in stepe discretization of tstepe interval [r0, R].
    :return:
        - r_points: Numpy array of radial coordinate points
        - p_points: Numpy array of interior pressure points
        - m_points: Numpy array of enclosed mass points
    """
    if M / R >= 4 / 9:
        raise ValueError("Schwarzschild interior solution requires M/R < 4/9")

    # Central pressure of schwarzschild_solution
    p_c = (3 * M / (4 * pi * R ** 3)) * (1 - sqrt(1 - 2 * M / R)) / (3 * sqrt(1 - 2 * M / R) - 1)
    # Uniform energy density
    rho = 3 * M / (4 * pi * R ** 3)
    # Taylor expansion (order 3) near center to get initial conditions
    p_r0 = p_c - (2 / 3) * pi * r0 ** 2 * (rho + p_c) * (rho + 3 * p_c)
    m_r0 = (4 / 3) * pi * rho * r0 ** 3
    # Radial points
    r_points = linspace(r0, R, n)
    # Radial step size
    step = r_points[1] - r_points[0]
    # Initialize solution arrays
    p_points = empty(n, float)
    m_points = empty(n, float)
    init = array([p_r0, m_r0], float)  # Initial conditions
    # Looping
    for index, r in ndenumerate(r_points):
        p_points[index] = init[0]
        m_points[index] = init[1]
        # RK4 step
        k1 = step * tov_equations(init, r, rho)
        k2 = step * tov_equations(init + 0.5 * k1, r + 0.5 * step, rho)
        k3 = step * tov_equations(init + 0.5 * k2, r + 0.5 * step, rho)
        k4 = step * tov_equations(init + k3, r + step, rho)
        init += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return array([r_points, p_points, m_points], float)


"""
RK4 method for TOV equations with free central pressure using EoS
-----------------------------------------------------------------------------------------
Interior solution for spherically symmetric relativistic stars.

From:
    Computational Physics notes, course 24/25
    Jacobo Ruiz de Elvira Carrascal

Units:
    Geometrized units in CGS
"""


def rk4_eos_pc_free(p_c: ndarray, rho: Callable, r0: float = 1e-4, step: float = 10) -> ndarray:
    """
    :param p_c: Numpy array of central pressure points in GEO CGS units
    :param rho: Function of energy density in GEO CGS units
    :param r0: Initial point near center
    :param step: Radial step size in CGS
    :return:
        - M_points: Numpy array of total mass enclosed points
        - R_points: Numpy array of total radius
    """
    p_min = pressure_cgs_to_geo(0.35e15)
    print(p_min)
    print(p_c)
    # Initialize  arrays of total mass and total radius
    M_total = []
    R_total = []
    # Looping over central pressure
    for p_central in p_c:
        print(p_central)
        # Initialize radius variable
        r = r0
        # Taylor expansion (order 3) near center to get initial conditions
        rho_c = rho(p_central)
        print("rhocentral",rho_c)
        p_r0 = p_central - (2 / 3) * pi * r0 ** 2 * (rho_c + p_central) * (rho_c + 3 * p_central)
        m_r0 = (4 / 3) * pi * rho_c * r0 ** 3
        # Initial conditions
        init = array([p_r0, m_r0], float)
        # Integrate until null pressure (star surface)
        while init[0] > p_min:
            # RK4 step
            # k1
            p1 = init[0]
            rho1 = rho(p1)
            k1 = step * tov_equations(init, r, rho1)

            # k2
            y2 = init + 0.5 * k1
            p2 = y2[0]
            rho2 = rho(p2)
            k2 = step * tov_equations(y2, r + 0.5 * step, rho2)

            # k3
            y3 = init + 0.5 * k2
            p3 = y3[0]
            rho3 = rho(p3)
            k3 = step * tov_equations(y3, r + 0.5 * step, rho3)

            # k4
            y4 = init + k3
            p4 = y4[0]
            rho4 = rho(p4)
            k4 = step * tov_equations(y4, r + step, rho4)

            init += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            r += step
        # Store total mass and radius for this central pressure
        M_total.append(init[1])
        R_total.append(r)
    return array([M_total, R_total], float)
