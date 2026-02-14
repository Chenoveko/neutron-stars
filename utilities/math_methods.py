# =====================
# Imports
# =====================
from typing import Callable, Tuple

from numpy import ndarray, array, pi, sqrt, isfinite
from scipy.integrate import solve_ivp

from utilities.physical_functions import tov_equations

"""
Numerical integration to solve TOV equations with uniform density (Schwarzschild solution)
-----------------------------------------------------------------------------------------
Interior solution for spherically symmetric relativistic stars with uniform-density.

From:
    SciPy documentation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp

Units:
    Geometrized units in CGS
"""


def solve_tov_schwarzschild(M: float, R: float, r0: float = 1e-8, method: str = 'RK45') -> ndarray:
    """
    :param M: total mass
    :param R: total radius
    :param r0: radial point near center
    :param method: Integration method to use
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

    # System to use solve_ivp
    def tov_system(r, y):
        p, m = y
        if p <= 0:
            return [0.0, 0.0]
        return tov_equations(y, r, rho)

    # Initial conditions
    init = array([p_r0, m_r0], float)
    # Integrate
    sol = solve_ivp(fun=tov_system, t_span=(r0, R), y0=init, method=method, first_step=10, max_step=20, rtol=1e-4,
                    atol=1e-6)
    return array([sol.t, sol.y[0], sol.y[1]], float)


"""
Numerical integration to solve TOV equations with free central pressure using EoS
-----------------------------------------------------------------------------------------
Interior solution for spherically symmetric relativistic stars.

From:
    SciPy documentation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp

Units:
    Geometrized units in CGS
"""


def solve_tov_eos(p_c: float, rho_func: Callable, r0: float = 1e-4, r_max: float = 20e5,
                  method: str = 'RK45') -> Tuple[ndarray, ndarray, ndarray, bool]:
    """
    Solves the TOV equations for a given equation of state
    :param p_c: Central pressure in GEO-CGS units
    :param rho_func: Function that returns the energy density as a function of pressure.
    :param r0: Initial radius near the center to avoid numerical singularities.
    :param r_max: Maximum radius up to which the integration is performed.
    :param method: Numerical integration method used by the ODE solver.
    :return:
        - r : Numpy array of radial points
        - p : Numpy array of interior pressure
        - m : Numpy array of enclosed mass
        - status: Stop event bool, 1 if stopped by an event, 0 if reached r_max
    """
    # Integrate until pressure has dropped by 15 orders of magnitude
    p_min = p_c * 1e-15
    # Taylor expansion (order 3) near center to get initial conditions
    rho_c = rho_func(p_c)
    p_r0 = p_c - (2 / 3) * pi * r0 ** 2 * (rho_c + p_c) * (rho_c + 3 * p_c)
    m_r0 = (4 / 3) * pi * rho_c * r0 ** 3
    # Initial conditions
    init = array([p_r0, m_r0], float)

    # System to use solve_ivp
    def tov_system(r, y):
        p, m = y
        rho_val = rho_func(p)
        return tov_equations(y, r, rho_val)

    # Event to trigger when pressure crosses p_min downward
    def event_pressure_threshold(r, y):
        return y[0] - p_min

    event_pressure_threshold.terminal = True
    event_pressure_threshold.direction = -1

    # Event to trigger when pressure becomes negative
    def event_negative_pressure(r, y):
        return y[0]

    event_negative_pressure.terminal = True
    event_negative_pressure.direction = -1

    # Event to trigger when mass saturates
    def event_mass_saturation(r, y):
        p, m = y
        eps_rel = 1e-8
        if (not isfinite(p)) or (p <= 0) or (not isfinite(m)) or (m <= 0):
            return 0.0
        rho = rho_func(p)
        if (not isfinite(rho)) or (rho <= 0):
            return 0.0
        dm_dr = 4.0 * pi * r ** 2 * rho
        return (dm_dr / m) - eps_rel

    event_mass_saturation.terminal = True
    event_mass_saturation.direction = -1
    # Integrate
    sol = solve_ivp(fun=tov_system, t_span=(r0, r_max), y0=init, method=method,
                    events=[event_pressure_threshold, event_negative_pressure, event_mass_saturation], max_step=1e3,
                    rtol=1e-4, atol=1e-6)
    r = sol.t
    p = sol.y[0]
    m = sol.y[1]
    status = sol.status
    return r, p, m, status
