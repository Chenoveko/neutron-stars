from numpy import pi, exp, sin, cos,atan
from typing import Callable


#######################################
# Axial perturbations inside the star #
#######################################
def axial_potential_in(r: float, m: Callable, p: Callable, rho: Callable, nu: Callable, l: int = 2) -> float:
    """
    Axial potential inside the star in GEO units
        V(r) = e^{2ν(r)} / r^3 * [ l(l+1) r + 4π r^3 (ρ(r) + p(r)) - 6 m(r) ]
    :param r: radial coordinate
    :param m: mass value m(r)
    :param p: pressure value p(r)
    :param rho: energy density value rho(r)
    :param nu: metric value nu(r)
    :param l: angular momentum number
    :return: axial potential
    """
    return exp(2.0 * nu(r)) / r ** 3 * (l * (l + 1) * r + 4.0 * pi * r ** 3 * (rho(r) + p(r)) - 6.0 * m(r))


def regge_wheeler_in(r: float, var, omega: complex, m: Callable, p: Callable, rho: Callable, nu: Callable,
                     l: int = 2) -> tuple[float, float, float, float]:
    """
    Regge-Wheeler linear ODE inside the star in GEO units,
    Equation:
        d²z/dr_*² + [omega² - V(r)] z = 0
    Using:
        dr*/dr = exp[-nu(r)]/f(r)^1/2
        f(r) = 1 - 2m(r)/r
        f'(r) = 2m(r)/r^2 - 2m'(r)/r
        m'(r) = 4*pi*r^2*rho(r)
    This gives:
        d²z/dr² =[-nu'(r) - f'(r)/(2f(r))] dz/dr- [exp(-2nu(r))/f(r)] * (omega² - V(r)) z
    :param r: radial coordinate
    :param var: state variables [Re(z),Im(z), Re(dz/dr),Im(dz/dr)]
    :param omega: angular frequency value
    :param m: mass value
    :param p: pressure value
    :param rho: energy density value
    :param nu: metric value
    :param l: angular momentum number
    :return: derivatives [Re(dz/dr),Im(dz/dr), Re(d²z/dr²),Im(d²z/dr²)]
    """
    # Unpack variables
    z_re, z_im, dzdr_re, dzdr_im = var
    # Reconstruct complex variables
    z = z_re + 1j * z_im
    dzdr = dzdr_re + 1j * dzdr_im
    # Metric functions
    f = 1.0 - 2.0 * m(r) / r
    # Derivatives
    m_prime = 4.0 * pi * r ** 2 * rho(r)
    f_prime = -2.0 * m_prime / r + 2.0 * m(r) / r ** 2
    # nu'(r) from TOV
    nu_prime = (m(r) + 4.0 * pi * r ** 3 * p(r)) / (r * (r - 2.0 * m(r)))
    # Potential
    v = axial_potential_in(r, m, p, rho, nu, l)
    # Second derivative (correct interior RW form)
    d2zdr2 = ((-nu_prime - f_prime / (2.0 * f)) * dzdr - (exp(-2.0 * nu(r)) / f) * (omega ** 2 - v) * z)
    return dzdr_re, dzdr_im, d2zdr2.real, d2zdr2.imag


########################################
# Axial perturbations outside the star #
########################################

def axial_potential_out(r: complex, M: float, l: int = 2) -> complex:
    """
    Axial potential outside the star in GEO units
        V(r) = (1 - 2M/r) * [l(l+1)/r^2 - 6M/r^3]
    :param r: radial coordinate
    :param M: total mass value
    :param l: angular momentum number
    :return: axial potential
    """
    return (1.0 - 2.0 * M / r) * (l * (l + 1) / r ** 2 - 6.0 * M / r ** 3)


def compact_rot_coord(t: float, R: float, alpha: float) -> complex:
    """
    Compactified rotated coordinate outside the star
    for Complex Exterior Scaling in GEO units
        r(t) = R + (1-t)/t exp(i*alpha)
    :param t: compactified coordinate t ∈ (0, 1]
    :param omega: angular frequency value
    :param R: total radius
    :param alpha: rotate angle
    :return: complex radial coordinate r(t)
    """
    return R + ((1.0 - t) / t) * exp(1j * alpha)


def regge_wheeler_out(t: float, var, omega: complex, alpha: float, M: float, R: float, l: int = 2) -> tuple[
    float, float]:
    """
    Regge-Wheeler linear ODE outside the star in GEO units
    Equation:
        d²Z/dr_*² + [omega² - V(r)] Z = 0
    Define:
        g(r) = (1/Z) dZ/dr
    Using:
        dr*/dr = 1 / f(r)
        f(r) = 1 - 2M/r
        d/dr_* = f(r) d/dr
    This gives:
        dZ/dr_* = f dZ/dr = f g Z
        d²Z/dr_*² = f² (dg/dr + g²) Z + f f' g Z
    Hence the Riccati equation:
        f² (dg/dr + g²) + f f' g + omega² - V(r) = 0
    i.e.:
        dg/dr = -g² - (f'/f) g - (omega² - V(r))/f²
    Using CES:
        r(t) = R + (1-t)/t exp(i alpha)
        dr/dt = -exp(i alpha)/t²
    Therefore:
        dg/dt = (dg/dr)(dr/dt)
    :param t: compactified coordinate t ∈ (0, 1]
    :param var: state variables [Re(g), Im(g)]
    :param omega: angular frequency value
    :param alpha: complex rotation angle
    :param M: total mass
    :param R: total radius
    :param l: angular momentum number
    :return: derivatives [Re(dg/dt), Im(dg/dt)]
    """
    if alpha_condition(omega, alpha):
        # Unpack variables
        g_re, g_im = var
        # Reconstruct complex variable
        g = g_re + 1j * g_im
        # CES parametrization
        r = compact_rot_coord(t, R, alpha)
        # Metric functions
        f = 1.0 - 2.0 * M / r
        f_prime = 2.0 * M / r ** 2
        # Potential
        v = axial_potential_out(r, M, l)
        # Derivatives
        drdt = -exp(1j * alpha) / t ** 2
        dgdr = -g ** 2 - (f_prime / f) * g - (omega ** 2 - v) / f ** 2
        dgdt = drdt * dgdr
        return dgdt.real, dgdt.imag
    else:
        raise ValueError("Alpha condition for CES is not met")

def alpha_condition(omega: complex, alpha: float) -> bool:
    """
    Alpha condition for CES
        Im(omega) * cos(alpha) - Re(omega) * sin(alpha) < 0
    :param omega: angular frequency value
    :param alpha: complex rotation angle
    :return: True if alpha condition is met, False otherwise
    """
    return omega.imag * cos(alpha) - omega.real * sin(alpha) < 0

def fix_alpha(omega: complex, alpha: float) -> float:
    """
    Ensure alpha satisfies CES condition. If not, pick a random valid one.
    :param omega: angular frequency value
    :param alpha: complex rotation angle
    :return: new alpha value
    """
    if alpha_condition(omega, alpha):
        return alpha
    print("Changing alpha value")
    return atan(omega.imag / omega.real) + 1e-3