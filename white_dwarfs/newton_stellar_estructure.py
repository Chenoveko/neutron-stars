import numpy as np
from lane_emden_polytrope import Polytrope
M_SUN = 1.98847e30  # kg
def n_from_gamma(gamma: float) -> float:
    """
    :param gamma:
    :return: polytropic index
    """
    return 1.0/(gamma - 1.0)

def alpha(n: float, K: float, rho_c: float, G: float) -> float:
    # alpha^2 = ((n+1)K/(4πG)) * rho_c^(1/n - 1)
    return np.sqrt(((n + 1.0) * K / (4.0*np.pi*G)) * rho_c**(1.0/n - 1.0))

def radius_mass_from_polytrope(p, K: float, rho_c: float, G: float):
    """
    p: instancia de Polytrope(n)
    K, rho_c, G: en unidades consistentes
    devuelve (R, M)
    """
    xi1, p2 = p.get_params()   # p2 = -xi1^2 * theta'(xi1)
    a = alpha(p.n, K, rho_c, G)
    R = a * xi1
    M = 4.0*np.pi * a**3 * rho_c * p2
    M_msun = M / M_SUN
    return R, M, M_msun

p = Polytrope(3)     # ejemplo
G = 6.67430e-11      # SI
K = 1.0              # tu K en SI
rho_c = 1.0          # tu rho_c en SI

R, M, M_msun = radius_mass_from_polytrope(p, K, rho_c, G)
print("M =", M, "kg")
print("M =", M_msun, "M_sun")