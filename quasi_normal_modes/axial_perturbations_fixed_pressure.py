# =====================
# Imports
# =====================
from numpy import array, pi,angle
from scipy.interpolate import PchipInterpolator

from equations_of_structure.interpol_data import rho_pchip_geo_apr
from solve_qnm import matching_function
from tov_solvers import solve_tov_eos
from utilities.physical_data import M_sun, omega_cgs_to_geo, mass_cgs_to_geo, omega_geo_to_cgs
from mullerpy import muller

# ==========Parameters==========#
M_sun_geo = mass_cgs_to_geo(M_sun)  # Solar mass in GEO units
p_central_apr_geo = 1.21607989338143e-14
# p_central_gnh3_geo = 5.4257477047488676e-15
# p_central_sly4_geo = 1.103445918598322e-14
tol_f = 1e-8
tol_omega = 1e-8
# =================================#
#            QNM APR EoS           #
# =================================#
# ==========TOV Solutions==========#
r_apr, p_apr, m_apr, nu_apr, _ = solve_tov_eos(p_central_apr_geo, rho_pchip_geo_apr)
# ==========Interpolation==========#
rho_apr = array([rho_pchip_geo_apr(p) for p in p_apr])
p_fun = PchipInterpolator(r_apr, p_apr)
m_fun = PchipInterpolator(r_apr, m_apr)
nu_fun = PchipInterpolator(r_apr, nu_apr)
rho_fun = PchipInterpolator(r_apr, rho_apr)
dm_fun = m_fun.derivative()
# ==========Mode parameters==========#
l = 2
# ==========Initial guesses of QNM==========#
# Use three distinct damped guesses. All of them should satisfy Re(omega) > 0 and Im(omega) < 0.
omega0_apr = omega_cgs_to_geo(4.8e4 - 2e4j)
omega1_apr = omega_cgs_to_geo(5.3e4 - 2e4j)
omega2_apr = omega_cgs_to_geo(5.3e4 - 8e3j)
omegas_apr = array([omega0_apr, omega1_apr, omega2_apr], dtype=complex)
alpha = -max([angle(omega) for omega in omegas_apr]) - 0.2
# ==========Matching function==========#
f_qnm = lambda omega: matching_function(omega, r_apr[0], r_apr[-1], m_fun, p_fun, rho_fun, nu_fun, dm_fun, m_apr[-1], alpha,l)
# ==========Initialization==========#
print('# ==========Initial guesses of QNM==========#')
for omega in omegas_apr:
    f = f_qnm(omega)
    if abs(f) < tol_f:
        print('Initial QNM candidate found')
        print("Omega real [KHz]", omega_geo_to_cgs(omega).real / (2 * pi * 1e3))
        print("Damping time [microsec]", 1 / abs(omega_geo_to_cgs(omega).imag) * 1e6)
        print("|f(omega)| =", abs(f))

# ==========Muller method looking for QNM==========#

print('# ==========Iterative Muller search of QNM==========#')

triangle_scale_apr = 1.0
omegas_base = omegas_apr.copy()


for i in range(10):
    centroid = omegas_base.mean()
    omegas_trial = centroid + triangle_scale_apr * (omegas_base - centroid)

    res = muller(f_qnm, omegas_trial, xtol=1e-8, ftol=1e-8, maxiter=50)

    if res.converged:
        omega_root = res.root
        omega_root_cgs = omega_geo_to_cgs(omega_root)

        freq_khz = omega_root_cgs.real / (2 * pi * 1e3)
        tau_us = 1e6 / abs(omega_root_cgs.imag)

        print("Candidate root found")
        print("Muller it:", res.iterations)
        print("Rescale it:", i)
        print("Omega real [KHz]:", freq_khz)
        print("Damping time [microsec]:", tau_us)
        print("|f(root)| =", abs(f_qnm(omega_root)))

        if (
            omega_root_cgs.real > 0
            and omega_root_cgs.imag < 0
            and 5.0 <= freq_khz <= 10.0
            and 10.0 <= tau_us <= 120.0
        ):
            print("QNM accepted")
            break
        else:
            print("Root rejected: shrinking triangle")
            triangle_scale_apr *= 0.5

    else:
        print("Muller did not converge: enlarging triangle")
        triangle_scale_apr += 0.2

