import numpy as np
import matplotlib.pyplot as plt

# Constantes CGS
c = 2.99792458e10
G = 6.67430e-8
M_sun_geom_km = 1.476625  # km
KM_TO_CM = 1e5

# Conversión geom(cm^-2) -> CGS
PRESSURE_GEOM_TO_CGS = c**4 / G  # dyn/cm^2

def tov(y, r, rho):
    m, p = y
    dm = 4*np.pi*r**2*rho
    dp = -(rho + p) * (m + 4*np.pi*r**3*p) / (r**2 * (1 - 2*m/r))
    return np.array([dm, dp], float)

def rk4_step(f, y, r, h, rho):
    k1 = h * f(y, r, rho)
    k2 = h * f(y + 0.5*k1, r + 0.5*h, rho)
    k3 = h * f(y + 0.5*k2, r + 0.5*h, rho)
    k4 = h * f(y + k3, r + h, rho)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

def R_from_rho_Pc(rho0, Pc):
    inside = 1.0 - ((rho0 + Pc)**2)/((rho0 + 3.0*Pc)**2)
    return np.sqrt((1.0/(8.0*np.pi*rho0))*inside) if inside > 0 else np.nan

def p_analytic_const_rho(r, rho0, R, M):
    r = np.asarray(r, float)
    A = np.sqrt(np.maximum(0.0, 1.0 - 2.0*M*r**2/(R**3)))
    B = np.sqrt(np.maximum(0.0, 1.0 - 2.0*M/R))
    return rho0*(A - B)/(3.0*B - A)

def m_analytic_const_rho(r, rho0):
    r = np.asarray(r, float)
    return (4/3)*np.pi*rho0*r**3

# Parámetros (geométricos en cm)
rho = 2.2e-14
p_c = 4.1e-15

h   = 1e2        # cm (1 m)
r0  = 1e-2       # cm
rmax = 50*KM_TO_CM  # 50 km en cm

m0 = (4/3)*np.pi*rho*r0**3
y = np.array([m0, p_c], float)
r = r0

rs, ms, ps = [r], [y[0]], [y[1]]

while r < rmax and y[1] > 0:
    y = rk4_step(tov, y, r, h, rho)
    r += h
    rs.append(r); ms.append(y[0]); ps.append(y[1])

rs = np.array(rs); ms = np.array(ms); ps = np.array(ps)

# Analítica
R_anal = R_from_rho_Pc(rho, p_c)
M_anal = m_analytic_const_rho(R_anal, rho)

p_anal = np.zeros_like(rs)
mask = rs <= R_anal
p_anal[mask] = p_analytic_const_rho(rs[mask], rho, R_anal, M_anal)
m_anal = m_analytic_const_rho(rs, rho)

# Para graficar: r en km
rs_km = rs / KM_TO_CM

# Presión a CGS
ps_cgs = ps * PRESSURE_GEOM_TO_CGS
p_anal_cgs = p_anal * PRESSURE_GEOM_TO_CGS

# Masa a Msun (m_geom en cm -> primero a km dividiendo por 1e5)
ms_solar = (ms / KM_TO_CM) / M_sun_geom_km
m_anal_solar = (m_anal / KM_TO_CM) / M_sun_geom_km

plt.figure()
plt.plot(rs_km, ps_cgs, label="Numérica (RK4)")
plt.plot(rs_km, p_anal_cgs, "--", label="Analítica (ρ cte)")
plt.xlabel("r [km]")
plt.ylabel(r"p [dyn/cm$^2$]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(rs_km, ms_solar, label="Numérica (RK4)")
plt.plot(rs_km, m_anal_solar, "--", label="Analítica (ρ cte)")
plt.xlabel("r [km]")
plt.ylabel(r"$m/M_\odot$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
