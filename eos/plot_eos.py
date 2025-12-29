import numpy as np
import matplotlib.pyplot as plt

# Array de numpy de los datos
akmalpr = np.loadtxt("eos_akmalpr.txt", comments="#")
glendnh3 = np.loadtxt("eos_glendnh3.txt", comments="#")
sly4 = np.loadtxt("eos_sly4.txt", comments="#")

# Slicing para sacar 2º y 3º columna (p y rho respectivamente)
p_akmalpr = akmalpr[:,3]
rho_akmalpr = akmalpr[:,2]

p_glendnh3 = glendnh3[:,3]
rho_glendnh3 = glendnh3[:,2]

p_sly4 = sly4[:,3]
rho_sly4 = sly4[:,2]

# Logaritmos base 10
logp_akmalpr = np.log10(p_akmalpr)
logrho_akmalpr = np.log10(rho_akmalpr)

logp_glendnh3 = np.log10(p_glendnh3)
logrho_glendnh3 = np.log10(rho_glendnh3)

logp_sly4 = np.log10(p_sly4)
logrho_sly4 = np.log10(rho_sly4)

# Gráfica de las 3 EOS
plt.figure()

plt.plot(logp_glendnh3, logrho_glendnh3, label="Glendenning NH3")
plt.plot(logp_sly4, logrho_sly4, label="SLy4")
plt.plot(logp_akmalpr, logrho_akmalpr, label="APR")

plt.xlabel(r'$\log_{10}(p)$')
plt.ylabel(r'$\log_{10}(\rho)$')

plt.grid(True,alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Gráfica con ampliación de las 3 EOS (log p > 31)
plt.figure()

plt.plot(logp_glendnh3, logrho_glendnh3, label="Glendenning NH3")
plt.plot(logp_sly4, logrho_sly4, label="SLy4")
plt.plot(logp_akmalpr, logrho_akmalpr, label="APR")

plt.xlabel(r'$\log_{10}(p)$')
plt.ylabel(r'$\log_{10}(\rho)$')

plt.xlim(31.7, 36.8)
plt.ylim(13, 16)

plt.grid(True,alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
