import numpy as np
import matplotlib.pyplot as plt

# Constantes en CGS
G = 6.67430e-8          # cm^3 g^-1 s^-2
c = 2.99792458e10       # cm s^-1
M_sun = 1.98847e33      # g

# Array de numpy de los datos
akmalpr  = np.loadtxt("../eos/eos_akmalpr.txt", comments="#")
glendnh3 = np.loadtxt("../eos/eos_glendnh3.txt", comments="#")
sly4     = np.loadtxt("../eos/eos_sly4.txt", comments="#")