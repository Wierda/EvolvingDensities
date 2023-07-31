#%%
import matplotlib.pyplot as plt
import numpy as np

import copy

from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from matplotlib.colors import LogNorm, Normalize
from evolving_densities import Density, smooth_Bspline

#%%

G_N = 4.302e-3 #Units of (pc/solar mass) (km/s)^2
c_light = 2.9979e05 #km/s

r_0 = 8E3 #pc
rho_0 = 0.3 * (3.0857E18)**3 / (1.9885E30/1.783E-27) #Msun/pc^3
a = 20E3 #pc

rho_0_prime = rho_0 * r_0/a * (1 + r_0/a)**2

def rho_NFW(r):
    return rho_0_prime * (a/r) / (1 + r/a)**2

m_BH = 1E6
r_S = 2*G_N*m_BH/c_light**2
def psi_BH(r):
    return G_N*m_BH/r

r_array = np.logspace(-8, 5, num = 1301)

density_NFW = Density('density_NFW_UZK', rho_NFW, r_array, N_particles = int(1E5))

density_UZK_check = density_NFW.non_adiabatic_growth(psi_BH, r_S = r_S, logr = True, figures = True)

#%%

rho_initial = density_NFW(r_array)
rho_mask = rho_initial > 0
logrho_initial_spline = UnivariateSpline(np.log(r_array[rho_mask]), np.log(rho_initial(r_array)[rho_mask]), k = 2, s = 0)
dlogrho_initial = logrho_initial_spline.derivative(n = 1)

rho_final_UZK = density_UZK_check(r_array)
rho_mask = np.logical_and(rho_final_UZK > 0, r_array > 4*r_S)
rho_UZK_smooth = smooth_Bspline(r_array[rho_mask], rho_final_UZK[rho_mask], increasing = False)
logrho_UZK_spline = UnivariateSpline(np.log(r_array[rho_mask]), np.log(rho_UZK_smooth(r_array)[rho_mask]), k = 2, s = 0)
dlogrho_UZK = logrho_UZK_spline.derivative(n = 1)


#%%

fig, ax = plt.subplots(figsize = (4, 4))
ax.plot(r_array, density_NFW(r_array), c = 'k', label = r'$\rho_{i}$')
ax.plot(r_array, density_UZK_check(r_array),  c = 'r', label = r'$\rho_{UZK}$')
ax.set_ylim(1E-3)
ax.set_xlim(1E-8, 1E5)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel(r'$\rho$')
ax.legend()
ax.set_xlabel(r'$r$ (pc)')
plt.show()


plt.figure()
plt.semilogx(r_array, dlogrho_initial(np.log(r_array)), '.', c = 'k', label = r'$\rho_{i}$')
plt.semilogx(r_array[r_array > 4*r_S], dlogrho_UZK(np.log(r_array[r_array > 4*r_S])), '.', c = 'r', label = 'UZK')
plt.axhline(-4/3, c = 'c', ls = '--')
plt.ylim(-2.5, -0.5)
plt.xlim(1E-8, 1E4)
plt.legend()
plt.ylabel(r'd log$(\rho)$/d log($r$)')
plt.xlabel(r'$r$ (pc)')
plt.figure()
