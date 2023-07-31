#%%
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import UnivariateSpline
from evolving_densities import Density, smooth_Bspline

#%%

G_N = 4.302e-3 #Units of (pc/solar mass) (km/s)^2
c_light = 2.9979e05 #km/s

gamma = 1.2
r_0 = 8E3 #pc
rho_0_PL = 226

def rho_PL(r):
    return rho_0_PL*(r/r_0)**(-gamma)

r_cut = 1E4

def rho_exp(r):
    return rho_PL(r)*np.exp(-r/r_cut)

m_BH = 1E6
r_S = 2*G_N*m_BH/c_light**2
def psi_BH(r):
    return G_N*m_BH/r

r_array = np.logspace(-8, 1, num = 901)

density_PL = Density('density_PL_GS', rho_exp, r_array, N_particles = int(5E0))

density_GS_check = density_PL.adiabatic_growth(psi_BH, r_S = r_S, refinement = 15, figures = True)

#%%

rho_initial = rho_exp(r_array)
rho_mask = rho_initial > 0
logrho_initial_spline = UnivariateSpline(np.log(r_array[rho_mask]), np.log(rho_initial[rho_mask]), k = 2, s = 0)
dlogrho_initial = logrho_initial_spline.derivative(n = 1)

rho_GS = density_GS_check(r_array)
rho_mask = np.logical_and(rho_GS > 0, r_array > 4*r_S)
rho_GS_smooth = smooth_Bspline(r_array[rho_mask], rho_GS[rho_mask], increasing = False)
logrho_GS_spline = UnivariateSpline(np.log(r_array[rho_mask]), np.log(rho_GS_smooth(r_array)[rho_mask]), k = 2, s = 0)
dlogrho_GS = logrho_GS_spline.derivative(n = 1)

#%%

plt.figure()
plt.loglog(r_array, rho_initial, c = 'k', label = 'initial PL')
plt.loglog(r_array, rho_GS, c = 'r', label = 'GS spike')
plt.xlim(1E-8, 1E1)
plt.ylim(1E0, 1E21)
plt.legend()
plt.ylabel(r'$\rho$ ($M_\odot$pc$^{-3}$)')
plt.xlabel(r'$r$ (pc)')
plt.figure()

plt.figure()
plt.semilogx(r_array, dlogrho_initial(np.log(r_array)), '.', c = 'k', label = 'initial PL')
plt.semilogx(r_array, dlogrho_GS(np.log(r_array)), '.', c = 'r', label = 'GS spike')
plt.axhline(-33/14, c = 'c', ls = '--')
plt.ylim(-2.5, -0.5)
plt.xlim(1E-8, 1E0)
plt.legend()
plt.ylabel(r'd log$(\rho)$/d log($r$)')
plt.xlabel(r'$r$ (pc)')
plt.figure()
