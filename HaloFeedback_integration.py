#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad 
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from copy import copy

import time

from evolving_densities import Density, monotonic_Bspline

import os

## Make sure you have HaloFeedback installed
# HaloFeedback_dir = '/'
# os.chdir(HaloFeedback_dir)

import HaloFeedback


#%%

#############################
#### fundamental constants ##
#############################

s = 1. #s 
year = 365.25*24.*3600.*s
km = 1e5 #cm
Msun = 1.98855e33  # kg
pc = 3.08567758149137e18 # cm

verbose = 2 

system = 'dynamic'

############################
#  Simulation parameters   #
############################

NPeriods_ini = 10. 
dN_max = 250. 

SHORT = False

#%%
######################################
# BH, spike and binary parameters   ##
######################################

G_N_pc = 4.302e-3 #Units of (pc/solar mass) (km/s)^2
c_light = 2.9979e05 #km/s

M_halo = 2E8
R_vir = 1E3 
c = 15
a_NFW = np.log(1 + c) - c/(1 + c)
rho_halo = M_halo/(4/3 * np.pi * R_vir**3)

rho_0_prime = rho_halo*c**3/(3*a_NFW)
R_s = R_vir/c

def rho_NFW(r):
    return rho_0_prime * (R_s/r) / (1 + r/R_s)**2 * np.exp(-r/R_vir)

r_array_full = np.logspace(-10, 3, num = 1301)
density_NFW = Density('density_NFW_SMS', rho_NFW, r_array_full, N_particles = int(1E5))
density_NFW.setup_potentials()

m_BH = 1E5
r_S = 2*G_N_pc*m_BH/c_light**2
def psi_BH(r):
    return G_N_pc*m_BH/r

psi_NFW = copy(density_NFW.get_psi())
delta_psi = lambda r: psi_BH(r) - psi_NFW(r)
density_GS_pred = density_NFW.adiabatic_growth(delta_psi, refinement = 15)

#%%

rho_GS_pred = density_GS_pred(r_array_full)
rho_mask = np.logical_and(rho_GS_pred > 0, r_array_full > 4*r_S)
rho_GS_mono = monotonic_Bspline(r_array_full[rho_mask], rho_GS_pred[rho_mask])

def log_fit_func_GS(x, b, c): # 

    return np.log(rho_0_prime*(R_s/c)) -b*np.log(x/c)

gamma_sp = 7/3
r_sp = 0.122 * R_s * ( m_BH / ( rho_0_prime * R_s**3 ) )**(1./(3.-1))

p0_GS = [gamma_sp, r_sp]
fit_GS = curve_fit(log_fit_func_GS, r_array_full[rho_mask], np.log(rho_GS_pred[rho_mask]), p0_GS)
print(p0_GS)
print(fit_GS[0])
print(np.sqrt(np.diag(fit_GS[1])))

norm_sp = rho_0_prime*(R_s/fit_GS[0][1])

#%%
r_array = np.logspace(np.log10(4*r_S), 0, num = 1000)

density_GS_spike = Density('density_GS_spike', rho_GS_mono, r_array, N_particles = 5)
density_GS_spike.setup_potentials()
psi_GS_spike = copy(density_GS_spike.get_psi())
phi_GS_spike = copy(density_GS_spike.get_phi())
delta_psi = lambda r: psi_BH(r) - psi_GS_spike(r)
delta_phi = lambda r: psi_BH(r_array[-1]) - psi_BH(r) - phi_GS_spike(r)
density_GS_spike.add_external_potential_from_function(delta_phi, delta_psi)
density_GS_spike.setup_phase_space(smoothen = True)

#%%

class BluntSpike(HaloFeedback.DistributionFunction):

    def __init__(self, density: Density, m_BH = 1E5, m_NS = 1E0, r_sp = 1, Lambda = -1):
        self._rho_init_func = density._rho_func
        # self._psi_init_func = density.get_psi()
        self._f_init_func = density.get_f_Eddington()
        self.r_sp = r_sp

        super().__init__(m_BH, m_NS, Lambda)

    def rho_init(self, r):
        return self._rho_init_func(r)
    
    def f_init(self, eps):
        return self._f_init_func(eps)
    

G_N = 6.67408e-8
c_light = 2.99792458e10

def run_inspiral(M1, M2, dist, r0_initial, system = 'dynamic'):

    M = M1 + M2

    r_grav = 2*G_N*M1/c_light**2
    r_isco = 3 * r_grav
    r_in   = 2 * r_grav

    r_end = r_grav*4

    
    def calc_Torb(r):
        return 2 * np.pi * np.sqrt(r ** 3 / (G_N * M))

    def calc_f(r):
        return 2/calc_Torb(r)

    def calc_vorb(r):
        return 2*np.pi*r/calc_Torb(r)

    f_initial  = calc_f(r0_initial)

    ################################
    ##### Equations of motion ######
    ################################
    
    
    def get_density(r):
        if (system == "vacuum"):
            rho_xi = 0.0
        elif (system == "static"):
            rho_xi = dist.rho_init(r/pc)*dist.xi_init
        elif (system in ["dynamic", "pbh"]):
            v_orb = calc_vorb(r)
            rho_xi = dist.rho(r/pc, v_cut=v_orb/(km/s))
        return rho_xi*Msun/pc**3

    # Radiation reaction
    # See, for example, eq. 226 in https://arxiv.org/pdf/1310.1528.pdf
    GW_prefactor = -64.*(G_N**3.)*M1*M2*(M)/(5.*(c_light**5.))
    def GW_term(t,r):
        return GW_prefactor/r**3

    # Gravitational "friction"
    # See, for example, https://arxiv.org/pdf/1604.02034.pdf
    lnLambda = np.log(np.sqrt(M1/M2))
    DF_prefactor = -(8*np.pi*G_N**0.5*M2*lnLambda/(M**0.5*M1))
    def DF_term(t, r):
        return DF_prefactor*r**2.5*get_density(r)

    
    # Derivatives (for feeding into the ODE solver)
    def drdt_ode(t, r):
        GW = GW_term(t, r)
        if (system == "vacuum"):
            DF = 0
        else:
            DF = DF_term(t, r)
        #print(DF/GW)
        return GW + DF

    #################################################
    ############ DYNAMIC DRESS ######################
    #################################################
    t_list = np.array([0.])
    r_list = np.array([r0_initial])
    f_list = np.array([f_initial])
    rho_list = np.array([get_density(r0_initial)])

    start_time = time.time()


    NPeriods = 1*NPeriods_ini
    r0 = r0_initial
    t0 = 0.
    i = 0


    dN = 1.0*NPeriods_ini


    while (r0 > r_end):
        
        dt = calc_Torb(r0)*dN
        #print(calc_Torb(r0))
        
        dN = np.clip(dN, 0, dN_max)
        
        v_orb = calc_vorb(r0)
        #print(v_orb/(km/s))
        if (system in ["dynamic", "pbh"]):
            dfdt1 = dist.dfdt(r0/pc, v_orb/(km/s), v_cut=v_orb/(km/s))
        
        
        
        if (system in ["dynamic", "pbh"]):
            excess_list = -(2/3)*dt*dfdt1/(dist.f_eps + 1e-30)
            excess = np.max(excess_list[1:]) #Omit the DF at isco                       
                                        
            if (excess > 1):
                dN /= excess*1.1
                if (verbose > 2):
                    print("Too large! New value of dN = ", dN)
                                                        
            elif (excess > 1e-1):
                dN /= 1.1
                if (verbose > 2):
                    print("Getting large! New value of dN = ", dN)
                                                                    
            elif ((excess < 1e-2) and (i%100 == 0) and (i > 0) and (dN < dN_max)):
                dN *= 1.1
                if (verbose > 2):
                    print("Increasing! New value of dN = ", dN)

            dt = calc_Torb(r0)*dN

        #Use Ralston's Method (RK2) to evolve the system 
        drdt1 = drdt_ode(t0, r0)

        r0          += (2/3)*dt*drdt1
        if (system in ["dynamic", "pbh"]):
            dist.f_eps  += (2/3)*dt*dfdt1

        drdt2 = drdt_ode(t0, r0)
        
        if (system in ["dynamic", "pbh"]):
            v_orb = calc_vorb(r0)
            dfdt2 = dist.dfdt(r0/pc, v_orb/(km/s), v_cut=v_orb/(km/s))

        r0          += (dt/12)*(9*drdt2 - 5*drdt1)
        if (system in ["dynamic", "pbh"]):
            dist.f_eps  += (dt/12)*(9*dfdt2 - 5*dfdt1)
        
        t0 += dt
        
        f0 = calc_f(r0)
        
        t_list = np.append(t_list, t0)
        r_list = np.append(r_list, r0)
        f_list = np.append(f_list, f0)
        rho_list = np.append(rho_list, get_density(r0))
        
        if (i%1000==0):
            if (verbose > 1):
                print(f">    r/r_end = {r0/r_end:.5f}; f_GW [Hz] = {f0:.5f}; t [s] = {t0:.5f}; rho_eff [Msun/pc^3] = {rho_list[-1]/(Msun/pc**3):.4e}")
        
        i = i+1
    
  
    #Correct final point to be exactly r_end (rather than < r_end)
    inds = np.argsort(r_list)
    t_last = np.interp(r_end, r_list[inds], t_list[inds])
    f_last = calc_f(r_end)

    t_list[-1] = t_last
    r_list[-1] = r_end
    f_list[-1] = f_last
    rho_list[-1] = get_density(r_end*1.000001)

    #Make some plots
    fig, ax = plt.subplots(ncols=2, nrows=1,figsize=(10, 5))
    ax[0].semilogy(t_list, r_list/pc)
    ax[0].set_xlabel(r"$t$ [s]")
    ax[0].set_ylabel(r"$R$ [pc]")
        
    ax[1].loglog(r_list/pc, rho_list/(Msun/pc**3))
    ax[1].set_xlabel(r"$R$ [pc]")
    ax[1].set_ylabel(r"$\rho_{\mathrm{eff}, v < v_\mathrm{orb}}(R)$ [$M_\odot\,\mathrm{pc}^{-3}$]")

    plt.tight_layout()


    plt.show()
    print("> Done")               
    print("> Time needed: %s seconds" % (time.time() - start_time))
    print(" ")

    return dist, t_list, r_list, f_list, rho_list                
      

M1 = 1E5*Msun
M2 = 1.*Msun # args.M2*Msun

#Initial values of a few different parameters
r_grav = 2*G_N*M1/c_light**2
r0_initial = 15*r_grav

dist_GS_analytical = HaloFeedback.PowerLawSpike(gamma = fit_GS[0][0], M_BH = M1/Msun, M_NS = M2/Msun, rho_sp = norm_sp, r_sp = fit_GS[0][1])
dist_GS_numerical = BluntSpike(density_GS_spike, m_BH = M1/Msun, m_NS = M2/Msun, r_sp = r_array[-1]/1E3)

results_dynamic_GS_analytical = run_inspiral(M1, M2, dist_GS_analytical, r0_initial, system = 'dynamic')
results_dynamic_GS_numerical = run_inspiral(M1, M2, dist_GS_numerical, r0_initial, system = 'dynamic')
results_vacuum = run_inspiral(M1, M2, dist_GS_analytical, r0_initial, system = 'vacuum')

#%%

f_vacuum_t = interp1d(results_vacuum[1]/year - np.max(results_vacuum[1]/year), results_vacuum[4], 
                      fill_value = (np.min(results_vacuum[4])/10, np.max(results_vacuum[4]*10)), bounds_error = False)
f_min = f_vacuum_t(-5)
f_array = np.linspace(f_min*0.9, np.max(results_vacuum[4]), num = 1000)
r_test = np.linspace(np.min(results_vacuum[3]/pc), np.max(results_vacuum[3]/pc), num = 1000)

t_vacuum_f = interp1d(results_vacuum[3], results_vacuum[1]/year - np.max(results_vacuum[1]/year), 
                      fill_value = (np.min(results_vacuum[1]/year - np.max(results_vacuum[1]/year)) - 1, 1), bounds_error = False)
t_GS_analytical_f = interp1d(results_dynamic_GS_analytical[3], results_dynamic_GS_analytical[1]/year - np.max(results_dynamic_GS_analytical[1]/year))
t_GS_numerical_f = interp1d(results_dynamic_GS_numerical[3], results_dynamic_GS_numerical[1]/year - np.max(results_dynamic_GS_numerical[1]/year))

fig, ax = plt.subplots()
ax.loglog(f_array,  -(t_vacuum_f(f_array) - t_GS_analytical_f(f_array))*365.25, c = 'k', label = 'GS spike, analytical')
ax.loglog(f_array,  -(t_vacuum_f(f_array) - t_GS_numerical_f(f_array))*365.25, c = 'r', label = 'GS spike, numerical')
ax.axvline(f_min, c = 'b', ls = '--')
ax.axvline(f_vacuum_t(-1), c = 'c', ls = '--')
ax.set_ylim(1/(24*60))
ax.set_xlim(f_array[0], f_array[-1])
ax.set_xlabel(r'$f_{GW}$ (Hz)')
ax.set_ylabel(r'$\Delta t_{inspiral}$ (days)')
ax.legend()
plt.show()

# %%

f_vacuum_t = interp1d(results_vacuum[1] - np.max(results_vacuum[1]), results_vacuum[3])
f_GS_t = interp1d(results_dynamic_GS_analytical[1] - np.max(results_dynamic_GS_analytical[1]), results_dynamic_GS_analytical[3])
f_GS_comp_t = interp1d(results_dynamic_GS_numerical[1] - np.max(results_dynamic_GS_numerical[1]), results_dynamic_GS_numerical[3])

t_array = np.linspace(-10, 0, num = 10001)*year

N_cycles_vacuum_t = np.array([quad(f_vacuum_t, t, t_array[-1])[0] for t in t_array])
N_cycles_GS_t = np.array([quad(f_GS_t, t, t_array[-1])[0] for t in t_array])
N_cycles_GS_comp_t = np.array([quad(f_GS_comp_t, t, t_array[-1])[0] for t in t_array])

print(N_cycles_vacuum_t[5000], N_cycles_vacuum_t[5000] - N_cycles_GS_t[5000], N_cycles_vacuum_t[5000] - N_cycles_GS_comp_t[5000])

# %%
