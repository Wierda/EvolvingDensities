#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from copy import copy

from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import root_scalar
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%

G_N = 4.302e-3 #Units of (pc/solar mass) (km/s)^2
c_light = 2.9979e05 #km/s

def monotonic_Bspline(x_array, y_array, increasing = False):

    logy = np.log10(y_array)
    logx = np.log10(x_array)

    N = len(logy)
    dd = 3
    E  = np.eye(N)
    D3 = np.diff(E, n = dd, axis=0)
    D1 = np.diff(E, n = 1, axis=0)
    la = 1E2
    kp = 1E7

    # Monotone smoothing
    ws = np.zeros(N - 1)

    factor = 2*int(increasing) - 1

    for it in range(30):
        Ws      = np.diag(ws * kp)
        mon_cof = np.linalg.solve(E + la * D3.T @ D3 + D1.T @ Ws @ D1, logy)
        ws_new  = (D1 @ mon_cof * factor < 0.0) * 1
        dw      = np.sum(ws != ws_new)
        ws      = ws_new
        if(dw == 0): break  
        print(dw)

    log_interp = interp1d(logx, mon_cof, kind = 'quadratic', fill_value = 0, bounds_error = 0)
    y_smooth_func = lambda x: np.heaviside(x_array[0] - x, 0)*10**mon_cof[0] + \
         np.heaviside(x - x_array[0], 1)*np.heaviside(x_array[-1] - x, 1)*10**log_interp(np.log10(x)) 

    return y_smooth_func


def smooth_Bspline(x_array, y_array, increasing = True):

    logy = np.log10(y_array)
    logx = np.log10(x_array)

    N = len(logy)
    dd = 3
    E  = np.eye(N)
    D3 = np.diff(E, n = dd, axis=0)
    D2 = np.diff(E, n = 2, axis=0)
    la = 1E2
    kp = 1E5

    # Monotone smoothing
    ws = np.zeros(N - 2)

    factor = 2*int(increasing) - 1

    for it in range(30):
        Ws      = np.diag(ws * kp)
        mon_cof = np.linalg.solve(E + la * D3.T @ D3 + D2.T @ Ws @ D2, logy)
        ws_new  = (D2 @ mon_cof * factor < 0.0) * 1
        dw      = np.sum(ws != ws_new)
        ws      = ws_new
        if(dw == 0): break  
        print(dw)

    log_interp = interp1d(logx, mon_cof, kind = 'quadratic', fill_value = 0, bounds_error = 0)
    y_smooth_func = lambda x: np.heaviside(x_array[0] - x, 0)*10**mon_cof[0] + \
         np.heaviside(x - x_array[0], 1)*np.heaviside(x_array[-1] - x, 1)*10**log_interp(np.log10(x)) 

    return y_smooth_func


def ppf_L_i(x: float, E_i: float, r_i: float, psi: callable) -> float:

    return r_i*np.sqrt( 2*(psi(r_i) - E_i) - 2*(psi(r_i) - E_i)*(1 - x)**2 )


class Density:

    def __init__(self, name, rho_func, r_array, N_particles):

        self._name = name

        self._rho_func = rho_func
        self._rho_properties_dict = {'r_array': r_array, 'rho_array': rho_func(r_array)}

        self._truncated = False
        self._r_break = None
        self._phi_break = None
        self._M_tot = None

        self._psi_func = None
        self._psi_properties_dict = {}

        self._phi_func = None
        self._phi_properties_dict = {}

        self._Eddington_func = None
        self._E_sample_func = None
        self._u_sample_func = None

        self._N = N_particles

        tracked_variables = ["r_i", "E_i", 'L_i', "T_r_i", "r_apo_i", "r_peri_i", "E_f", "T_r_f", "r_apo_f", "r_peri_f", "global_weight", "errored", "eaten", "GS eaten"]
        self.output_variables_dataframe = pd.DataFrame(np.zeros((N_particles, len(tracked_variables))), columns = tracked_variables)
        self.output_rseries_dataframe = pd.DataFrame(np.zeros((N_particles, len(r_array))))


    def __call__(self, r):
        return self._rho_func(r)


    def add_external_potential_from_other(self, other):

        psi_old = copy(self._psi_func)
        self._psi_func = lambda r: psi_old(r) + other._psi_func(r)
        for key in self._psi_properties_dict.keys():
            self._psi_properties_dict[key] += other._psi_properties_dict[key]

        phi_old = copy(self._phi_func)
        self._phi_func = lambda r: phi_old(r) + other._phi_func(r)
        for key in self._phi_properties_dict.keys():
            self._phi_properties_dict[key] += other._phi_properties_dict[key]

        mask = self._rho_properties_dict['rho_array'] != 0
        logrho_array = np.log10(self._rho_properties_dict['rho_array'][mask])
        scaled_logrho_array = (logrho_array - np.min(logrho_array))/(np.max(logrho_array) - np.min(logrho_array))
        psi_array = self._psi_properties_dict['psi_array']
        self._E_sample_func = interp1d(scaled_logrho_array, psi_array[mask])
        self._u_sample_func = interp1d(psi_array[mask], scaled_logrho_array)


    def add_external_potential_from_function(self, phi_func, psi_func):

        r_array = self._rho_properties_dict['r_array']

        phi_old = copy(self._phi_func)
        self._phi_func = lambda r: phi_old(r) + phi_func(r)
        self._phi_properties_dict['phi_min'] += phi_func(r_array[0])
        self._phi_properties_dict['phi_max'] += phi_func(r_array[-1])
        self._phi_properties_dict['shift'] += psi_func(r_array[0]) + phi_func(r_array[0])
        self._phi_properties_dict['phi_array'] += phi_func(r_array)

        psi_old = copy(self._psi_func)
        self._psi_func = lambda r: psi_old(r) + psi_func(r)
        self._psi_properties_dict['psi_array'] += psi_func(r_array)
        self._psi_properties_dict['psi_min'] += psi_func(r_array[-1])
        self._psi_properties_dict['psi_max'] += psi_func(r_array[0])
        self._psi_properties_dict['dpsi_min'] += (phi_func(r_array[1]) - phi_func(r_array[0]))/10

        self._Eddington_func = None

        mask = self._rho_properties_dict['rho_array'] != 0
        logrho_array = np.log10(self._rho_properties_dict['rho_array'][mask])
        scaled_logrho_array = (logrho_array - np.min(logrho_array))/(np.max(logrho_array) - np.min(logrho_array))
        psi_array = self._psi_properties_dict['psi_array']
        self._E_sample_func = interp1d(scaled_logrho_array, psi_array[mask])
        self._u_sample_func = interp1d(psi_array[mask], scaled_logrho_array)


    def smoothen_density(self):

        r_array = self._rho_properties_dict['r_array']
        rho_array = self._rho_properties_dict['rho_array']
        rho_mask = rho_array > 0

        smooth_rho_r_func = monotonic_Bspline(r_array[rho_mask], rho_array[rho_mask]) 

        self._rho_func = smooth_rho_r_func
        self._rho_properties_dict['rho_array'] = smooth_rho_r_func(self._rho_properties_dict['r_array'])


    def get_psi(self):
        return self._psi_func
    

    def get_phi(self):
        return self._phi_func


    def get_f_Eddington(self):
        return self._Eddington_func


    def setup_potentials(self):

        r_array = self._rho_properties_dict['r_array']
        phi_array = np.array([self.phi_of_r(r) for r in r_array])
        shift = (phi_array[-1] + G_N*self.M_enclosed(r_array[-1])/r_array[-1])
        psi_array = shift - phi_array

        self._phi_func = UnivariateSpline(r_array, phi_array, k = 3, s = 0)

        self._phi_properties_dict['phi_min'] = np.min(phi_array)
        self._phi_properties_dict['phi_max'] = np.max(phi_array)
        self._phi_properties_dict['shift'] = shift
        self._phi_properties_dict['phi_array'] = phi_array

        self._psi_func = UnivariateSpline(r_array, psi_array, k = 3, s = 0)

        self._psi_properties_dict['psi_min'] = np.min(psi_array)
        self._psi_properties_dict['psi_max'] = np.max(psi_array)
        self._psi_properties_dict['dpsi_min'] = (phi_array[1] - phi_array[0])/10
        self._psi_properties_dict['psi_array'] = psi_array

        mask = self._rho_properties_dict['rho_array'] != 0
        logrho_array = np.log10(self._rho_properties_dict['rho_array'][mask])
        scaled_logrho_array = (logrho_array - np.min(logrho_array))/(np.max(logrho_array) - np.min(logrho_array))
        self._E_sample_func = interp1d(scaled_logrho_array, psi_array[mask])
        self._u_sample_func = interp1d(psi_array[mask], scaled_logrho_array)

        # logr_array = np.log10(r_array)
        # scaled_logr_array = (logr_array - np.min(logr_array))/(np.max(logr_array) - np.min(logr_array))
        # self._E_sample_func = interp1d(scaled_logr_array, psi_array)
        # self._u_sample_func = interp1d(psi_array, scaled_logr_array)


    def phi_of_r(self, r):

        r_min = np.min(self._rho_properties_dict['r_array'])

        r_grid = np.logspace(np.log10(r_min), np.log10(r), num = 1000)

        # Integration doesn't work properly for point mass potentials,
        # so when the density reaches 0, stitch in point mass potential
        if sum(self._rho_func(r_grid) == 0) > 1:
            if not self._truncated:
                print("Density is truncated")
                self._truncated = True
                self._r_break = r_grid[self._rho_func(r_grid) == 0][0]
                self._phi_break = self.phi_of_r(self._r_break)

                self._M_tot = self.M_enclosed(self._r_break)


            return self._phi_break + G_N*self._M_tot/self._r_break - G_N*self._M_tot/r


        try: 
            self._rho_func(0)
            if np.isinf(self._rho_func(0)): raise ValueError("Singularity at r = 0")
            r_grid = np.insert(r_grid, 0, 0)
        except:
            pass

        integrand = lambda x: x*self._rho_func(x)

        first_integral = cumulative_trapezoid(integrand(r_grid), r_grid, initial = 0)

        second_integral = np.trapz(first_integral, r_grid)

        return 4*np.pi*G_N*second_integral/r


    def M_enclosed(self, r):
        r_min = np.min(self._rho_properties_dict['r_array'])

        r_grid = np.logspace(np.log10(r_min), np.log10(r), num = 1000)
        try: 
            self._rho_func(0)
            r_grid = np.insert(r_grid, 0, 0)
        except:
            pass

        integrand = lambda x: x*x*self._rho_func(x)

        result = quad(integrand, 0, r)[0]

        if np.isnan(result) or (result == 0 and r > r_min):
            result = np.trapz(integrand(r_grid), r_grid)

        return 4*np.pi*result
    

    def setup_phase_space(self, smoothen = False):

        if self._psi_func == None:
            print("Calculating initial densities and potentials")
            self.setup_potentials()

        mode = ['psi_logged', 'psi', 'phi']
        eddington_funcs = []

        for i in range(len(mode)):
            print('mode: ', mode[i])
            try:
                self.calculate_f_eddington(mode = mode[i], smoothen = smoothen)
                eddington_funcs.append(self._Eddington_func)
            except Exception as e:
                print('Error message: ', e)
                continue

        if len(eddington_funcs) == 0: raise Exception("The Eddington function could not be computed in any of the modes")
        elif len(eddington_funcs) == 1: return
        r_array = self._rho_properties_dict['r_array']
        rho_array = self._rho_properties_dict['rho_array']
        mask = rho_array != 0
        averages = np.zeros(len(eddington_funcs))
        stds = np.zeros(len(eddington_funcs))
        for i in range(len(eddington_funcs)):
            density_check = np.array([self.reconstruct_density_check(r, eddington_funcs[i]) for r in r_array])
            averages[i] = np.average(density_check[mask]/rho_array[mask])
            stds[i] = np.std(density_check[mask]/rho_array[mask])

        quality = np.abs(1 - averages) + stds
        best = np.argmin(quality)
        print(averages)
        print(stds)
        print(best)
        print(self._Eddington_func == eddington_funcs[best])

        self._Eddington_func = eddington_funcs[best]


    def reconstruct_density_check(self, r, eddington_func):

        E_max = self._psi_func(r)
        psi_array = self._psi_properties_dict['psi_array']
        reversed_psi = psi_array[::-1]

        integrand_grid = eddington_func(reversed_psi)*np.sqrt(2*(E_max - reversed_psi))

        mask_isnan = np.isnan(integrand_grid)
        mask_isinf = np.isinf(integrand_grid)
        mask = np.logical_or(mask_isnan, mask_isinf)
        integrand_grid[mask] = 0

        integral = np.trapz(integrand_grid, reversed_psi)

        return 4*np.pi*integral   
    

    def calculate_f_eddington(self, mode, smoothen = False):
        if mode == 'phi':
            self.calculate_f_Eddington_function_phi(smoothen = smoothen)
        else:
            self.calculate_f_Eddington_function_psi(logged = (mode == 'psi_logged'), smoothen = smoothen)


    def calculate_f_Eddington_function_psi(self, logged: bool = False, smoothen = False):
        
        # psi_max = self._psi_properties_dict['psi_max']
        # psi_min = self._psi_properties_dict['psi_min']
        # dpsi_min = self._psi_properties_dict['dpsi_min']

        if logged:
            d2rho_dpsi2 = self.calculate_d2rho_dpsi2_logged(smoothen = smoothen)
        else:
            d2rho_dpsi2 = self.calculate_d2rho_dpsi2(smoothen = smoothen)

        # energy_array = np.logspace(np.log10(psi_min), np.log10(np.float64(psi_max)), num = 2000)
        # energy_array = np.append(energy_array, np.float64(psi_max) - np.logspace(np.log10(dpsi_min), np.log10(psi_max)//1 - 2, num = 4000))
        # energy_array = np.append(energy_array, np.linspace(psi_min, np.float64(psi_max), num = 4000))
        # energy_array = np.unique(energy_array)

        # energy_array = self._psi_properties_dict['psi_array']

        energy_array = self._E_sample_func(np.linspace(0, 1, num = 10000))

        phase_space = np.array([self.f_Eddington_psi(E, d2rho_dpsi2, logged = logged) for E in energy_array])
        if np.any(phase_space < 0): raise ValueError("De Eddington function is fucking negatief")

        f_eddington_interp = interp1d(energy_array, phase_space, fill_value = 0, bounds_error = False)

        plt.figure()
        plt.loglog(energy_array, phase_space, 'k.')
        plt.show()

        self._Eddington_func = f_eddington_interp
    

    def calculate_f_Eddington_function_phi(self, smoothen = False):

        # psi_max = self._psi_properties_dict['psi_max']
        # psi_min = self._psi_properties_dict['psi_min']
        # dpsi_min = self._psi_properties_dict['dpsi_min']

        d2rho_dphi2 = self.calculate_d2rho_dphi2(smoothen = smoothen)

        # energy_array = np.logspace(np.log10(psi_min), np.log10(np.float64(psi_max)), num = 2000)
        # energy_array = np.append(energy_array, np.float64(psi_max) - np.logspace(np.log10(dpsi_min), np.log10(psi_max)//1 - 2, num = 4000))
        # energy_array = np.append(energy_array, np.linspace(psi_min, np.float64(psi_max), num = 4000))
        # energy_array = np.unique(energy_array)

        # energy_array = self._psi_properties_dict['psi_array']

        energy_array = self._E_sample_func(np.linspace(0, 1, num = 10000))

        phase_space = np.array([self.f_Eddington_phi(E, d2rho_dphi2) for E in energy_array])
        if np.any(phase_space < 0): raise ValueError("De Eddington function is fucking negatief")

        f_eddington_interp = interp1d(energy_array, phase_space, fill_value = 0, bounds_error = False)

        plt.figure()
        plt.loglog(energy_array, phase_space, 'k.')
        plt.show()

        self._Eddington_func = f_eddington_interp


    def f_Eddington_psi(self, E: float, d2rho_dpsi2: callable, logged: bool = False) -> float:

        # psi_max = self._psi_properties_dict['psi_max']
        # psi_min = self._psi_properties_dict['psi_min']
        # dpsi_min = self._psi_properties_dict['dpsi_min']
        
        # psi_array = self._psi_properties_dict['psi_array']

        if logged:
            integrand = lambda psi: d2rho_dpsi2(np.log10(psi))/np.sqrt(E - psi)
        else:
            integrand = lambda psi: d2rho_dpsi2(psi)/np.sqrt(E - psi)

        # psi_grid = np.logspace(np.log10(psi_min), np.log10(E), num = 2000)
        # psi_grid = np.append(psi_grid, E - np.logspace(np.log10(dpsi_min), min(np.log10(psi_max)//1 - 2, 0.99*np.log10(E - 0.99*psi_min)), num = 4000))
        # psi_grid = np.append(psi_grid, np.linspace(psi_min, E, num = 4000))
        # psi_grid = np.unique(psi_grid)

        # psi_grid = psi_array[psi_array < E][::-1]

        u_max = self._u_sample_func(E)
        psi_grid = self._E_sample_func(np.linspace(0, u_max, num = 10000))

        integrand_grid = integrand(psi_grid)
        # integrand_grid[integrand_grid < 0] = 0
        integrand_grid[np.isnan(integrand_grid)] = 0
        integrand_grid[np.isinf(integrand_grid)] = 0
        result = np.trapz(integrand_grid[integrand_grid > 0], psi_grid[integrand_grid > 0])

        f = (result)/(np.pi**2*8**0.5)

        return f


    def f_Eddington_phi(self, E: float, d2rho_dphi2: callable) -> float:

        # psi_max = self._psi_properties_dict['psi_max']
        # psi_min = self._psi_properties_dict['psi_min']
        # dpsi_min = self._psi_properties_dict['dpsi_min']

        shift = self._phi_properties_dict['shift']
        # psi_array = self._psi_properties_dict['psi_array']

        integrand = lambda psi: d2rho_dphi2(shift - psi)/np.sqrt(E - psi)

        # psi_grid = np.logspace(np.log10(psi_min), np.log10(E), num = 2000)
        # psi_grid = np.append(psi_grid, E - np.logspace(np.log10(dpsi_min), min(np.log10(psi_max)//1 - 2, 0.99*np.log10(E - 0.99*psi_min)), num = 4000))
        # psi_grid = np.append(psi_grid, np.linspace(psi_min, E, num = 4000))
        # psi_grid = np.unique(psi_grid)

        # psi_grid = psi_array[psi_array < E][::-1]

        u_max = self._u_sample_func(E)
        psi_grid = self._E_sample_func(np.linspace(0, u_max, num = 10000))

        integrand_grid = integrand(psi_grid)
        integrand_grid[np.isnan(integrand_grid)] = 0
        integrand_grid[np.isinf(integrand_grid)] = 0
        result = np.trapz(integrand_grid[integrand_grid > 0], psi_grid[integrand_grid > 0])

        f = (result)/(np.pi**2*8**0.5)

        return f
    

    def calculate_d2rho_dpsi2_logged(self, smoothen = False) -> callable:

        rho_array = self._rho_properties_dict['rho_array']
        psi_array = self._psi_properties_dict['psi_array']

        # Calculation of rho(psi) using splines. 

        reversed_rho = rho_array[::-1]
        reversed_psi = psi_array[::-1]

        # Solution to the low E issues: interpolation in the loglog scale, then convert back.

        mask = reversed_rho > 0

        if smoothen:
            smooth_rho_psi = smooth_Bspline(reversed_psi[mask], reversed_rho[mask], increasing = True)
            smooth_rho_array = smooth_rho_psi(reversed_psi)
            mask = smooth_rho_array > 0

            logrho_logpsi = UnivariateSpline(np.log10(reversed_psi[mask]), np.log10(smooth_rho_array[mask]), k = 3, s = 0)
        else:
            logrho_logpsi = UnivariateSpline(np.log10(reversed_psi[mask]), np.log10(reversed_rho[mask]), k = 3, s = 0)

        dlogrho_dlogpsi = logrho_logpsi.derivative(n = 1)
        dlogrho_dlogpsi2 = logrho_logpsi.derivative(n = 2)

        d2rho_dpsi2_logged = lambda logpsi: 10**logrho_logpsi(logpsi)/(10**logpsi)**2 * (dlogrho_dlogpsi2(logpsi)/np.log(10) + (dlogrho_dlogpsi(logpsi))**2 - dlogrho_dlogpsi(logpsi))

        return d2rho_dpsi2_logged


    def calculate_d2rho_dpsi2(self, smoothen = False) -> callable:

        rho_array = self._rho_properties_dict['rho_array']
        psi_array = self._psi_properties_dict['psi_array']

        reversed_rho = rho_array[::-1]
        reversed_psi = psi_array[::-1]

        # Calculation of rho(psi) using interp and arrays cause splines can't take float128s. 
        # rho_psi_interp = interp1d(reversed_psi, reversed_rho, fill_value = 'extrapolate')

        if smoothen:
            mask = reversed_rho > 0
            smooth_rho_psi = smooth_Bspline(reversed_psi[mask], reversed_rho[mask], increasing = True)
            smooth_rho_array = smooth_rho_psi(reversed_psi)
            drho_dpsi_interp = interp1d(reversed_psi, np.diff(smooth_rho_array, prepend = reversed_rho[0])/np.diff(reversed_psi, prepend = reversed_psi[0]), 
                                kind = 'linear', fill_value = 'extrapolate')
        else:
            drho_dpsi_interp = interp1d(reversed_psi, np.diff(reversed_rho, prepend = reversed_rho[0])/np.diff(reversed_psi, prepend = reversed_psi[0]), 
                                kind = 'linear', fill_value = 'extrapolate')
            
        drho_dpsi_array = drho_dpsi_interp(reversed_psi)
        d2rho_dpsi2_interp = interp1d(reversed_psi, np.diff(drho_dpsi_array, prepend = drho_dpsi_array[0])/np.diff(reversed_psi, prepend = reversed_psi[0]), 
                                kind = 'linear', fill_value = 'extrapolate')

        return d2rho_dpsi2_interp


    def calculate_d2rho_dphi2(self, smoothen = False) -> callable:

        rho_array = self._rho_properties_dict['rho_array']
        phi_array = self._phi_properties_dict['phi_array']

        if smoothen:
            mask = rho_array > 0
            smooth_rho_phi = smooth_Bspline(phi_array[mask], rho_array[mask], increasing = True)
            smooth_rho_array = smooth_rho_phi(phi_array)

            rho_phi = UnivariateSpline(phi_array, smooth_rho_array, k = 3, s = 0)
        else:
            rho_phi = UnivariateSpline(phi_array, rho_array, k = 3, s = 0)

        
        d2rho_dphi2 = rho_phi.derivative(n = 2)

        return d2rho_dphi2


    def sample_single_orbit(self, i: int, ppf_L_i: callable, ppf_r_i: callable, logr: bool = False) -> int:
        
        psi_initial = self._psi_func
        rho_initial = self._rho_func
        f_eddington = self._Eddington_func
        psi_max = self._psi_properties_dict['psi_max']
        psi_min = self._psi_properties_dict['psi_min']
        dpsi_min = self._psi_properties_dict['dpsi_min']


        if logr: r_i_sample = 10**ppf_r_i(np.random.random())
        else: r_i_sample = ppf_r_i(np.random.random())
        # print("Sampled r_i = {:e}".format(r_i_sample))
        if rho_initial(r_i_sample) == 0: 
            self.sample_single_orbit(i, ppf_L_i, ppf_r_i, logr = logr)
            return

        # calculate ppf_E_i and draw E_i
        E_max = psi_initial(r_i_sample)

        E_initial = np.logspace(np.log10(psi_min), np.log10(E_max), num = 2000)[:-1]
        E_zoom = E_max - np.logspace(np.log10(dpsi_min), min(np.log10(psi_max)//1 - 2, np.log10(psi_max - psi_min)), num = 4000)
        E_initial = np.append(E_initial, E_zoom)
        E_initial = np.append(E_initial, np.linspace(psi_min, E_max, num = 4000))
        E_initial = np.unique(E_initial)

        pdf_E_i = 4*np.pi*f_eddington(E_initial)*np.sqrt(2*(E_max - E_initial))/rho_initial(r_i_sample)
        if np.all(pdf_E_i == 0): # the probability of finding a particle here is effectively zero
            self.sample_single_orbit(i, ppf_L_i, ppf_r_i, logr = logr)
            return
        cdf_E_i = cumulative_trapezoid(pdf_E_i, E_initial, initial = 0)
        # print(cdf_E_i[-1])
        if np.isnan(cdf_E_i[-1]): raise ValueError("De fucking cdf is niet goed want nans, r_i = {:e}".format(r_i_sample))
        if np.all(cdf_E_i == 0): raise ValueError("De fucking cdf is niet goed want all zero, r_i = {:e}, pdf: {}".format(r_i_sample, np.all(pdf_E_i == 0)))
        if cdf_E_i[-1] == 0: raise ValueError("De fucking cdf is niet goed want zero, r_i = {:e}, pdf: {}".format(r_i_sample, np.all(pdf_E_i == 0)))
        if np.any(cdf_E_i < 0): raise ValueError("De fucking cdf is niet goed want negatief, r_i = {:e}".format(r_i_sample))

        ppf_E_i = interp1d(cdf_E_i/cdf_E_i[-1], E_initial)
        E_i_sample = ppf_E_i(np.random.random())
        # print("Sampled E_i = {:e}".format(E_i_sample))
        if np.isnan(E_i_sample): raise ValueError("De fucking cdf is anders niet goed, r_i = {:e}".format(r_i_sample))

        L_i_sample = ppf_L_i(np.random.random(), E_i_sample, r_i_sample, psi_initial)

        self.output_variables_dataframe.loc[i, 'r_i'] = r_i_sample
        self.output_variables_dataframe.loc[i, 'E_i'] = E_i_sample
        self.output_variables_dataframe.loc[i, 'L_i'] = L_i_sample
    

    def radial_orbital_time(self, i, orbit_vr_squared_analytical, r_S = 0):

        r_array = self._rho_properties_dict['r_array']
        
        v_r_squared_array = orbit_vr_squared_analytical(r_array)

        r_real = r_array[v_r_squared_array >= 0]

        if len(r_real) == 0: # No roots within r_array
            self.output_variables_dataframe.loc[i, 'errored'] = 1
            return
        elif r_real[0] == r_array[0]:  # Only one root at large r (small r root cannot be resolved)
            self.output_variables_dataframe.loc[i, 'errored'] = 2
            return
        elif r_real[-1] == r_array[-1]: # Only one root at small r (large r root outside of r_array)
            self.output_variables_dataframe.loc[i, 'errored'] = 3
            return 
        else: 
            mid = r_real[len(r_real)//2]
            r_periapsis = root_scalar(orbit_vr_squared_analytical, bracket = [r_array[0], mid]).root
            r_apoapsis = root_scalar(orbit_vr_squared_analytical, bracket = [mid, r_array[-1]]).root

        if r_periapsis <= 4*r_S: self.output_variables_dataframe.loc[i, 'eaten'] = 1

        # Break the integrand into two different ones to deal with the peri- and apoapsis

        # r = r_peri + t^2
        integrand_small_r = lambda t: 2*t/np.sqrt(orbit_vr_squared_analytical(r_periapsis + t**2))

        # r = r_apo - s^2
        integrand_large_r = lambda s: 2*s/np.sqrt(orbit_vr_squared_analytical(r_apoapsis - s**2))

        r_mean = np.exp((np.log(r_periapsis) + np.log(r_apoapsis))/2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T_r_small_r = quad(integrand_small_r, 1E-10, np.sqrt(r_mean - r_periapsis))[0]
            T_r_large_r = quad(integrand_large_r, 1E-10, np.sqrt(r_apoapsis - r_mean))[0]

        T_r = np.where(np.isnan(T_r_small_r), 0, T_r_small_r) + T_r_large_r # s * pc/km 

        return T_r, r_periapsis, r_apoapsis


    def calculate_orbital_time_distribution(self, logr: bool = True, figures = False, cmap = 'viridis'):

        # If the dataframe has already been (partially) filled,
        # skip all the setup and go straight to the calculation
        if ~np.all(self.output_variables_dataframe == 0):

            for i in range(self._N):
                if i%100 == 0: print("Orbit: ", i)

                # if self.output_variables_dataframe.loc[i, 'errored'] > 0: continue

                E_i = self.output_variables_dataframe.loc[i, 'E_i']
                L_i = self.output_variables_dataframe.loc[i, 'L_i']
                orbit_vr_squared_analytical = lambda r: 2*(self._psi_func(r) - E_i) - (L_i**2/r**2)

                T_r_output = self.radial_orbital_time(i, orbit_vr_squared_analytical)
                # If radial orbital time errors the output is None
                if T_r_output == None: continue
                T_r, r_peri, r_apo = T_r_output
                self.output_variables_dataframe.loc[i, 'T_r_i'] = T_r
                self.output_variables_dataframe.loc[i, 'r_peri_i'] = r_peri
                self.output_variables_dataframe.loc[i, 'r_apo_i'] = r_apo

            if figures:
                r_apo_i_samples = self.output_variables_dataframe['r_apo_i']
                r_peri_i_samples = self.output_variables_dataframe['r_peri_i']

                hist_apo_peri_i, x_edges, y_edges = np.histogram2d(r_apo_i_samples, r_peri_i_samples, bins = r_array[::10], 
                                            density = False)
                hist_apo_peri_i_masked = np.ma.masked_equal(hist_apo_peri_i.T, 0)

                plt.figure()
                plt.pcolormesh(x_edges, y_edges, hist_apo_peri_i_masked, 
                                        cmap = cmap, norm = LogNorm(vmin = hist_apo_peri_i_masked.min(), vmax = hist_apo_peri_i_masked.max()))
                plt.plot(r_array, r_array, c = 'r')
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel(r'$r_{apo}$')
                plt.ylabel(r'$r_{peri}$')
                plt.colorbar()
                plt.show()
            return

        r_array = self._rho_properties_dict['r_array']

        # calculate ppf_r_i (because it is unchanging)
        print("Calculating the ppf for p(r_i)")
        M_tot = self.M_enclosed(r_array[-1])
        pdf_r_i = lambda r: 4*np.pi*r**2*self._rho_func(r)/M_tot

        r_initial = np.logspace(max(-10, np.log10(r_array.min())), np.log10(r_array.max()), num = 10000)

        if logr:
            ppf_r_i = lambda x: np.log10(r_initial[-1]/r_initial[0])*x + np.log10(r_initial[0])
        else:
            pdf_r_array = pdf_r_i(r_initial)
            cdf_r_i = cumulative_trapezoid(pdf_r_array, r_initial, initial = 0)
            ppf_r_i = interp1d(cdf_r_i/cdf_r_i[-1], r_initial)

        # calculate f_Eddington function
        if self._Eddington_func == None:
            print("Calculating Eddington function")
            self.setup_phase_space()

        if figures:
            psi_max = self._psi_properties_dict['psi_max']
            psi_min = self._psi_properties_dict['psi_min']
            dpsi_min = self._psi_properties_dict['dpsi_min']

            E_initial = np.logspace(np.log10(psi_min), np.log10(psi_max), num = 1000)
            E_initial = np.append(E_initial, psi_max - np.logspace(np.log10(dpsi_min), max(np.log10(psi_max)//1 - 2, 0), num = 2000))
            E_initial = np.append(E_initial, np.linspace(psi_min, psi_max, num = 1000))
            E_initial = np.unique(E_initial)

            plt.figure()
            plt.loglog(E_initial, self._Eddington_func(E_initial), color = 'r')
            plt.xlabel(r'$\mathcal{E}$')
            plt.ylabel(r'$f(\mathcal{E})$')
            plt.show()


        for i in range(self._N):
            if i%100 == 0: print("Orbit: ", i)

            self.sample_single_orbit(i, ppf_L_i, ppf_r_i, logr = logr)
            
            r_i = self.output_variables_dataframe.loc[i, 'r_i']
            E_i = self.output_variables_dataframe.loc[i, 'E_i']
            L_i = self.output_variables_dataframe.loc[i, 'L_i']

            if logr:
                self.output_variables_dataframe.loc[i, 'global_weight'] = pdf_r_i(r_i)*r_i
            else:
                self.output_variables_dataframe.loc[i, 'global_weight'] = 1

            orbit_vr_squared_analytical = lambda r: 2*(self._psi_func(r) - E_i) - (L_i**2/r**2)

            T_r_output = self.radial_orbital_time(i, orbit_vr_squared_analytical)
            # If radial orbital time errors the output is None
            if T_r_output == None: continue
            T_r, r_peri, r_apo = T_r_output
            self.output_variables_dataframe.loc[i, 'T_r_i'] = T_r
            self.output_variables_dataframe.loc[i, 'r_peri_i'] = r_peri
            self.output_variables_dataframe.loc[i, 'r_apo_i'] = r_apo

        if figures:
            r_apo_i_samples = self.output_variables_dataframe['r_apo_i']
            r_peri_i_samples = self.output_variables_dataframe['r_peri_i']

            hist_apo_peri_i, x_edges, y_edges = np.histogram2d(r_apo_i_samples, r_peri_i_samples, bins = r_array[::10], 
                                        density = False)
            hist_apo_peri_i_masked = np.ma.masked_equal(hist_apo_peri_i.T, 0)

            plt.figure()
            plt.pcolormesh(x_edges, y_edges, hist_apo_peri_i_masked, 
                                    cmap = cmap, norm = LogNorm(vmin = hist_apo_peri_i_masked.min(), vmax = hist_apo_peri_i_masked.max()))
            plt.plot(r_array, r_array, c = 'r')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$r_{apo}$')
            plt.ylabel(r'$r_{peri}$')
            plt.colorbar()
            plt.show()


    def integrate_probability(self, i, orbit_vr_squared_analytical, r_S = 0):

        r_array = self._rho_properties_dict['r_array']

        T_r_output = self.radial_orbital_time(i, orbit_vr_squared_analytical, r_S = r_S)
        # If radial orbital time errors the output is None
        if T_r_output == None: return
        T_r, r_peri, r_apo = T_r_output
        self.output_variables_dataframe.loc[i, 'T_r_f'] = T_r
        self.output_variables_dataframe.loc[i, 'r_peri_f'] = r_peri
        self.output_variables_dataframe.loc[i, 'r_apo_f'] = r_apo

        # p(r | r_i, E_i, L_i) calculations
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message = 'invalid value encountered')
            p_r = (1/T_r * 1/np.sqrt(orbit_vr_squared_analytical(r_array)))

        nan_mask = np.isnan(p_r)
        if len(p_r[~nan_mask]) == 0: return

        p_r[nan_mask] = 0
        # cdf = cumulative_trapezoid(p_r, r_array, initial = 0)

        self.output_rseries_dataframe.loc[i] = p_r # /cdf[-1]


    def non_adiabatic_growth(self, delta_psi: callable, r_S = 0, logr: bool = True, inplace = False, figures = False, cmap = 'viridis') -> int:

        """
        Calculate N orbits for orbits from the initial potential, adjusting to the final potential,
        at time steps in t_array.
        """

        r_array = self._rho_properties_dict['r_array']

        # calculate rho and psi arrays cause they useful
        if self._psi_func == None:
            print("Calculating initial densities and potentials")
            self.setup_potentials()

        psi_array = self._psi_properties_dict['psi_array']

        psi_final_array = psi_array + delta_psi(r_array)
        psi_final = UnivariateSpline(r_array, psi_final_array, k = 3, s = 0)

        if figures:
            plt.figure()
            plt.loglog(r_array, psi_array, color = 'k', label = r'$\psi_i$')
            plt.loglog(r_array, psi_final_array, color = 'r', label = r'$\psi_f$')
            plt.xlabel(r'$r$')
            plt.ylabel(r'$\psi(r)$')
            plt.legend()
            plt.show()

        # If the dataframe has already been (partially) filled,
        # skip all the setup and go straight to the calculation
        if ~np.all(self.output_variables_dataframe == 0):
            print("Beginning orbit calculations")
            for i in range(self._N):
                if i%100 == 0: print("Orbit: ", i)
                
                # if self.output_variables_dataframe.loc[i, 'errored'] > 0: continue

                r_i = self.output_variables_dataframe.loc[i, 'r_i']
                E_i = self.output_variables_dataframe.loc[i, 'E_i']
                L_i = self.output_variables_dataframe.loc[i, 'L_i']

                orbit_vr_squared_analytical = lambda r: 2*(psi_final(r) - E_i \
                    - delta_psi(r_i)) - (L_i**2/r**2)

                self.output_variables_dataframe.loc[i, 'E_f'] = E_i + delta_psi(r_i)

                if L_i <= 2*r_S*c_light: self.output_variables_dataframe.loc[i, 'GS eaten'] = 1
                if np.all(psi_final(r_array)*(1-4*r_S/r_array) < E_i + delta_psi(r_i)):
                    self.output_variables_dataframe.loc[i, 'GS eaten'] += 1
                    self.output_variables_dataframe.loc[i, 'GS eaten'] *= 2

                self.integrate_probability(i, orbit_vr_squared_analytical, r_S = r_S)

            if figures:
                r_apo_f_samples = self.output_variables_dataframe['r_apo_f']
                r_peri_f_samples = self.output_variables_dataframe['r_peri_f']

                hist_apo_peri_f, x_edges, y_edges = np.histogram2d(r_apo_f_samples, r_peri_f_samples, bins = r_array[::10], 
                                            density = False)
                hist_apo_peri_f_masked = np.ma.masked_equal(hist_apo_peri_f.T, 0)

                plt.figure()
                plt.pcolormesh(x_edges, y_edges, hist_apo_peri_f_masked, 
                                        cmap = cmap, norm = LogNorm(vmin = hist_apo_peri_f_masked.min(), vmax = hist_apo_peri_f_masked.max()))
                plt.plot(r_array, r_array, c = 'r')
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel(r'$r_{apo}$')
                plt.ylabel(r'$r_{peri}$')
                plt.colorbar()
                plt.show()

            M_tot = self.M_enclosed(r_array[-1])
            mask = self.output_variables_dataframe['eaten'] == 0
            p_r_marginal = np.average(self.output_rseries_dataframe.loc[mask], axis = 0, weights = self.output_variables_dataframe.loc[mask, 'global_weight'])
            norm = np.trapz(p_r_marginal, r_array)
            rho_final_array = M_tot*(p_r_marginal/norm)/(4*np.pi*r_array**2)
            rho_final = UnivariateSpline(r_array, rho_final_array, k = 3, s = 0)

            if figures:
                rho_check_eddington = np.array([self.reconstruct_density_check(r, self._Eddington_func) for r in r_array])
                bins = r_array[::10]
                p_r_t_0, edges = np.histogram(self.output_variables_dataframe['r_i'], bins = bins, density = True, weights = self.output_variables_dataframe['global_weight'])
                rho_sampled = M_tot*p_r_t_0/(4*np.pi*r_array[5::10]**2)

                fig, ax = plt.subplots(figsize = (4, 5))
                ax.plot(r_array, self._rho_properties_dict['rho_array'], c = 'k', label = r'$\rho_i$')
                ax.plot(r_array[5::10], rho_sampled, c = 'c', label = r'$\rho_{sampled}$')
                ax.plot(r_array, rho_check_eddington, c = 'b', label = r'$\rho_{check}$')
                ax.plot(r_array, rho_final_array,  c = 'r', label = r'$\rho_f$')
                ax.set_ylim(bottom = 1E-11)
                ax.set_xlim(left = r_array[0])
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_ylabel(r'$\rho$')
                ax.legend()
                divider = make_axes_locatable(ax)
                ax_ratio = divider.append_axes("bottom", 1.2, pad=0.1, sharex=ax)
                ax.xaxis.set_tick_params(labelbottom=False)
                ax_ratio.axhline(1, c = 'k')
                ax_ratio.plot(r_array[5::10], rho_sampled/self._rho_properties_dict['rho_array'][5::10], c = 'c')
                ax_ratio.plot(r_array, rho_check_eddington/self._rho_properties_dict['rho_array'], c = 'b')
                ax_ratio.set_ylim(-0.1, 2.1)
                ax_ratio.set_xlabel(r'$r$ (pc)')
                ax_ratio.set_ylabel(r'$\rho/\rho_o$')
                plt.show()

            if inplace:
                # Update the density instance with either new values when available, and reset if not
                self._rho_func = rho_final
                self._rho_properties_dict['rho_array'] = rho_final_array
                
                self._psi_func = None
                self._psi_properties_dict = None

                self._phi_func = None
                self._phi_properties_dict = None

                self._Eddington_func = None

                self.output_rseries_dataframe.where(self.output_rseries_dataframe == 0, 0, inplace = True)
                self.output_variables_dataframe.where(self.output_variables_dataframe == 0, 0, inplace = True)

            else:
                return Density(self._name+'nonadiabatic', rho_final, r_array, self._N)

        # calculate f_Eddington function
        if self._Eddington_func == None:
            print("Calculating Eddington function")
            self.setup_phase_space()

        if figures:

            psi_max = self._psi_properties_dict['psi_max']
            psi_min = self._psi_properties_dict['psi_min']
            dpsi_min = self._psi_properties_dict['dpsi_min']

            E_initial = np.logspace(np.log10(psi_min), np.log10(psi_max), num = 1000)
            E_initial = np.append(E_initial, psi_max - np.logspace(np.log10(dpsi_min), max(np.log10(psi_max)//1 - 2, 0), num = 2000))
            E_initial = np.append(E_initial, np.linspace(psi_min, psi_max, num = 1000))
            E_initial = np.unique(E_initial)

            plt.figure()
            plt.loglog(E_initial, self._Eddington_func(E_initial), color = 'r')
            plt.xlabel(r'$\mathcal{E}$')
            plt.ylabel(r'$f(\mathcal{E})$')
            plt.show()

        # calculate ppf_r_i (because it is unchanging)
        print("Calculating the ppf for p(r_i)")
        M_tot = self.M_enclosed(r_array[-1])
        pdf_r_i = lambda r: 4*np.pi*r**2*self._rho_func(r)/M_tot

        r_initial = np.logspace(max(-10, np.log10(r_array.min())), np.log10(r_array.max()), num = 10000)

        if logr:
            ppf_r_i = lambda x: np.log10(r_initial[-1]/r_initial[0])*x + np.log10(r_initial[0])
        else:
            pdf_r_array = pdf_r_i(r_initial)
            cdf_r_i = cumulative_trapezoid(pdf_r_array, r_initial, initial = 0)
            ppf_r_i = interp1d(cdf_r_i/cdf_r_i[-1], r_initial)

        print("Beginning orbit calculations")
        for i in range(self._N):
            if i%100 == 0: print("Orbit: ", i)

            if np.all(self.output_variables_dataframe.loc[i] == 0):
                self.sample_single_orbit(i, ppf_L_i, ppf_r_i, logr = logr)
            
            r_i = self.output_variables_dataframe.loc[i, 'r_i']
            E_i = self.output_variables_dataframe.loc[i, 'E_i']
            L_i = self.output_variables_dataframe.loc[i, 'L_i']

            if logr:
                self.output_variables_dataframe.loc[i, 'global_weight'] = pdf_r_i(r_i)*r_i
            else:
                self.output_variables_dataframe.loc[i, 'global_weight'] = 1

            orbit_vr_squared_analytical = lambda r: 2*(psi_final(r) - E_i \
                - delta_psi(r_i)) - (L_i**2/r**2)
            
            self.output_variables_dataframe.loc[i, 'E_f'] = E_i + delta_psi(r_i)

            if L_i <= 2*r_S*c_light: self.output_variables_dataframe.loc[i, 'GS eaten'] = 1
            if np.all(psi_final(r_array)*(1-4*r_S/r_array) < E_i + delta_psi(r_i)):
                self.output_variables_dataframe.loc[i, 'GS eaten'] += 1
                self.output_variables_dataframe.loc[i, 'GS eaten'] *= 2

            self.integrate_probability(i, orbit_vr_squared_analytical, r_S = r_S)

        if figures:
            r_apo_f_samples = self.output_variables_dataframe['r_apo_f']
            r_peri_f_samples = self.output_variables_dataframe['r_peri_f']

            hist_apo_peri_f, x_edges, y_edges = np.histogram2d(r_apo_f_samples, r_peri_f_samples, bins = r_array[::10], 
                                        density = False)
            hist_apo_peri_f_masked = np.ma.masked_equal(hist_apo_peri_f.T, 0)

            plt.figure()
            plt.pcolormesh(x_edges, y_edges, hist_apo_peri_f_masked, 
                                    cmap = cmap, norm = LogNorm(vmin = hist_apo_peri_f_masked.min(), vmax = hist_apo_peri_f_masked.max()))
            plt.plot(r_array, r_array, c = 'r')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$r_{apo}$')
            plt.ylabel(r'$r_{peri}$')
            plt.colorbar()
            plt.show()

        mask = self.output_variables_dataframe['eaten'] == 0
        p_r_marginal = np.average(self.output_rseries_dataframe.loc[mask], axis = 0, weights = self.output_variables_dataframe.loc[mask, 'global_weight'])
        norm = np.trapz(p_r_marginal, r_array)
        rho_final_array = M_tot*(p_r_marginal/norm)/(4*np.pi*r_array**2)
        rho_final = UnivariateSpline(r_array, rho_final_array, k = 3, s = 0)

        if figures:
            rho_check_eddington = np.array([self.reconstruct_density_check(r, self._Eddington_func) for r in r_array])
            bins = r_array[::10]
            p_r_t_0, edges = np.histogram(self.output_variables_dataframe['r_i'], bins = bins, density = True, weights = self.output_variables_dataframe['global_weight'])
            rho_sampled = M_tot*p_r_t_0/(4*np.pi*r_array[5::10]**2)

            fig, ax = plt.subplots(figsize = (4, 5))
            ax.plot(r_array, self._rho_properties_dict['rho_array'], c = 'k', label = r'$\rho_i$')
            ax.plot(r_array[5::10], rho_sampled, c = 'c', label = r'$\rho_{sampled}$')
            ax.plot(r_array, rho_check_eddington, c = 'b', label = r'$\rho_{check}$')
            ax.plot(r_array, rho_final_array,  c = 'r', label = r'$\rho_f$')
            ax.set_ylim(bottom = 1E-11)
            ax.set_xlim(left = r_array[0])
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r'$\rho$')
            ax.legend()
            divider = make_axes_locatable(ax)
            ax_ratio = divider.append_axes("bottom", 1.2, pad=0.1, sharex=ax)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax_ratio.axhline(1, c = 'k')
            ax_ratio.plot(r_array[5::10], rho_sampled/self._rho_properties_dict['rho_array'][5::10], c = 'c')
            ax_ratio.plot(r_array, rho_check_eddington/self._rho_properties_dict['rho_array'], c = 'b')
            ax_ratio.set_ylim(-0.1, 2.1)
            ax_ratio.set_xlabel(r'$r$ (pc)')
            ax_ratio.set_ylabel(r'$\rho/\rho_o$')
            plt.show()

        if inplace:
            # Update the density instance with either new values when available, and reset if not
            self._rho_func = rho_final
            self._rho_properties_dict['rho_array'] = rho_final_array
            
            self._psi_func = None
            self._psi_properties_dict = None

            self._phi_func = None
            self._phi_properties_dict = None

            self._Eddington_func = None

            self.output_rseries_dataframe.where(self.output_rseries_dataframe == 0, 0, inplace = True)
            self.output_variables_dataframe.where(self.output_variables_dataframe == 0, 0, inplace = True)

        else:
            return Density(self._name+'nonadiabatic', rho_final, r_array, self._N)


    # def radial_action(self, E, L, psi_func):

    #     integrand = lambda e, l, r, psi: np.sqrt(2*psi - 2*e - l**2/r**2)

    #     r_range = np.logspace(-8, 5, num = 500)
    #     psi_range = psi_func(r_range)

    #     mask_E = E < psi_range
    #     if np.all(~mask_E): return 1E-30
    #     mask_L = np.full_like(mask_E, False)
    #     mask_L[mask_E] = L < r_range[mask_E]*np.sqrt(2*(psi_range[mask_E] - E))
    #     if np.all(~mask_L): return 1E-30

    #     integrand_grid = np.zeros_like(r_range)
    #     integrand_grid[mask_L] = integrand(E, L, r_range[mask_L], psi_range[mask_L])

    #     result = np.trapz(integrand_grid, r_range)
    #     return 2*result

    def radial_action(self, E, L, psi_func):

        r_array = self._rho_properties_dict['r_array']
        
        v_r_squared = lambda r: 2*psi_func(r) - 2*E - L**2/r**2
        v_r_squared_array = v_r_squared(r_array)

        r_real = r_array[v_r_squared_array >= 0]

        if len(r_real) == 0: # No roots within r_array
            return 1E-30
        elif len(r_real) == 1:
            r_refined = np.linspace(r_real[0]*0.9, r_real[0]*1.1, num = 100)
            v_r_squared_refined = v_r_squared(r_refined)

            r_real_refined = r_refined[v_r_squared_refined >= 0]
            if len(r_real_refined) < 2: return 1E-30
            I_r = np.trapz(np.sqrt(v_r_squared(r_real_refined)), r_real_refined)
            return I_r

        # elif r_real[0] == r_array[0]:  # Only one root at large r (small r root cannot be resolved)
        #     return 1E-30
        # elif r_real[-1] == r_array[-1]: # Only one root at small r (large r root outside of r_array)
        #     return 1E-30

        I_r = np.trapz(np.sqrt(v_r_squared(r_real)), r_real)

        return I_r


    def reconstruct_density(self, r, E_mesh, L_mesh, phase_space_mesh, psi_func, r_S = 0):

        E_max = psi_func(r)

        integrand_grid = phase_space_mesh * L_mesh/(2*E_max - 2*E_mesh - (L_mesh/r)**2)**0.5

        mask_BH_E = E_mesh > psi_func(r)*(1 - 4*r_S/r)

        mask_isnan = np.isnan(integrand_grid)
        mask_isinf = np.isinf(integrand_grid)
        mask = np.logical_or(mask_isnan, mask_isinf)
        mask = np.logical_or(mask, mask_BH_E)
        integrand_grid[mask] = 0

        inner_integral = np.trapz(integrand_grid, L_mesh, axis = 1)

        inner_integral[np.isnan(inner_integral)] = 0
        outer_integral = np.trapz(inner_integral, E_mesh[:, 0])

        return 4*np.pi*np.abs(outer_integral)/r**2


    def adiabatic_growth(self, delta_psi, r_S = 0, refinement = 5, inplace = False, figures = False, cmap = 'viridis'):

        r_array = self._rho_properties_dict['r_array']

        if self._psi_func == None:
            print("Calculating initial densities and potentials")
            self.setup_potentials()

        psi_array = self._psi_properties_dict['psi_array']

        psi_final_array = psi_array + delta_psi(r_array)
        psi_final = UnivariateSpline(r_array, psi_final_array, k = 3, s = 0)

        if figures:
            plt.figure()
            plt.loglog(r_array, psi_array, color = 'k', label = r'$\psi_i$')
            plt.loglog(r_array, psi_final_array, color = 'r', label = r'$\psi_f$')
            plt.xlabel(r'$r$ (pc)')
            plt.ylabel(r'$\psi(r)$ ')
            plt.legend()
            plt.show()

        if self._Eddington_func == None:
            print("Calculating Eddington function")
            self.setup_phase_space()

        psi_max = self._psi_properties_dict['psi_max']
        psi_min = self._psi_properties_dict['psi_min']
        dpsi_min = self._psi_properties_dict['dpsi_min']

        E_initial = np.logspace(np.log10(psi_min), np.log10(psi_max), num = 1000)
        E_initial = np.append(E_initial, psi_max - np.logspace(np.log10(dpsi_min), min(np.log10(psi_max)//1 - 2, np.log10(psi_max - psi_min)), num = 2000))
        E_initial = np.append(E_initial, np.linspace(psi_min, psi_max, num = 1000))
        E_initial = np.unique(E_initial)

        # E_initial = self._E_sample_func(np.linspace(0, 1, num = 10000))

        if figures:
            plt.figure()
            plt.loglog(E_initial, self._Eddington_func(E_initial), color = 'r')
            plt.xlabel(r'${\mathcal{E}}$')
            plt.ylabel(r'$f(\mathcal{E})$')
            plt.show()

        psi_final_array = psi_array + delta_psi(r_array)
        psi_final_min = np.min(psi_final_array)
        psi_final_max = np.max(psi_final_array)
        psi_final = UnivariateSpline(r_array, psi_final_array, k = 3, s = 0)

        L_max = np.max(r_array*np.sqrt(psi_final_array))
        L_min = 2*c_light*r_S
        L_initial = np.logspace(np.log10(max(L_min, 1E-10)), np.log10(L_max), num = 500)
        L_initial = np.append(L_initial, np.linspace(0, L_max, num = 500))
        L_initial = np.unique(L_initial)
        Ev_initial, Lv_initial = np.meshgrid(E_initial, L_initial, indexing = 'ij')

        print("Calculating initial radial action")
        Ir_initial = np.array([[self.radial_action(e, l, self._psi_func) for l in L_initial] for e in E_initial])

        mask = Ir_initial[1:] > 1E-30
        Ir_initial_diff = np.log10(Ir_initial[:-1][mask]/Ir_initial[1:][mask])

        threshold = np.log10(np.max(Ir_initial)/np.min(Ir_initial[Ir_initial > 1E-30]))/refinement

        if figures:
            plt.figure()
            plt.scatter(Ir_initial[Ir_initial > 1E-30].flatten(), Lv_initial[Ir_initial > 1E-30].flatten(), 
                     c = 'k', s = 1, label = 'Initial')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$I_r$')
            plt.ylabel(r'$L$')
            plt.legend(loc = 2)
            plt.show()

            plt.figure()
            plt.plot(Ev_initial[1:][mask], Ir_initial_diff, 'k.')
            plt.xlabel('E')
            plt.xscale('log')
            plt.show()

            print(threshold)

        E_initial_old = np.copy(E_initial)

        n = 0
        while True:
            if n >= 10: break
            if np.all(Ir_initial_diff < threshold): break

            E_initial_undersampled = np.unique(Ev_initial[1:][mask][Ir_initial_diff >= threshold])
            index_undersampled = np.arange(len(E_initial_old))[np.isin(E_initial_old, E_initial_undersampled)]

            E_initial_new = np.copy(E_initial_old)

            for i in range(len(index_undersampled)):
                ind = index_undersampled[i]

                E_extra = np.logspace(np.log10(E_initial_old[ind - 1]), np.log10(E_initial_old[ind]), num = 12)
                Ir_extra = np.array([[self.radial_action(e, l, self._psi_func) for l in L_initial] for e in E_extra[1:-1]])

                E_initial_new = np.insert(E_initial_new, ind + i*10, E_extra[1:-1])
                Ir_initial = np.insert(Ir_initial, ind + i*10, Ir_extra, axis = 0)

            mask = Ir_initial[1:] > 1E-30
            Ir_initial_diff = np.log10(Ir_initial[:-1][mask]/Ir_initial[1:][mask])

            Ev_initial, Lv_initial = np.meshgrid(E_initial_new, L_initial, indexing = 'ij')
            E_initial_old = np.copy(E_initial_new)
            n += 1

            if figures:
                plt.figure()
                plt.plot(Ev_initial[1:][mask], Ir_initial_diff, 'k.')
                plt.xlabel('E')
                plt.xscale('log')
                plt.show()

                plt.figure()
                plt.scatter(Ir_initial[Ir_initial > 1E-30].flatten(), Lv_initial[Ir_initial > 1E-30].flatten(), 
                     c = 'k', s = 1, label = 'Initial')
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel(r'$I_r$')
                plt.ylabel(r'$L$')
                plt.legend(loc = 2)
                plt.show()

        if figures:

            Ir_initial_masked = np.ma.masked_values(Ir_initial, 1E-30)
            plt.figure()
            plt.pcolormesh(psi_array[0] - Ev_initial[:-1, 1:], Lv_initial[:-1, 1:], Ir_initial_masked[:-1, 1:], 
                        cmap = cmap, norm = LogNorm(vmin = Ir_initial_masked.min(), vmax = Ir_initial_masked.max()))
            plt.xlim(left = np.ma.masked_where(Ir_initial[:-1, 1:] <= 1E-30, psi_array[0] - Ev_initial[:-1, 1:]).min())
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\Psi_{i, max} - \mathcal{E}_i$')
            plt.ylabel(r'$L_i$')
            plt.colorbar(label = r'$I_r(\mathcal{E}_i, L_i)$')
            plt.show()


        switch_index = np.argmin(np.abs(psi_final_array - 2*psi_array))

        E_final = np.logspace(np.log10(psi_final_min), np.log10(psi_final_max), num = 200)
        E_final = np.append(E_final, psi_final_max - np.logspace(np.log10(dpsi_min), min(np.log10(psi_max)//1 - 2, np.log10(psi_max - psi_min)), num = 200))
        E_final = np.append(E_final, np.logspace(np.log10(psi_final_array[switch_index]) - 1, np.log10(psi_final_array[switch_index]), num = 200))
        E_final = np.append(E_final, np.linspace(psi_final_min, psi_final_max, num = 200))
        E_final = np.unique(E_final)

        Ev_final, Lv_final = np.meshgrid(E_final, L_initial, indexing = 'ij')

        print("Calculating final radial action")
        Ir_final = np.array([[self.radial_action(e, l, psi_final) for l in L_initial] for e in E_final])

        mask = Ir_final[1:] > 1E-30
        Ir_final_diff = np.log10(Ir_final[:-1][mask]/Ir_final[1:][mask])

        threshold = np.log10(np.max(Ir_final)/np.min(Ir_final[Ir_final > 1E-30]))/refinement

        if figures:
            plt.figure()
            plt.scatter(Ir_initial[Ir_initial > 1E-30].flatten(), Lv_initial[Ir_initial > 1E-30].flatten(), 
                     c = 'k', s = 1, label = 'Initial')
            plt.scatter(Ir_final[Ir_final > 1E-30].flatten(), Lv_final[Ir_final > 1E-30].flatten(), 
                     c = 'r', s = 1, label = 'Final')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$I_r$')
            plt.ylabel(r'$L$')
            plt.legend(loc = 2)
            plt.show()

            plt.figure()
            plt.plot(Ev_final[1:][mask], Ir_final_diff, 'k.')
            plt.xlabel('E')
            plt.xscale('log')
            plt.show()

            print(threshold)

        E_final_old = np.copy(E_final)

        n = 0
        while True:
            if n >= 10: break
            if np.all(Ir_final_diff < threshold): break

            E_final_undersampled = np.unique(Ev_final[1:][mask][Ir_final_diff >= threshold])
            index_undersampled = np.arange(len(E_final_old))[np.isin(E_final_old, E_final_undersampled)]

            E_final_new = np.copy(E_final_old)

            for i in range(len(index_undersampled)):
                ind = index_undersampled[i]

                E_extra = np.logspace(np.log10(E_final_old[ind - 1]), np.log10(E_final_old[ind]), num = 12)
                Ir_extra = np.array([[self.radial_action(e, l, psi_final) for l in L_initial] for e in E_extra[1:-1]])

                E_final_new = np.insert(E_final_new, ind + i*10, E_extra[1:-1])
                Ir_final = np.insert(Ir_final, ind + i*10, Ir_extra, axis = 0)

            mask = Ir_final[1:] > 1E-30
            Ir_final_diff = np.log10(Ir_final[:-1][mask]/Ir_final[1:][mask])

            Ev_final, Lv_final = np.meshgrid(E_final_new, L_initial, indexing = 'ij')
            E_final_old = np.copy(E_final_new)
            n += 1

            if figures:
                plt.figure()
                plt.plot(Ev_final[1:][mask], Ir_final_diff, 'k.')
                plt.xlabel('E')
                plt.xscale('log')
                plt.show()

                plt.figure()
                plt.scatter(Ir_initial[Ir_initial > 1E-30].flatten(), Lv_initial[Ir_initial > 1E-30].flatten(), 
                     c = 'k', s = 1, label = 'Initial')
                plt.scatter(Ir_final[Ir_final > 1E-30].flatten(), Lv_final[Ir_final > 1E-30].flatten(), 
                     c = 'r', s = 1, label = 'Final')
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel(r'$I_r$')
                plt.ylabel(r'$L$')
                plt.legend(loc = 2)
                plt.show()


        if figures:

            Ir_final_masked = np.ma.masked_values(Ir_final, 1E-30)

            plt.figure()
            plt.pcolormesh(Ev_final[:-1, 1:], Lv_final[:-1, 1:], Ir_final_masked[:-1, 1:], 
                        cmap = cmap, norm = LogNorm(vmin = Ir_final_masked.min(), vmax = Ir_final_masked.max()))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\mathcal{E}_f$')
            plt.ylabel(r'$L_f$')
            plt.colorbar(label = r'$I_r(\mathcal{E}_f, L_f)$')
            plt.show()

            plt.figure()
            plt.pcolormesh(psi_final_array[0] - Ev_final[:-1, 1:], Lv_final[:-1, 1:], Ir_final_masked[:-1, 1:], 
                        cmap = cmap, norm = LogNorm(vmin = Ir_final_masked.min(), vmax = Ir_final_masked.max()))
            plt.xlim(left = np.ma.masked_where(Ir_final[:-1, 1:] <= 1E-30, psi_final_array[0] - Ev_final[:-1, 1:]).min())
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\Psi_0 - \mathcal{E}_f$')
            plt.ylabel(r'$L_f$')
            plt.colorbar(label = r'$I_r(\mathcal{E}_f, L_f)$')
            plt.show()


        print("Calculating E_i(E_f, L_f)")
        initial_energies = np.zeros_like(Ev_final)
        for i in range(len(E_final_old)):
            for j in range(len(L_initial)):
                if Ir_final[i, j] <= 1E-30: initial_energies[i, j] = np.nan; continue
                radial_action_initial_flat = Ir_initial[:, j]
                ind = np.argmin(np.abs(radial_action_initial_flat - Ir_final[i, j]))

                initial_energies[i, j] = E_initial_old[ind]

        if np.all(self.output_variables_dataframe['E_i'] != 0):
            for i in range(self._N):
                E_i = self.output_variables_dataframe.loc[i, 'E_i']
                L_i = self.output_variables_dataframe.loc[i, 'L_i']
                L_ind = np.argmin(np.abs(L_initial - L_i))
                E_ind = np.argmin(np.abs(initial_energies[:, L_ind] - E_i))

                self.output_variables_dataframe.loc[i, 'E_f'] = Ev_final[E_ind, L_ind]

        if figures:
            initial_energies_masked = np.ma.masked_invalid(initial_energies)
            plt.figure()
            plt.pcolormesh(Ev_final[:-1, 1:], Lv_final[:-1, 1:], psi_array[0] - initial_energies_masked[:-1, 1:], 
                        cmap = cmap, norm = LogNorm(vmin = psi_array[0] - initial_energies_masked[:-1, 1:].max(), vmax = psi_array[0] - initial_energies_masked[:-1, 1:].min()))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\mathcal{E}_f$')
            plt.ylabel(r'$L_f$')
            plt.colorbar(label = r'$\Psi_{i, max} - \mathcal{E}_i$')
            plt.show()

        phase_space = self._Eddington_func(initial_energies)

        print("Calculating final density")
        rho_final_array = np.array([self.reconstruct_density(r, Ev_final, Lv_final, phase_space, psi_final, r_S = r_S) for r in r_array])
        rho_final = UnivariateSpline(r_array, rho_final_array, k = 3, s = 0)

        if figures:
            rho_check_eddington = np.array([self.reconstruct_density_check(r, self._Eddington_func) for r in r_array])

            fig, ax = plt.subplots(figsize = (4, 5))
            ax.plot(r_array, self._rho_properties_dict['rho_array'], c = 'k', label = r'$\rho_i$')
            ax.plot(r_array, rho_check_eddington, c = 'b', label = r'$\rho_{check}$')
            ax.plot(r_array, rho_final_array,  c = 'r', label = r'$\rho_f$')
            ax.set_ylim(bottom = 1E-11)
            ax.set_xlim(left = r_array[0])
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r'$\rho$')
            ax.legend()
            divider = make_axes_locatable(ax)
            ax_ratio = divider.append_axes("bottom", 1.2, pad=0.1, sharex=ax)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax_ratio.axhline(1, c = 'k')
            ax_ratio.plot(r_array, rho_check_eddington/self._rho_properties_dict['rho_array'], c = 'b')
            ax_ratio.set_ylim(0, 2)
            ax_ratio.set_xlabel(r'$r$ (pc)')
            ax_ratio.set_ylabel(r'$\rho/\rho_o$')
            plt.show()

        if inplace:
            # Update the density instance with either new values if available, and reset if not
            self._rho_func = rho_final
            self._rho_properties_dict['rho_array'] = rho_final_array
            
            self._psi_func = None
            self._psi_properties_dict = None

            self._phi_func = None
            self._phi_properties_dict = None

            self._Eddington_func = None

            self.output_rseries_dataframe.where(self.output_rseries_dataframe == 0, 0, inplace = True)
            self.output_variables_dataframe.where(self.output_variables_dataframe == 0, 0, inplace = True)

        else:
            return Density(self._name+'adiabatic', rho_final, r_array, self._N)

