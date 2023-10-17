#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import warnings

from copy import copy

from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import root_scalar
from scipy.stats import skew
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.chdir('/Users/renske/Documents/MONZ/v2')

from colourmap_maker import lch_colour_map

rgb_palette = np.array([[255, 97, 229], [241, 0, 123], 
                        [151, 0, 71], [54, 3, 27], 
                        [179, 126, 12], [255, 190, 11], 
                        [250, 237, 185], [86, 235, 215],
                        [0, 128, 117], [0, 59, 59]], dtype = np.float32)/255

rgb_palette_dict = {'purple pizzazz': rgb_palette[0], 'flickr pink': rgb_palette[1],
                    'jazzberry jam': rgb_palette[2], 'dark sienna': rgb_palette[3],
                    'dark goldenrod': rgb_palette[4], 'amber': rgb_palette[5],
                    'blond': rgb_palette[6], 'turquiose': rgb_palette[7],
                    'pine green': rgb_palette[8], 'rich black': rgb_palette[9]}

#%%

G_N = 4.302e-3 #Units of (pc/solar mass) (km/s)^2
c_light = 2.9979e05 #km/s

class BoundedPerceptron(torch.nn.Module):
    def __init__(self, nhidden):
        super(BoundedPerceptron, self).__init__()
        self.linear1 = torch.nn.Linear(2, nhidden)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(nhidden, 1)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        u1 = self.linear1(x)
        u2 = self.relu(u1)
        y = self.linear2(u2)
        return self.softplus(y)
    
def train_model(input, target, model, optimizer, criterion, nbatch = 32, nepochs = 10):

  # This is a standard training loop
    loss_hist = []

    for epoch in range(nepochs): 
        # It is important to shuffle the dataset at each epoch so that the minibatches are not always the same
        shuffle = np.random.permutation(input.shape[0])
        input_flat = input.reshape(input.shape[0], *input.shape[1:])
        input_shuffled = input_flat[shuffle].reshape(tuple(input.shape))
        target_shuffled = target[shuffle]
            
        
        for i in range(len(target_shuffled)//nbatch):
            input_batch = input_shuffled[i*nbatch:(i+1)*nbatch]
            target_batch = target_shuffled[i*nbatch:(i+1)*nbatch]

            optimizer.zero_grad()

            # forward pass
            pred_batch = model(input_batch)#.squeeze(1)

            # compute loss
            loss = criterion(pred_batch, target_batch)
            # backward pass
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

        print(epoch, loss.item())
        loss_hist.append(loss.item())

    return loss_hist

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


class BaseDensity:

    def __init__(self, name, rho_func, r_array):

        # Global properties
        self._name = name

        self._r_array = r_array

        self._rho_func = rho_func
        self._rho_array = rho_func(r_array)
        self._M_tot = None
        self._M_system = 0

        self._truncated = False
        self._phi_break = 0
        self._r_break_small = r_array[0]
        self._r_break_large = r_array[-1]

        self._psi_func = None
        self._psi_array = None
        self._psi_min = None
        self._psi_max = None
        self._dpsi_min = None

        self._phi_func = None
        self._phi_array = None
        self._phi_min = None
        self._phi_max = None
        self._shift = None

        self._Eddington_func = None
        self._E_sample_func = None
        self._u_sample_func = None

        self.setup_potentials()


    def __call__(self, r):
        return self._rho_func(r)
    
    def get_psi(self):
        return self._psi_func   

    def get_phi(self):
        return self._phi_func

    def get_f_Eddington(self):
        return self._Eddington_func
    

    def M_enclosed(self, r):
        r_min = np.min(self._r_array)

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
    

    def smoothen_density(self):

        rho_mask = self._rho_array > 0

        smooth_rho_r_func = monotonic_Bspline(self._r_array[rho_mask], self._rho_array[rho_mask]) 

        self._rho_func = smooth_rho_r_func
        self._rho_array = smooth_rho_r_func(self._r_array)


    ## Everything potentials
    
    def setup_potentials(self):

        mask = self._rho_array > 1E-30
        if not mask[0]:
            self._r_break_small = self._r_array[mask][0]
        if not mask[-1]:
            self._r_break_large = self._r_array[mask][-1]

        phi_array = np.array([self.phi_of_r(r) for r in self._r_array])
        if self._M_tot == None:
            self._M_tot = self.M_enclosed(self._r_array[-1])
            self._M_system += self._M_tot
        shift = (phi_array[-1] + G_N*self._M_tot/self._r_array[-1])
        psi_array = shift - phi_array

        self._phi_func = UnivariateSpline(self._r_array, phi_array, k = 3, s = 0)

        self._phi_min = np.min(phi_array)
        self._phi_max = np.max(phi_array)
        self._shift = shift
        self._phi_array = phi_array

        self._psi_func = UnivariateSpline(self._r_array, psi_array, k = 3, s = 0)

        self._psi_min = np.min(psi_array)
        self._psi_max = np.max(psi_array)
        self._dpsi_min = (psi_array[0] - psi_array[1])/10
        self._psi_array = psi_array

        mask = self._rho_array != 0
        logrho_array = np.log10(self._rho_array[mask])
        scaled_logrho_array = (logrho_array - np.min(logrho_array))/(np.max(logrho_array) - np.min(logrho_array))
        self._E_sample_func = interp1d(scaled_logrho_array, psi_array[mask])
        self._u_sample_func = interp1d(psi_array[mask], scaled_logrho_array)

    
    def add_external_potential_from_other(self, other):

        psi_old = copy(self._psi_func)
        self._psi_func = lambda r: psi_old(r) + other._psi_func(r)
        self._psi_array += other._psi_array
        self._psi_min += other._psi_min
        self._psi_max += other._psi_max
        self._dpsi_min += other._dpsi_min

        phi_old = copy(self._phi_func)
        self._phi_func = lambda r: phi_old(r) + other._phi_func(r)
        self._phi_array += other._phi_array
        self._phi_min += other._phi_min
        self._phi_max += other._phi_max
        self._shift += other.shift
        
        self._M_system += other._M_tot

        mask = self._rho_array != 0
        logrho_array = np.log10(self._rho_array[mask])
        scaled_logrho_array = (logrho_array - np.min(logrho_array))/(np.max(logrho_array) - np.min(logrho_array))
        self._E_sample_func = interp1d(scaled_logrho_array, self._psi_array[mask])
        self._u_sample_func = interp1d(self._psi_array[mask], scaled_logrho_array)


    def add_external_potential_from_function(self, psi_func, M_additional):

        psi_old = copy(self._psi_func)
        self._psi_func = lambda r: psi_old(r) + psi_func(r)
        self._psi_array += psi_func(self._r_array)
        self._psi_min += psi_func(self._r_array[-1])
        self._psi_max += psi_func(self._r_array[0])
        self._dpsi_min += (psi_func(self._r_array[0]) - psi_func(self._r_array[1]))/10

        self._M_system += M_additional
        shift = self._shift + G_N*M_additional/self._r_array[-1]

        phi_old = copy(self._phi_func)
        phi_array = shift - self._psi_array
        self._phi_func = lambda r: phi_old(r) + phi_array
        self._phi_min += phi_array[0]
        self._phi_max += phi_array[-1]
        self._shift += shift
        self._phi_array += phi_array

        self._Eddington_func = None

        mask = self._rho_array != 0
        logrho_array = np.log10(self._rho_array[mask])
        scaled_logrho_array = (logrho_array - np.min(logrho_array))/(np.max(logrho_array) - np.min(logrho_array))
        self._E_sample_func = interp1d(scaled_logrho_array, self._psi_array[mask])
        self._u_sample_func = interp1d(self._psi_array[mask], scaled_logrho_array)

    
    def phi_of_r(self, r):

        if r < self._r_break_small:
            return 0

        r_min = np.min(self._r_array)

        r_grid = np.logspace(np.log10(r_min), np.log10(r), num = 1000)

        # Integration doesn't work properly for point mass potentials,
        # so when the density reaches 0, stitch in point mass potential
        if r > self._r_break_large:
            if not self._truncated:
                print("Density is truncated")
                self._truncated = True
                self._phi_break = self.phi_of_r(self._r_break_large)

                self._M_tot = self.M_enclosed(self._r_break_large)
                self._M_system += self._M_tot

            return self._phi_break + G_N*self._M_tot/self._r_break_large - G_N*self._M_tot/r


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
    

    ## Everything phase space

    def setup_phase_space(self, smoothen = False):

        mode = ['psi_logged', 'psi', 'phi']
        eddington_funcs = []

        for i in range(len(mode)):
            print('mode: ', mode[i])
            try:
                eddington_func = self.calculate_f_eddington(mode = mode[i], smoothen = smoothen)
                eddington_funcs.append(eddington_func)
            except Exception as e:
                print('Error message: ', e)
                continue

        if len(eddington_funcs) == 0: raise Exception("The Eddington function could not be computed in any of the modes")
        elif len(eddington_funcs) == 1: return

        mask = self._rho_array != 0
        averages = np.zeros(len(eddington_funcs))
        stds = np.zeros(len(eddington_funcs))
        for i in range(len(eddington_funcs)):
            density_check = np.array([self.reconstruct_density_check(r, eddington_funcs[i]) for r in self._r_array])
            averages[i] = np.average(density_check[mask]/self._rho_array[mask])
            stds[i] = np.std(density_check[mask]/self._rho_array[mask])

        quality = np.abs(1 - averages) + stds
        best = np.argmin(quality)
        print(averages)
        print(stds)
        print(best)
        print(self._Eddington_func == eddington_funcs[best])

        self._Eddington_func = eddington_funcs[best]


    def reconstruct_density_check(self, r, eddington_func):

        E_max = self._psi_func(r)
        reversed_psi = self._psi_array[::-1]

        integrand_grid = eddington_func(reversed_psi)*np.sqrt(2*(E_max - reversed_psi))

        mask_isnan = np.isnan(integrand_grid)
        mask_isinf = np.isinf(integrand_grid)
        mask = np.logical_or(mask_isnan, mask_isinf)
        integrand_grid[mask] = 0

        integral = np.trapz(integrand_grid, reversed_psi)

        return 4*np.pi*integral   
    

    def calculate_f_eddington(self, mode, smoothen = False):
        if mode == 'phi':
            f_eddington = self.calculate_f_Eddington_function_phi(smoothen = smoothen)
        else:
            f_eddington = self.calculate_f_Eddington_function_psi(logged = (mode == 'psi_logged'), smoothen = smoothen)
        return f_eddington


    def calculate_f_Eddington_function_psi(self, logged: bool = False, smoothen = False):

        if logged:
            d2rho_dpsi2 = self.calculate_d2rho_dpsi2_logged(smoothen = smoothen)
        else:
            d2rho_dpsi2 = self.calculate_d2rho_dpsi2(smoothen = smoothen)

        energy_array = self._E_sample_func(np.linspace(0, 1, num = 10000))

        phase_space = np.array([self.f_Eddington_psi(E, d2rho_dpsi2, logged = logged) for E in energy_array])
        if np.any(phase_space < 0): raise ValueError("De Eddington function is fucking negatief")

        f_eddington_interp = interp1d(energy_array, phase_space, fill_value = 0, bounds_error = False)

        plt.figure()
        plt.loglog(energy_array, phase_space, 'k.')
        plt.show()

        return f_eddington_interp
    
    def f_Eddington_psi(self, E: float, d2rho_dpsi2: callable, logged: bool = False) -> float:

        if logged:
            integrand = lambda psi: d2rho_dpsi2(np.log10(psi))/np.sqrt(E - psi)
        else:
            integrand = lambda psi: d2rho_dpsi2(psi)/np.sqrt(E - psi)

        u_max = self._u_sample_func(E)
        psi_grid = self._E_sample_func(np.linspace(0, u_max, num = 10000))

        integrand_grid = integrand(psi_grid)
        # integrand_grid[integrand_grid < 0] = 0
        integrand_grid[np.isnan(integrand_grid)] = 0
        integrand_grid[np.isinf(integrand_grid)] = 0
        result = np.trapz(integrand_grid[integrand_grid > 0], psi_grid[integrand_grid > 0])

        f = (result)/(np.pi**2*8**0.5)

        return f
    

    def calculate_d2rho_dpsi2_logged(self, smoothen = False) -> callable:

        # Calculation of rho(psi) using splines. 

        reversed_rho = self._rho_array[::-1]
        reversed_psi = self._psi_array[::-1]

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

        reversed_rho = self._rho_array[::-1]
        reversed_psi = self._psi_array[::-1]

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
    

    def calculate_f_Eddington_function_phi(self, smoothen = False):

        d2rho_dphi2 = self.calculate_d2rho_dphi2(smoothen = smoothen)

        energy_array = self._E_sample_func(np.linspace(0, 1, num = 10000))

        phase_space = np.array([self.f_Eddington_phi(E, d2rho_dphi2) for E in energy_array])
        if np.any(phase_space < 0): raise ValueError("De Eddington function is fucking negatief")

        f_eddington_interp = interp1d(energy_array, phase_space, fill_value = 0, bounds_error = False)

        plt.figure()
        plt.loglog(energy_array, phase_space, 'k.')
        plt.show()

        return f_eddington_interp


    def f_Eddington_phi(self, E: float, d2rho_dphi2: callable) -> float:

        shift = self._shift

        integrand = lambda psi: d2rho_dphi2(shift - psi)/np.sqrt(E - psi)

        u_max = self._u_sample_func(E)
        psi_grid = self._E_sample_func(np.linspace(0, u_max, num = 10000))

        integrand_grid = integrand(psi_grid)
        integrand_grid[np.isnan(integrand_grid)] = 0
        integrand_grid[np.isinf(integrand_grid)] = 0
        result = np.trapz(integrand_grid[integrand_grid > 0], psi_grid[integrand_grid > 0])

        f = (result)/(np.pi**2*8**0.5)

        return f
    
    def calculate_d2rho_dphi2(self, smoothen = False) -> callable:

        if smoothen:
            mask = self._rho_array > 0
            smooth_rho_phi = smooth_Bspline(self._phi_array[mask], self._rho_array[mask], increasing = True)
            smooth_rho_array = smooth_rho_phi(self._phi_array)

            rho_phi = UnivariateSpline(self._phi_array, smooth_rho_array, k = 3, s = 0)
        else:
            rho_phi = UnivariateSpline(self._phi_array, self._rho_array, k = 3, s = 0)

        
        d2rho_dphi2 = rho_phi.derivative(n = 2)

        return d2rho_dphi2

    

class EvolvingDensity(BaseDensity):

    def __init__(self, name, rho_func, r_array, N_particles):

        super().__init__(name, rho_func, r_array)        

        self._N = N_particles
        self._sampled = False

        tracked_variables = ["r_i", "E_i", 'L_i', "T_r_i", "r_apo_i", "r_peri_i", "T_r_f", "r_apo_f", "r_peri_f", "global_weight", "errored", "eaten", "GS eaten"]
        self.output_variables_dataframe = pd.DataFrame(np.zeros((N_particles, len(tracked_variables))), columns = tracked_variables)
        self.output_rseries_dataframe = pd.DataFrame(np.zeros((N_particles, len(r_array))))


    def sample_single_orbit(self, i: int, ppf_r_i: callable, logr: bool = False) -> int:
        
        psi_initial = self._psi_func
        rho_initial = self._rho_func
        f_eddington = self._Eddington_func


        if logr: r_i_sample = 10**ppf_r_i(np.random.random())
        else: r_i_sample = ppf_r_i(np.random.random())
        # print("Sampled r_i = {:e}".format(r_i_sample))
        if rho_initial(r_i_sample) == 0: 
            self.sample_single_orbit(i, ppf_L_i, ppf_r_i, logr = logr)
            return

        # calculate ppf_E_i and draw E_i
        E_max = psi_initial(r_i_sample)

        E_initial = np.logspace(np.log10(self._psi_min), np.log10(E_max), num = 2000)[:-1]
        E_zoom = E_max - np.logspace(np.log10(self._dpsi_min), min(np.log10(self._psi_max)//1 - 2, np.log10(self._psi_max - self._psi_min)), num = 4000)
        E_initial = np.append(E_initial, E_zoom)
        E_initial = np.append(E_initial, np.linspace(self._psi_min, E_max, num = 4000))
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

        ppf_L_i = lambda x: r_i_sample*np.sqrt( 2*(psi_initial(r_i_sample) - E_i_sample) - 2*(psi_initial(r_i_sample) - E_i_sample)*(1 - x)**2 )

        L_i_sample = ppf_L_i(np.random.random())

        self.output_variables_dataframe.loc[i, 'r_i'] = r_i_sample
        self.output_variables_dataframe.loc[i, 'E_i'] = E_i_sample
        self.output_variables_dataframe.loc[i, 'L_i'] = L_i_sample
    

    def radial_orbital_time(self, i, orbit_vr_squared_analytical, r_S = 0):    
        
        v_r_squared_array = orbit_vr_squared_analytical(self._r_array)

        r_real = self._r_array[v_r_squared_array >= 0]

        if len(r_real) == 0: # No roots within r_array
            self.output_variables_dataframe.loc[i, 'errored'] = 1
            return
        elif r_real[0] == self._r_array[0]:  # Only one root at large r (small r root cannot be resolved)
            self.output_variables_dataframe.loc[i, 'errored'] = 2
            return
        elif r_real[-1] == self._r_array[-1]: # Only one root at small r (large r root outside of r_array)
            self.output_variables_dataframe.loc[i, 'errored'] = 3
            return
        else: 
            mid = r_real[len(r_real)//2]
            r_periapsis = root_scalar(orbit_vr_squared_analytical, bracket = [self._r_array[0], mid]).root
            r_apoapsis = root_scalar(orbit_vr_squared_analytical, bracket = [mid, self._r_array[-1]]).root

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

        if np.isnan(T_r_large_r):
            self.output_variables_dataframe.loc[i, 'errored'] = 4
            return
        
        if np.isnan(T_r_small_r):
            self.output_variables_dataframe.loc[i, 'errored'] = 5

        T_r = np.where(np.isnan(T_r_small_r), 0, T_r_small_r) + T_r_large_r # s * pc/km 

        return T_r, r_periapsis, r_apoapsis
    
    
    def radial_action(self, E, L, psi_func):
        
        v_r_squared = lambda r: 2*psi_func(r) - 2*E - L**2/r**2
        v_r_squared_array = v_r_squared(self._r_array)

        r_real = self._r_array[v_r_squared_array >= 0]

        if len(r_real) == 0: # No roots within r_array
            return 1E-30
        elif len(r_real) == 1:
            r_refined = np.linspace(r_real[0]*0.9, r_real[0]*1.1, num = 100)
            v_r_squared_refined = v_r_squared(r_refined)

            r_real_refined = r_refined[v_r_squared_refined >= 0]
            if len(r_real_refined) < 2: return 1E-30
            I_r = np.trapz(np.sqrt(v_r_squared(r_real_refined)), r_real_refined)
            return I_r

        I_r = np.trapz(np.sqrt(v_r_squared(r_real)), r_real)

        return I_r


    def integrate_probability(self, i, orbit_vr_squared_analytical, r_S = 0):

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
            p_r = (1/T_r * 1/np.sqrt(orbit_vr_squared_analytical(self._r_array)))

        nan_mask = np.isnan(p_r)
        inf_mask = np.isinf(p_r)
        bad_mask = np.logical_or(nan_mask, inf_mask)
        if len(p_r[~bad_mask]) == 0: return

        p_r[bad_mask] = 0
        # cdf = cumulative_trapezoid(p_r, r_array, initial = 0)

        self.output_rseries_dataframe.loc[i] = p_r # /cdf[-1]


    def calculate_orbital_time_distribution(self, logr: bool = True, figures = False):

        # calculate ppf_r_i (because it is unchanging)
        print("Calculating the ppf for p(r_i)")
        if self._M_tot == None:
            self._M_tot = self.M_enclosed(self._r_array[-1])
            self._M_system += self._M_tot
        pdf_r_i = lambda r: 4*np.pi*r**2*self._rho_func(r)/self._M_tot

        if logr:
            ppf_r_i = lambda x: np.log10(self._r_array[-1]/self._r_array[0])*x + np.log10(self._r_array[0])
        else:
            r_initial = np.logspace(max(-10, np.log10(self._r_array[0])), np.log10(self._r_array[-1]), num = 10000)
            pdf_r_array = pdf_r_i(r_initial)
            cdf_r_i = cumulative_trapezoid(pdf_r_array, r_initial, initial = 0)
            ppf_r_i = interp1d(cdf_r_i/cdf_r_i[-1], r_initial)

        # calculate f_Eddington function
        if not self._sampled and self._Eddington_func == None:
            print("Calculating Eddington function")
            self.setup_phase_space()

        if figures and self._Eddington_func != None:

            E_initial = np.logspace(np.log10(self._psi_min), np.log10(self._psi_max), num = 1000)
            E_initial = np.append(E_initial, self._psi_max - np.logspace(np.log10(self._dpsi_min), max(np.log10(self._psi_max)//1 - 2, 0), num = 2000))
            E_initial = np.append(E_initial, np.linspace(self._psi_min, self._psi_max, num = 1000))
            E_initial = np.unique(E_initial)

            plt.figure()
            plt.loglog(E_initial, self._Eddington_func(E_initial), color = rgb_palette_dict['flickr pink'])
            plt.xlabel(r'$\mathcal{E}$')
            plt.ylabel(r'$f(\mathcal{E})$')
            plt.savefig('./figures/'+self._name+'f_Eddington.pdf')
            plt.show()


        for i in range(self._N):
            if i%100 == 0: print("Orbit: ", i)

            if not self._sampled:
                self.sample_single_orbit(i, ppf_r_i, logr = logr)
            
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

        self._sampled = True

        if figures:
            r_apo_i_samples = self.output_variables_dataframe['r_apo_i']
            r_peri_i_samples = self.output_variables_dataframe['r_peri_i']

            hist_apo_peri_i, x_edges, y_edges = np.histogram2d(r_apo_i_samples, r_peri_i_samples, bins = self._r_array[::10], 
                                        density = False)
            hist_apo_peri_i_masked = np.ma.masked_equal(hist_apo_peri_i.T, 0)

            lch_cmap = lch_colour_map([rgb_palette_dict['dark sienna'], rgb_palette_dict['flickr pink'], rgb_palette_dict['blond']], N_bins = 256)
            plt.figure()
            plt.pcolormesh(x_edges, y_edges, hist_apo_peri_i_masked, 
                                    cmap = lch_cmap, norm = LogNorm(vmin = hist_apo_peri_i_masked.min(), vmax = hist_apo_peri_i_masked.max()))
            plt.plot(self._r_array, self._r_array, c = rgb_palette_dict['turquiose'])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$r_{apo}$')
            plt.ylabel(r'$r_{peri}$')
            plt.colorbar()
            plt.savefig('./figures/'+self._name+'hist_r_apo_peri_i.pdf')
            plt.show()


    def instantaneous_growth(self, delta_psi: callable, delta_M, r_S = 0, logr: bool = True, figures = False) -> int:

        """
        Calculate N orbits for orbits from the initial potential, adjusting to the final potential,
        at time steps in t_array.
        """

        psi_final_array = self._psi_array + delta_psi(self._r_array)
        psi_final = UnivariateSpline(self._r_array, psi_final_array, k = 3, s = 0)

        if figures:
            plt.figure()
            plt.loglog(self._r_array, self._psi_array, color = rgb_palette_dict['dark sienna'], label = r'$\psi_i$')
            plt.loglog(self._r_array, psi_final_array, color = rgb_palette_dict['flickr pink'], label = r'$\psi_f$')
            plt.xlabel(r'$r$')
            plt.ylabel(r'$\psi(r)$')
            plt.legend()
            plt.savefig('./figures/'+self._name+'_psi_of_r.pdf')
            plt.show()

       
        # calculate f_Eddington function
        if not self._sampled and self._Eddington_func == None:
            print("Calculating Eddington function")
            self.setup_phase_space()

        if figures and self._Eddington_func != None:

            psi_max = self._psi_max
            psi_min = self._psi_min
            dpsi_min = self._dpsi_min

            E_initial = np.logspace(np.log10(psi_min), np.log10(psi_max), num = 1000)
            E_initial = np.append(E_initial, psi_max - np.logspace(np.log10(dpsi_min), max(np.log10(psi_max)//1 - 2, 0), num = 2000))
            E_initial = np.append(E_initial, np.linspace(psi_min, psi_max, num = 1000))
            E_initial = np.unique(E_initial)

            plt.figure()
            plt.loglog(E_initial, self._Eddington_func(E_initial), color = rgb_palette_dict['flickr pink'])
            plt.xlabel(r'$\mathcal{E}$')
            plt.ylabel(r'$f(\mathcal{E})$')
            plt.savefig('./figures/'+self._name+'f_Eddington.pdf')
            plt.show()

        # calculate ppf_r_i (because it is unchanging)
        print("Calculating the ppf for p(r_i)")
        if self._M_tot == None:
            self._M_tot = self.M_enclosed(self._r_array[-1])
            self._M_system += self._M_tot
        pdf_r_i = lambda r: 4*np.pi*r**2*self._rho_func(r)/self._M_tot

        if logr:
            ppf_r_i = lambda x: np.log10(self._r_array[-1]/self._r_array[0])*x + np.log10(self._r_array[0])
        else:
            r_initial = np.logspace(max(-10, np.log10(self._r_array[0])), np.log10(self._r_array[-1]), num = 10000)
            pdf_r_array = pdf_r_i(r_initial)
            cdf_r_i = cumulative_trapezoid(pdf_r_array, r_initial, initial = 0)
            ppf_r_i = interp1d(cdf_r_i/cdf_r_i[-1], r_initial)

        print("Beginning orbit calculations")
        E_f_samples = np.zeros(self._N)
        for i in range(self._N):
            if i%100 == 0: print("Orbit: ", i)

            if not self._sampled:
                self.sample_single_orbit(i, ppf_r_i, logr = logr)
            
            r_i = self.output_variables_dataframe.loc[i, 'r_i']
            E_i = self.output_variables_dataframe.loc[i, 'E_i']
            L_i = self.output_variables_dataframe.loc[i, 'L_i']

            if logr:
                self.output_variables_dataframe.loc[i, 'global_weight'] = pdf_r_i(r_i)*r_i
            else:
                self.output_variables_dataframe.loc[i, 'global_weight'] = 1

            E_f = E_i + delta_psi(r_i)

            orbit_vr_squared_analytical = lambda r: 2*(psi_final(r) - E_f) - (L_i**2/r**2)
            
            E_f_samples[i] = E_f

            if L_i <= 2*r_S*c_light: self.output_variables_dataframe.loc[i, 'GS eaten'] = 1
            if np.all(psi_final(self._r_array)*(1-4*r_S/self._r_array) < E_i + delta_psi(r_i)):
                self.output_variables_dataframe.loc[i, 'GS eaten'] += 1
                self.output_variables_dataframe.loc[i, 'GS eaten'] *= 2

            self.integrate_probability(i, orbit_vr_squared_analytical, r_S = r_S)

        self._sampled = True

        if figures:
            r_apo_f_samples = self.output_variables_dataframe['r_apo_f']
            r_peri_f_samples = self.output_variables_dataframe['r_peri_f']

            hist_apo_peri_f, x_edges, y_edges = np.histogram2d(r_apo_f_samples, r_peri_f_samples, bins = self._r_array[::10], 
                                        density = False)
            hist_apo_peri_f_masked = np.ma.masked_equal(hist_apo_peri_f.T, 0)

            lch_cmap = lch_colour_map([rgb_palette_dict['dark sienna'], rgb_palette_dict['flickr pink'], rgb_palette_dict['blond']], N_bins = 256)
            plt.figure()
            plt.pcolormesh(x_edges, y_edges, hist_apo_peri_f_masked, 
                                    cmap = lch_cmap, norm = LogNorm(vmin = hist_apo_peri_f_masked.min(), vmax = hist_apo_peri_f_masked.max()))
            plt.plot(self._r_array, self._r_array, c = rgb_palette_dict['turquiose'])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$r_{apo}$')
            plt.ylabel(r'$r_{peri}$')
            plt.colorbar()
            plt.savefig('./figures/'+self._name+'hist_r_apo_peri_f.pdf')
            plt.show()

        mask_eaten = self.output_variables_dataframe['eaten'] == 0
        mask_errored = self.output_variables_dataframe['errored'] == 0
        mask = np.logical_and(mask_eaten, mask_errored)
        p_r_marginal = np.average(self.output_rseries_dataframe.loc[mask], axis = 0, weights = self.output_variables_dataframe.loc[mask, 'global_weight'])
        norm = np.trapz(p_r_marginal, self._r_array)
        rho_final_array = self._M_tot*(p_r_marginal/norm)/(4*np.pi*self._r_array**2)
        rho_final = UnivariateSpline(self._r_array, rho_final_array, k = 3, s = 0)

        if figures:
            bins = self._r_array[::10]
            p_r_t_0, edges = np.histogram(self.output_variables_dataframe['r_i'], bins = bins, density = True, weights = self.output_variables_dataframe['global_weight'])
            rho_sampled = self._M_tot*p_r_t_0/(4*np.pi*self._r_array[5::10]**2)

            fig, ax = plt.subplots(figsize = (4, 5))
            ax.plot(self._r_array, self._rho_array, c = rgb_palette_dict['rich black'], label = r'$\rho_i$')
            ax.plot(self._r_array[5::10], rho_sampled, c = rgb_palette_dict['amber'], label = r'$\rho_{sampled}$')
            ax.plot(self._r_array, rho_final_array,  c = rgb_palette_dict['flickr pink'], label = r'$\rho_f$')
            ax.set_ylim(bottom = 1E-11)
            ax.set_xlim(left = self._r_array[0])
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r'$\rho$')
            ax.legend()
            divider = make_axes_locatable(ax)
            ax_ratio = divider.append_axes("bottom", 1.2, pad=0.1, sharex=ax)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax_ratio.axhline(1, c = rgb_palette_dict['rich black'])
            ax_ratio.plot(self._r_array[5::10], rho_sampled/self._rho_array[5::10], c = rgb_palette_dict['amber'])
            ax_ratio.set_ylim(-0.1, 2.1)
            ax_ratio.set_xlabel(r'$r$ (pc)')
            ax_ratio.set_ylabel(r'$\rho/\rho_o$')
            plt.savefig('./figures/'+self._name+'_reconstructed_densities.pdf')
            plt.show()

        r_f_samples = np.zeros(self._N)
        new_weights_samples = np.zeros(self._N)

        for i in range(self._N):
            if not mask[i]:
                continue

            E_f = E_f_samples[i]
            L_i = self.output_variables_dataframe.loc[i, 'L_i']
            T_r_f = self.output_variables_dataframe.loc[i, 'T_r_f']
            r_peri_f = self.output_variables_dataframe.loc[i, 'r_peri_f']
            r_apo_f = self.output_variables_dataframe.loc[i, 'r_apo_f']

            # if T_r_f == 0: continue
            # if r_peri_f < 4*r_S: continue
            if np.isnan(T_r_f): continue

            p_r_given_E_L = lambda r: 1/np.sqrt(2*(psi_final(r) - E_f) - L_i**2/r**2) * 1/T_r_f

            ppf_r_f = lambda x: np.log10(r_apo_f/r_peri_f)*x + np.log10(r_peri_f)
            r_f = 10**ppf_r_f(np.random.random())
            r_f_samples[i] = r_f
            new_weights_samples[i] = p_r_given_E_L(r_f)*r_f*np.log10(r_apo_f/r_peri_f)

        zero_mask = r_f_samples != 0
        N_noneaten = len(r_f_samples[zero_mask])
        new_density = EvolvingDensity(self._name+'instantaneous', rho_final, self._r_array, N_noneaten)

        new_density.smoothen_density()
        new_density.add_external_potential_from_function(delta_psi, delta_M)

        new_density.output_variables_dataframe['E_i'] = E_f_samples[zero_mask]
        new_density.output_variables_dataframe['L_i'] = self.output_variables_dataframe.loc[zero_mask, 'L_i'].values
        new_density.output_variables_dataframe['T_r_i'] = self.output_variables_dataframe.loc[zero_mask, 'T_r_f'].values
        new_density.output_variables_dataframe['r_apo_i'] = self.output_variables_dataframe.loc[zero_mask, 'r_apo_f'].values
        new_density.output_variables_dataframe['r_peri_i'] = self.output_variables_dataframe.loc[zero_mask, 'r_peri_f'].values

        new_density.output_variables_dataframe['r_i'] = r_f_samples[zero_mask]
        new_density.output_variables_dataframe['global_weight'] = new_weights_samples[zero_mask] * self.output_variables_dataframe.loc[zero_mask, 'global_weight'].values
        new_density._sampled = True

        return new_density


    def adiabatic_growth(self, delta_psi: callable, delta_M, r_S = 0, logr: bool = True, figures = False):

        psi_final_array = self._psi_array + delta_psi(self._r_array)
        psi_final = UnivariateSpline(self._r_array, psi_final_array, k = 3, s = 0)

        if figures:
            plt.figure()
            plt.loglog(self._r_array, self._psi_array, color = rgb_palette_dict['dark sienna'], label = r'$\psi_i$')
            plt.loglog(self._r_array, psi_final_array, color = rgb_palette_dict['flickr pink'], label = r'$\psi_f$')
            plt.xlabel(r'$r$ (pc)')
            plt.ylabel(r'$\psi(r)$ ')
            plt.legend()
            plt.savefig('./figures/'+self._name+'_psi_of_r.pdf')
            plt.show()

        if not self._sampled and self._Eddington_func == None:
            print("Calculating initial densities and potentials")
            self.setup_phase_space()

        if figures and self._Eddington_func != None:

            E_grid = np.logspace(np.log10(self._psi_min), np.log10(self._psi_max), num = 1000)
            E_grid = np.append(E_grid, self._psi_max - np.logspace(np.log10(self._dpsi_min), min(np.log10(self._psi_max)//1 - 2, np.log10(self._psi_max - self._psi_min)), num = 2000))
            E_grid = np.append(E_grid, np.linspace(self._psi_min, self._psi_max, num = 1000))
            E_grid = np.unique(E_grid)
            plt.figure()
            plt.loglog(E_grid, self._Eddington_func(E_grid), color = rgb_palette_dict['flickr pink'])
            plt.xlabel(r'${\mathcal{E}}$')
            plt.ylabel(r'$f(\mathcal{E})$')
            plt.savefig('./figures/'+self._name+'_f_Eddington.pdf')
            plt.show()


        L_matrix = np.zeros((len(self._r_array), len(self._r_array)))

        for i in range(len(self._r_array)):
            r = self._r_array[i]
            E_max = self._psi_func(r)
            L_matrix[i] = r*np.sqrt(2*(E_max - self._psi_array))

        L_matrix[np.isnan(L_matrix)] = 0

        L_max = np.zeros_like(self._r_array)
        for i in range(len(self._r_array)):
            L_max[i] = np.max(L_matrix[:, i])

        E_L_max = UnivariateSpline(L_max, self._psi_array, k = 1, s = 0)

        L_initial = np.logspace(-10, np.log10(np.max(L_max)), num = 500)
        L_initial = np.append(L_initial, np.linspace(0, np.max(L_max), num = 501)[1:])
        L_initial = np.unique(L_initial)

        E_initial = self._psi_array[0] - np.logspace(np.log10(self._psi_max - E_L_max(L_initial)), np.log10(self._psi_max - self._psi_min), num = 2000)
        E_initial = np.vstack((E_initial, np.logspace(np.log10(self._psi_min), np.log10(self._psi_max), num = 1000)))
        E_initial = np.sort(E_initial, axis = 0)
        E_initial[-1] = E_L_max(L_initial)


        Ev_initial, Lv_initial = np.meshgrid(E_initial[:, 0], L_initial, indexing = 'ij')

        print("Calculating initial radial action")
        Ir_initial = np.array([[self.radial_action(E_initial[i, j], L_initial[j], self._psi_func) for j in range(len(L_initial))] for i in range(E_initial.shape[0])])

        mask = Ir_initial[1:] > 1E-30
        Ir_initial_diff = np.log10(Ir_initial[:-1][mask]/Ir_initial[1:][mask])

        if figures:
            plt.figure()
            plt.scatter(Ir_initial[Ir_initial > 1E-30].flatten(), Lv_initial[Ir_initial > 1E-30].flatten(), 
                     c = rgb_palette_dict['dark sienna'], s = 1, label = 'Initial')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$I_r$')
            plt.ylabel(r'$L$')
            plt.legend(loc = 2)
            plt.savefig('./figures/'+self._name+'_Ir_vs_L_initial.png', dpi = 300)
            plt.show()

            plt.figure()
            plt.plot(E_initial[1:][mask], Ir_initial_diff, 'k.')
            plt.xlabel('E')
            plt.xscale('log')
            plt.show()


        if figures:

            Ir_initial_masked = np.ma.masked_values(Ir_initial, 1E-30)
            lch_cmap = lch_colour_map([rgb_palette_dict['dark sienna'], rgb_palette_dict['flickr pink'], rgb_palette_dict['blond']], N_bins = 256)
            plt.figure()
            plt.pcolormesh(self._psi_max - E_initial, Lv_initial, Ir_initial_masked, 
                        cmap = lch_cmap, norm = LogNorm(vmin = Ir_initial_masked.min(), vmax = Ir_initial_masked.max()))
            plt.xlim(left = np.ma.masked_where(Ir_initial <= 1E-30, self._psi_array[0] - Ev_initial).min())
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\Psi_{i, max} - \mathcal{E}_i$')
            plt.ylabel(r'$L_i$')
            plt.colorbar(label = r'$I_r(\mathcal{E}_i, L_i)$')
            # plt.savefig('./figures/'+self._name+'_Ir_of_E_and_L_initial.png', dpi = 300)
            plt.show()

        
        L_matrix = np.zeros((len(self._r_array), len(self._r_array)))

        for i in range(len(self._r_array)):
            r = self._r_array[i]
            E_max = psi_final(r)
            L_matrix[i] = r*np.sqrt(2*(E_max - psi_final_array))

        L_matrix[np.isnan(L_matrix)] = 0

        L_max = np.zeros_like(self._r_array)
        for i in range(len(self._r_array)):
            L_max[i] = np.max(L_matrix[:, i])

        E_L_max_final = UnivariateSpline(L_max, psi_final_array, k = 1, s = 0)


        E_final = np.logspace(np.log10(np.min(psi_final_array)), np.log10(E_L_max_final(L_initial)), num = 2000)
        E_final = np.vstack((E_final, psi_final_array[0] - np.logspace(np.log10(psi_final_array[0] - 0.999*E_L_max_final(L_initial)), np.log10(psi_final_array[0] - psi_final_array[-1]), num = 2000)))
        E_final = np.sort(E_final, axis = 0)
        E_final[-1] = E_L_max_final(L_initial)

        Ev_final, Lv_final = np.meshgrid(E_final[:, 0], L_initial, indexing = 'ij')

        print("Calculating final radial action")
        Ir_final = np.array([[self.radial_action(E_final[i, j], L_initial[j], psi_final) for j in range(len(L_initial))] for i in range(E_final.shape[0])])

        mask = Ir_final[1:] > 1E-30
        Ir_final_diff = np.log10(Ir_final[:-1][mask]/Ir_final[1:][mask])

        if figures:
            plt.figure()
            plt.scatter(Ir_initial[Ir_initial > 1E-30].flatten(), Lv_initial[Ir_initial > 1E-30].flatten(), 
                     c = rgb_palette_dict['dark sienna'], s = 1, label = 'Initial')
            plt.scatter(Ir_final[Ir_final > 1E-30].flatten(), Lv_final[Ir_final > 1E-30].flatten(), 
                     c = rgb_palette_dict['flickr pink'], s = 1, label = 'Final')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$I_r$')
            plt.ylabel(r'$L$')
            plt.legend(loc = 2)
            plt.savefig('./figures/'+self._name+'_Ir_vs_L_final.png', dpi = 300)
            plt.show()

            plt.figure()
            plt.plot(Ev_final[1:][mask], Ir_final_diff, 'k.')
            plt.xlabel('E')
            plt.xscale('log')
            plt.show()


        if figures:

            Ir_final_masked = np.ma.masked_values(Ir_final, 1E-30)
            lch_cmap = lch_colour_map([rgb_palette_dict['dark sienna'], rgb_palette_dict['flickr pink'], rgb_palette_dict['blond']], N_bins = 256)
            plt.figure()
            plt.pcolormesh(E_final, Lv_final, Ir_final_masked, 
                        cmap = lch_cmap, norm = LogNorm(vmin = Ir_final_masked.min(), vmax = Ir_final_masked.max()))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\mathcal{E}_f$')
            plt.ylabel(r'$L_f$')
            plt.colorbar(label = r'$I_r(\mathcal{E}_f, L_f)$')
            plt.savefig('./figures/'+self._name+'_Ir_of_E_and_L_final.png', dpi = 300)
            plt.show()

            plt.figure()
            plt.pcolormesh(psi_final_array[0] - E_final, Lv_final, Ir_final_masked, 
                        cmap = lch_cmap, norm = LogNorm(vmin = Ir_final_masked.min(), vmax = Ir_final_masked.max()))
            plt.xlim(left = np.ma.masked_where(Ir_final <= 1E-30, psi_final_array[0] - Ev_final).min())
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\Psi_0 - \mathcal{E}_f$')
            plt.ylabel(r'$L_f$')
            plt.colorbar(label = r'$I_r(\mathcal{E}_f, L_f)$')
            plt.savefig('./figures/'+self._name+'_Ir_of_E_and_L_final_2.png', dpi = 300)
            plt.show()


        print("Calculating E_i(E_f, L_f)")
        final_energies = np.zeros_like(Ev_initial)
        for j in range(len(L_initial)):
            radial_action_final_flat = Ir_final[:, j]
            mask = np.isnan(radial_action_final_flat)
            for i in range(len(E_initial[:, 0])):
                if Ir_initial[i, j] <= 1E-30: final_energies[i, j] = np.nan; continue
                ind = np.argmin(np.abs(radial_action_final_flat[~mask] - Ir_initial[i, j]))

                final_energies[i, j] = E_final[ind, j]
            print(j)

        if figures:
            final_energies_masked = np.ma.masked_invalid(final_energies)
            lch_cmap = lch_colour_map([rgb_palette_dict['dark sienna'], rgb_palette_dict['flickr pink'], rgb_palette_dict['blond']], N_bins = 256)
            plt.figure()
            plt.pcolormesh(E_initial, Lv_initial, final_energies_masked, 
                        cmap = lch_cmap, norm = LogNorm(vmin = final_energies_masked.min(), vmax = final_energies_masked.max()))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\mathcal{E}_f$')
            plt.ylabel(r'$L_f$')
            plt.colorbar(label = r'$\Psi_{i, max} - \mathcal{E}_i$')
            plt.savefig('./figures/'+self._name+'_Ei_of_Ef_and_L.png', dpi = 300)
            plt.show()


        print("Fitting E_i to E_f map")

        # We want a distribution of energy points with high std and low skew
        E_max_factor = np.std(np.log10(E_initial[-1])) - abs(skew(np.log10(E_initial[-1])))
        psi_E_max_factor = np.std(np.log10(self._psi_max - E_initial[-1])) - abs(skew(np.log10(self._psi_max - E_initial[-1])))                 

        Li_flat = np.log10(Lv_initial.flatten())
        if psi_E_max_factor > E_max_factor:
            Ei_flat = np.log10(self._psi_max - E_initial.flatten())
        else:
            Ei_flat = np.log10(E_initial.flatten())
        Ef_flat = np.log10(final_energies.flatten())
        mask = np.isnan(Ef_flat)

        Li_scaled = (Li_flat[~mask] - Li_flat[~mask].mean())/Li_flat[~mask].std()
        Ei_scaled = (Ei_flat[~mask] - Ei_flat[~mask].mean())/Ei_flat[~mask].std()
        # Ef_scaled = (Ef_flat[~mask] - Ef_flat[~mask].mean())/Ef_flat[~mask].std()
        Ef_scaled = np.log10(E_L_max_final(Lv_initial.flatten()[~mask])) - Ef_flat[~mask]

        data = np.vstack((Ei_scaled, Li_scaled)).T

        X_tensor = torch.from_numpy(data).to(torch.float)
        y_tensor = torch.from_numpy(Ef_scaled.reshape(len(Ef_scaled), 1)).to(torch.float)

        if figures:
            lch_cmap = lch_colour_map([rgb_palette_dict['dark sienna'], rgb_palette_dict['flickr pink'], rgb_palette_dict['blond']], N_bins = 256)
            plt.figure()
            plt.scatter(data[:, 0], data[:, 1], c = y_tensor, s = 1,
                        cmap = lch_cmap, norm = Normalize(vmin = y_tensor.min(), vmax = y_tensor.max()))
            plt.xlabel(r'$\Psi_{i, max} - \mathcal{E}_i$')
            plt.ylabel(r'$L_f$')
            plt.colorbar(label = r'$\mathcal{E}_f$')
            plt.show()

        model = BoundedPerceptron(500)
        optimizer = torch.optim.Adam(model.parameters(), lr = 7.5E-6)
        criterion = torch.nn.MSELoss()

        model.train()
        loss_hist = train_model(X_tensor, y_tensor , model, optimizer, criterion, nbatch = 256, nepochs = 300)
        model.eval()

        if figures:
            y_pred = model(X_tensor).detach().numpy()

            fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (6, 3))
            ax1.semilogy(loss_hist)
            ax2.scatter(data[:, 0], data[:, 1], c = y_pred, s = 1,
                    cmap = lch_cmap, norm = Normalize(vmin = y_tensor.min(), vmax = y_tensor.max()))
            plt.show()


        if self._M_tot == None:
            self._M_tot = self.M_enclosed(self._r_array[-1])
            self._M_system += self._M_tot
        pdf_r_i = lambda r: 4*np.pi*r**2*self._rho_func(r)/self._M_tot

        if logr:
            ppf_r_i = lambda x: np.log10(self._r_array[-1]/self._r_array[0])*x + np.log10(self._r_array[0])
        else:
            r_initial = np.logspace(max(-10, np.log10(self._r_array[0])), np.log10(self._r_array[-1]), num = 10000)
            pdf_r_array = pdf_r_i(r_initial)
            cdf_r_i = cumulative_trapezoid(pdf_r_array, r_initial, initial = 0)
            ppf_r_i = interp1d(cdf_r_i/cdf_r_i[-1], r_initial)


        print("Beginning orbit calculations")
        E_f_samples = np.zeros(self._N)
        for i in range(self._N):
            if i%100 == 0: print("Orbit: ", i)

            if not self._sampled:
                self.sample_single_orbit(i, ppf_r_i, logr = logr)
            
            r_i = self.output_variables_dataframe.loc[i, 'r_i']
            E_i = self.output_variables_dataframe.loc[i, 'E_i']
            L_i = self.output_variables_dataframe.loc[i, 'L_i']

            if psi_E_max_factor > E_max_factor:
                E_i_model = (np.log10(self._psi_max - E_i) - Ei_flat[~np.isnan(Ef_flat)].mean())/Ei_flat[~np.isnan(Ef_flat)].std()
            else: 
                E_i_model = (np.log10(E_i) - Ei_flat[~np.isnan(Ef_flat)].mean())/Ei_flat[~np.isnan(Ef_flat)].std()
            L_i_model = (np.log10(L_i) - Li_flat[~np.isnan(Ef_flat)].mean())/Li_flat[~np.isnan(Ef_flat)].std()

            if logr:
                self.output_variables_dataframe.loc[i, 'global_weight'] = pdf_r_i(r_i)*r_i
            else:
                self.output_variables_dataframe.loc[i, 'global_weight'] = 1

            E_f_model = float(model(torch.tensor([[E_i_model, L_i_model]]).to(torch.float)).detach())
            E_f = E_L_max_final(L_i) / 10**(E_f_model)

            orbit_vr_squared_analytical = lambda r: 2*(psi_final(r) - E_f) - (L_i**2/r**2)
            
            E_f_samples[i] = E_f

            if L_i <= 2*r_S*c_light: self.output_variables_dataframe.loc[i, 'GS eaten'] = 1
            if np.all(psi_final(self._r_array)*(1-4*r_S/self._r_array) < E_i + delta_psi(r_i)):
                self.output_variables_dataframe.loc[i, 'GS eaten'] += 1
                self.output_variables_dataframe.loc[i, 'GS eaten'] *= 2

            self.integrate_probability(i, orbit_vr_squared_analytical, r_S = r_S)

        self._sampled = True

        if figures:
            r_apo_f_samples = self.output_variables_dataframe['r_apo_f']
            r_peri_f_samples = self.output_variables_dataframe['r_peri_f']

            hist_apo_peri_f, x_edges, y_edges = np.histogram2d(r_apo_f_samples, r_peri_f_samples, bins = self._r_array[::10], 
                                        density = False)
            hist_apo_peri_f_masked = np.ma.masked_equal(hist_apo_peri_f.T, 0)

            lch_cmap = lch_colour_map([rgb_palette_dict['dark sienna'], rgb_palette_dict['flickr pink'], rgb_palette_dict['blond']], N_bins = 256)
            plt.figure()
            plt.pcolormesh(x_edges, y_edges, hist_apo_peri_f_masked, 
                                    cmap = lch_cmap, norm = LogNorm(vmin = hist_apo_peri_f_masked.min(), vmax = hist_apo_peri_f_masked.max()))
            plt.plot(self._r_array, self._r_array, c = rgb_palette_dict['turquiose'])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$r_{apo}$')
            plt.ylabel(r'$r_{peri}$')
            plt.colorbar()
            plt.savefig('./figures/'+self._name+'hist_r_apo_peri_f.pdf')
            plt.show()

        if self._M_tot == None:
            self._M_tot = self.M_enclosed(self._r_array[-1])
            self._M_system += self._M_tot
        mask_eaten = self.output_variables_dataframe['eaten'] == 0
        mask_errored = self.output_variables_dataframe['errored'] == 0
        mask = np.logical_and(mask_eaten, mask_errored)
        print(sum(mask))
        p_r_marginal = np.average(self.output_rseries_dataframe.loc[mask], axis = 0, weights = self.output_variables_dataframe.loc[mask, 'global_weight'])
        norm = np.trapz(p_r_marginal, self._r_array)
        rho_final_array = self._M_tot*(p_r_marginal/norm)/(4*np.pi*self._r_array**2)
        rho_final = UnivariateSpline(self._r_array, rho_final_array, k = 3, s = 0)

        if figures:
            bins = self._r_array[::10]
            p_r_t_0, edges = np.histogram(self.output_variables_dataframe['r_i'], bins = bins, density = True, weights = self.output_variables_dataframe['global_weight'])
            rho_sampled = self._M_tot*p_r_t_0/(4*np.pi*self._r_array[5::10]**2)

            fig, ax = plt.subplots(figsize = (4, 5))
            ax.plot(self._r_array, self._rho_array, c = rgb_palette_dict['rich black'], label = r'$\rho_i$')
            ax.plot(self._r_array[5::10], rho_sampled, c = rgb_palette_dict['amber'], label = r'$\rho_{sampled}$')
            ax.plot(self._r_array, rho_final_array,  c = rgb_palette_dict['flickr pink'], label = r'$\rho_f$')
            ax.set_ylim(bottom = 1E-11)
            ax.set_xlim(left = self._r_array[0])
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r'$\rho$')
            ax.legend()
            divider = make_axes_locatable(ax)
            ax_ratio = divider.append_axes("bottom", 1.2, pad=0.1, sharex=ax)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax_ratio.axhline(1, c = rgb_palette_dict['rich black'])
            ax_ratio.plot(self._r_array[5::10], rho_sampled/self._rho_array[5::10], c = rgb_palette_dict['amber'])
            ax_ratio.set_ylim(-0.1, 2.1)
            ax_ratio.set_xlabel(r'$r$ (pc)')
            ax_ratio.set_ylabel(r'$\rho/\rho_o$')
            plt.savefig('./figures/'+self._name+'_reconstructed_densities.pdf')
            plt.show()

        r_f_samples = np.zeros(self._N)
        new_weights_samples = np.zeros(self._N)

        for i in range(self._N):
            if not mask[i]:
                continue

            E_f = E_f_samples[i]
            L_i = self.output_variables_dataframe.loc[i, 'L_i']
            T_r_f = self.output_variables_dataframe.loc[i, 'T_r_f']
            r_peri_f = self.output_variables_dataframe.loc[i, 'r_peri_f']
            r_apo_f = self.output_variables_dataframe.loc[i, 'r_apo_f']

            # if T_r_f == 0: continue
            # if r_peri_f < 4*r_S: continue
            if np.isnan(T_r_f): continue

            p_r_given_E_L = lambda r: 1/np.sqrt(2*(psi_final(r) - E_f) - L_i**2/r**2) * 1/T_r_f

            ppf_r_f = lambda x: np.log10(r_apo_f/r_peri_f)*x + np.log10(r_peri_f)
            r_f = 10**ppf_r_f(np.random.random())
            r_f_samples[i] = r_f
            new_weights_samples[i] = p_r_given_E_L(r_f)*r_f*np.log10(r_apo_f/r_peri_f)

        zero_mask = r_f_samples != 0
        N_noneaten = len(r_f_samples[zero_mask])
        new_density = EvolvingDensity(self._name+'instantaneous', rho_final, self._r_array, N_noneaten)

        new_density.smoothen_density()
        new_density.add_external_potential_from_function(delta_psi, delta_M)

        new_density.output_variables_dataframe['E_i'] = E_f_samples[zero_mask]
        new_density.output_variables_dataframe['L_i'] = self.output_variables_dataframe.loc[zero_mask, 'L_i'].values
        new_density.output_variables_dataframe['T_r_i'] = self.output_variables_dataframe.loc[zero_mask, 'T_r_f'].values
        new_density.output_variables_dataframe['r_apo_i'] = self.output_variables_dataframe.loc[zero_mask, 'r_apo_f'].values
        new_density.output_variables_dataframe['r_peri_i'] = self.output_variables_dataframe.loc[zero_mask, 'r_peri_f'].values

        new_density.output_variables_dataframe['r_i'] = r_f_samples[zero_mask]
        new_density.output_variables_dataframe['global_weight'] = new_weights_samples[zero_mask] * self.output_variables_dataframe.loc[zero_mask, 'global_weight'].values
        new_density._sampled = True

        return new_density