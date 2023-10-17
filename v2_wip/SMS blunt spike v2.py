#%%
import matplotlib.pyplot as plt
import numpy as np

from copy import copy
import os
os.chdir('/Users/renske/Documents/MONZ/v2')

from scipy.interpolate import interp1d, UnivariateSpline
from matplotlib.colors import LogNorm, Normalize
from evolving_densities_v2 import BaseDensity, EvolvingDensity, smooth_Bspline
from polytropic import PolytropicSolver
from colourmap_maker import lch_colour_map, luv_colour_map

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
import sys
from importlib import reload
reload(sys.modules['evolving_densities_v2'])
from evolving_densities_v2 import BaseDensity, EvolvingDensity

#%%

G_N = 4.302e-3 #Units of (pc/solar mass) (km/s)^2
c_light = 2.9979e05 #km/s

r_array = np.logspace(-10, 3, num = 1301)

M_halo = 2E8
z = 20
R_vir = 1E3 #0.784*(Planck18.Om0/Planck18.Om(15))**(-1/3) * (M_halo*1E-8*Planck18.h)**(1/3) * (10/(z + 1)) / Planck18.h *1E3
c = 15
a_NFW = np.log(1 + c) - c/(1 + c)
rho_halo = M_halo/(4/3 * np.pi * R_vir**3)

rho_0_prime = rho_halo*c**3/(3*a_NFW)
R_s = R_vir/c

def rho_NFW(r):
    return rho_0_prime * (R_s/r) / (1 + r/R_s)**2 * np.exp(-r/R_vir)

rho_0_gas_i = 1E-23 * (3.0857E18)**3 / (1.9885E33) # M_sun/pc^3
r_c_i = 1E1
def rho_gas_initial(r):
    return rho_0_gas_i / (1 + r/r_c_i)**2

rho_0_gas_f = 1E-15 * (3.0857E18)**3 / (1.9885E33) # M_sun/pc^3
r_c_f = 1E-3
def rho_gas_final(r):
    return rho_0_gas_f / (1 + r/r_c_f)**2

plt.figure()
plt.loglog(r_array, rho_NFW(r_array), c = rgb_palette_dict['rich black'])
plt.loglog(r_array, rho_gas_initial(r_array), c = rgb_palette_dict['pine green'])
plt.loglog(r_array, rho_gas_final(r_array), c = rgb_palette_dict['turquiose'])
plt.ylim(bottom = 1E-4)
plt.show()

#%%

rho_M_data = np.genfromtxt('/Users/renske/Documents/MONZ/rho_M.csv', delimiter = ',')

logrho_M_interp = interp1d(rho_M_data[:, 0], np.log10(rho_M_data[:, 1]), kind = 'quadratic', bounds_error = False,
                           fill_value = (np.log10(rho_M_data[0, 1]), 0))
rho_M_interp = lambda x: 10**logrho_M_interp(x) * (3.0857E18)**3 / (1.9885E33)

x_test = np.linspace(0, np.max(rho_M_data[:, 0]), num = 1000)
x_test = np.append(x_test, np.max(rho_M_data[:, 0]) - np.logspace(-8, -1, num = 1000))
x_test = np.unique(x_test)

r_reconstructed = [0]
M_tot = 1E5 # M_sun

for i in range(len(x_test) - 1):

    dM = (x_test[i + 1] - x_test[i]) * M_tot # M_sun

    rho = (rho_M_interp(x_test[i]) + rho_M_interp(x_test[i + 1]))/2 

    r = r_reconstructed[-1]

    dr = np.cbrt(3*dM/(4*np.pi*rho) + r**3) - r

    r_reconstructed.append(r + dr)

    if i%100 == 0: print(r_reconstructed[-1], dr)

r_reconstructed = np.array(r_reconstructed)

rho_SMS_bloated_interp = interp1d(np.array(r_reconstructed), rho_M_interp(x_test), fill_value = (rho_M_interp(x_test[0]), 0), bounds_error = False)

m_proto = 1E3
n = 3/2
rho_c = 0.1 * (3.0857E18)**3 / (1.9885E33) # M_sun/pc^3
poly_solver = PolytropicSolver(n)
r_array_poly, rho_array_poly = poly_solver(rho_c, m_proto)
rho_proto_poly_interp = interp1d(r_array_poly, rho_array_poly, fill_value = (rho_array_poly[0], 0), bounds_error = False)

r_core = 5E-7

#%%

plt.figure()
plt.loglog(r_array, rho_NFW(r_array), c = rgb_palette_dict['dark sienna'], label = 'initial NFW')
plt.loglog(r_array, rho_gas_initial(r_array), c = rgb_palette_dict['pine green'], label = 'initial gas cloud')
plt.loglog(r_array, rho_gas_final(r_array), c = rgb_palette_dict['turquiose'], label = 'collapsed gas cloud')
plt.loglog(r_array, rho_SMS_bloated_interp(r_array), c = rgb_palette_dict['flickr pink'], label = r'$10^5 M_\odot$ SMS')
# plt.loglog(r_array, rho_proto_poly_interp(r_array), c = rgb_palette_dict['flickr pink'])
plt.xlim(1E-9, 1E3)
plt.ylim(bottom = 1E-4)
plt.xlabel(r'$r$ (pc)')
plt.ylabel(r'$\rho$ ($M_\odot$ pc$^{-3}$)')
plt.legend()
# plt.savefig('./figures/density_NFW_gas_star_densities.pdf')
plt.show()

#%%

density_NFW = EvolvingDensity('density_NFW_SMS', rho_NFW, r_array, N_particles = int(1E5))
psi_NFW = density_NFW.get_psi()
M_tot_NFW = density_NFW.M_enclosed(r_array[-1])

density_gas_initial = BaseDensity('density_gas_initial', rho_gas_initial, r_array)
psi_gas_i = density_gas_initial.get_psi()
M_tot_gas_i = density_gas_initial.M_enclosed(r_array[-1])

density_gas_final = BaseDensity('density_gas_final', rho_gas_final, r_array)
psi_gas_f = density_gas_final.get_psi()
M_tot_gas_f = density_gas_final.M_enclosed(r_array[-1])

# plt.figure()
# plt.loglog(r_array, psi_NFW(r_array), c = rgb_palette_dict['rich black'], label = 'initial NFW')
# plt.loglog(r_array, psi_gas_i(r_array), c = rgb_palette_dict['pine green'], label = 'initial gas cloud')
# plt.loglog(r_array, psi_gas_f(r_array), c = rgb_palette_dict['turquiose'], label = 'collapsed gas cloud')
# plt.xlabel('$r$ (pc)')
# plt.ylabel(r'$\psi$')
# plt.legend()
# plt.savefig('./figures/density_NFW_gas_potentials.pdf')
# plt.show()

print(M_tot_gas_i/1E5, M_tot_gas_f/1E5, M_tot_NFW/1E8)

#%%

density_proto_poly = BaseDensity('density_proto_poly', rho_proto_poly_interp, r_array)
psi_proto_poly = density_proto_poly.get_psi()
M_tot_proto_poly = density_proto_poly.M_enclosed(r_array[-1])

density_SMS_bloated = BaseDensity('density_SMS_bloated', rho_SMS_bloated_interp, r_array)
density_SMS_bloated.smoothen_density()
density_SMS_bloated._truncated = False
density_SMS_bloated.setup_potentials()
psi_SMS_bloated = density_SMS_bloated.get_psi()
M_tot_SMS_bloated = density_SMS_bloated.M_enclosed(r_array[-1])

#%%
plt.figure()
plt.loglog(r_array, psi_NFW(r_array), c = rgb_palette_dict['dark sienna'], label = 'initial NFW')
plt.loglog(r_array, psi_gas_i(r_array), c = rgb_palette_dict['pine green'], label = 'initial gas cloud')
plt.loglog(r_array, psi_gas_f(r_array), c = rgb_palette_dict['turquiose'], label = 'collapsed gas cloud')
plt.loglog(r_array, psi_SMS_bloated(r_array), c = rgb_palette_dict['flickr pink'], label = r'$10^5 M_\odot$ SMS')
plt.xlim(1E-9, 1E3)
plt.legend()
plt.xlabel('$r$ (pc)')
plt.ylabel(r'$\psi$')
# plt.savefig('./figures/density_NFW_gas_star_potentials.pdf')
plt.show()

print( np.log10(M_tot_proto_poly), np.log10(M_tot_SMS_bloated), np.log10(M_tot_NFW))

#%%

m_BH = M_tot_SMS_bloated*1.
r_S = 2*G_N*m_BH/c_light**2
def psi_BH(r):
    return G_N*m_BH/r

r_sp = 0.122 * R_s * ( m_BH / ( rho_0_prime * R_s**3 ) )**(1./(3.-1))

#%%

# density_GS_pred_no_r_S = density_NFW.adiabatic_growth(psi_BH, r_S = 0, figures = False)
density_GS_pred = density_NFW.adiabatic_growth(psi_BH, r_S = r_S, figures = False)

# rho_GS_no_r_S = density_GS_pred_no_r_S(r_array)
# rho_mask = rho_GS_no_r_S > 0
# rho_GS_no_r_S_smooth = smooth_Bspline(r_array[rho_mask], rho_GS_no_r_S[rho_mask], increasing = False)
# logrho_GS_no_r_S_spline = UnivariateSpline(np.log(r_array[rho_mask]), np.log(rho_GS_no_r_S_smooth(r_array)[rho_mask]), k = 2, s = 0)
# dlogrho_GS_no_r_S = logrho_GS_no_r_S_spline.derivative(n = 1)

rho_GS = density_GS_pred(r_array)
rho_mask = np.logical_and(rho_GS > 0, r_array > 4*r_S)
rho_GS_smooth = smooth_Bspline(r_array[rho_mask], rho_GS[rho_mask], increasing = False)
logrho_GS_spline = UnivariateSpline(np.log(r_array[rho_mask]), np.log(rho_GS_smooth(r_array)[rho_mask]), k = 2, s = 0)
dlogrho_GS = logrho_GS_spline.derivative(n = 1)

#%%

density_NFW.calculate_orbital_time_distribution(figures = True)

#%%

r_i_samples = density_NFW.output_variables_dataframe['r_i']
T_r_samples = density_NFW.output_variables_dataframe['T_r_i']* 3.0857E16 / 3.15576E16 # Myr
weight_samples = density_NFW.output_variables_dataframe['global_weight']

hist_ri_Tr, x_edges, y_edges = np.histogram2d(r_i_samples, T_r_samples, bins = (r_array[::10], np.logspace(np.log10(T_r_samples[T_r_samples != 0].min()), np.log10(T_r_samples.max()), num = 101)), 
                                    density = True, weights = weight_samples)
hist_ri_Tr_masked = np.ma.masked_equal(hist_ri_Tr.T, 0)

lch_cmap = lch_colour_map([rgb_palette_dict['dark sienna'], rgb_palette_dict['flickr pink'], rgb_palette_dict['blond']], N_bins = 256)
plt.figure()
plt.pcolormesh(x_edges, y_edges, hist_ri_Tr_masked,
                        cmap = lch_cmap, norm = LogNorm(vmin = hist_ri_Tr_masked.min(), vmax = hist_ri_Tr_masked.max()))
plt.axhline(2, c = rgb_palette_dict['turquiose'], ls = '--')
plt.axvline(r_sp, c = rgb_palette_dict['rich black'], ls = '--')
plt.xlim(1E-9, 1E3)
plt.ylim(np.min(y_edges[:-1][hist_ri_Tr[r_array[5::10] >= 1E-9][0] > 0]))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$r_i$ (pc)')
plt.ylabel(r'$T_r$ (Myr)')
plt.colorbar()
# plt.savefig('./figures/density_NFW_hist_ri_Tr.pdf')
plt.show()

#%%
mask = T_r_samples > 2
p_r_large_T, edges = np.histogram(r_i_samples[mask], bins = r_array[::10], density = True, weights = weight_samples[mask])
rho_large_T = M_tot_NFW*p_r_large_T/(4*np.pi*r_array[5::10]**2)
p_r_tot, edges = np.histogram(r_i_samples, bins = r_array[::10], density = True, weights = weight_samples)
rho_tot = M_tot_NFW*p_r_tot/(4*np.pi*r_array[5::10]**2)

plt.figure()
plt.semilogx(r_array[5::10], rho_large_T/rho_tot * (sum(weight_samples[mask])/sum(weight_samples)), c = rgb_palette_dict['flickr pink'])
plt.axvline(r_sp, c = rgb_palette_dict['rich black'], ls = '--')
plt.xlim(1E-9, 1E3)
plt.xlabel(r'$r_i$ (pc)')
plt.ylabel(r'$\rho(r | T_r > 2$ kyr$)/\rho(r)$')
# plt.savefig('./figures/density_NFW_fraction_large_T.pdf')
plt.show()

ind_sp = np.argmin(np.abs(r_array[5::10] - r_sp))
print(rho_large_T[ind_sp]/rho_tot[ind_sp])


#%%

rho_initial = density_NFW(r_array)
rho_mask = rho_initial > 0
rho_initial_smooth = smooth_Bspline(r_array[rho_mask], rho_initial[rho_mask], increasing = False)
logrho_initial_spline = UnivariateSpline(np.log(r_array[rho_mask]), np.log(rho_initial_smooth(r_array)[rho_mask]), k = 2, s = 0)
dlogrho_initial = logrho_initial_spline.derivative(n = 1)


#%%

# delta_psi = lambda r: psi_SMS_bloated(r) - psi_proto_poly(r)

density_after_SMS = density_NFW.adiabatic_growth(psi_SMS_bloated, M_tot_SMS_bloated, figures = True)
#%%

r_f_samples = density_after_SMS.output_variables_dataframe['r_i']
weights_samples = density_after_SMS.output_variables_dataframe['global_weight']
mask = np.isnan(weights_samples)
pdf_r_f = lambda r: 4*np.pi*r**2*density_after_SMS(r)/M_tot_NFW

hist_r_f, edges = np.histogram(r_f_samples[~mask], bins = r_array[::10], density = True, weights = weights_samples[~mask])

plt.figure()
plt.loglog(r_array, density_after_SMS(r_array), c = rgb_palette_dict['rich black'], label = r'Target $p(r_f)$')
plt.loglog(r_array[5::10], hist_r_f*M_tot_NFW / (4*np.pi*r_array[5::10]**2), color = rgb_palette_dict['flickr pink'], label = r'$p(r_f)$')
plt.xscale('log')
plt.yscale('log')
# plt.ylim(bottom = 1E-16)
plt.legend()
plt.xlabel(r'$r$')
plt.ylabel( r'$p(r)$')
plt.show()

#%%

plt.figure()
plt.loglog(r_array, rho_initial, c = rgb_palette_dict['dark sienna'], label = 'NFW')
# plt.loglog(r_array, rho_GS_no_r_S, c = rgb_palette_dict['amber'], label = r'GS, $m_{BH} = 10^5 M\odot$')
plt.loglog(r_array, density_after_SMS(r_array), c = rgb_palette_dict['flickr pink'], label = 'after SMS formation')
plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
plt.ylim(1E0, 1E25)
plt.xlim(1E-9, 1E3)
plt.legend()
plt.ylabel(r'$\rho$ ($M_\odot$pc$^{-3}$)')
plt.xlabel(r'$r$ (pc)')
# plt.savefig('./figures/density_NFW_rho_SMS.pdf')
plt.show()

#%%

rho_after_SMS = density_after_SMS(r_array)
rho_mask = rho_after_SMS > 0
rho_after_SMS_smooth = smooth_Bspline(r_array[rho_mask], rho_after_SMS[rho_mask], increasing = False)
logrho_after_SMS_spline = UnivariateSpline(np.log(r_array[rho_mask]), np.log(rho_after_SMS_smooth(r_array)[rho_mask]), k = 2, s = 0)
dlogrho_after_SMS = logrho_after_SMS_spline.derivative(n = 1)

#%%
plt.figure()
plt.semilogx(r_array, dlogrho_initial(np.log(r_array)), '.', c = rgb_palette_dict['dark sienna'], label = 'NFW')
# plt.semilogx(r_array, dlogrho_GS_no_r_S(np.log(r_array)), '.', c = rgb_palette_dict['amber'], label = r'GS, $m_{BH} = 10^5 \mathcal{M}_\odot$')
plt.semilogx(r_array, dlogrho_after_SMS(np.log(r_array)), '.', c = rgb_palette_dict['flickr pink'], label = 'after SMS formation')
# plt.axvline(r_S, c = rgb_palette_dict['rich black'], ls = '--')
plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
plt.ylim(-2.5, 0.5)
plt.xlim(1E-9, 1E3)
plt.legend()
plt.ylabel(r'd log$(\rho)$/d log($r$)')
plt.xlabel(r'$r$ (pc)')
# plt.savefig('./figures/density_NFW_dlogrho_SMS.pdf')
plt.figure()


#%%

delta_psi = lambda r: psi_BH(r) - psi_SMS_bloated(r)
density_after_DCBH = density_after_SMS.instantaneous_growth(delta_psi, 0., r_S = r_S, figures = True)

#%%

plt.figure()
plt.loglog(r_array, rho_initial, c = rgb_palette_dict['dark sienna'], label = 'NFW')
# plt.loglog(r_array, rho_GS, c = rgb_palette_dict['amber'], label = r'GS, $m_{BH} = 10^5 M\odot$')
plt.loglog(r_array, rho_after_SMS, c = rgb_palette_dict['flickr pink'], label = 'after SMS formation')
plt.loglog(r_array, density_after_DCBH(r_array), c = rgb_palette_dict['jazzberry jam'], label = 'after DCBH formation')
plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
plt.axvline(4*r_S, c = rgb_palette_dict['rich black'], ls = '--')
# plt.ylim(1E15, 1E23)
plt.xlim(1E-8, 1E-5)
plt.legend(facecolor = 'w', frameon = True, framealpha = 1, edgecolor = 'w')
plt.ylabel(r'$\rho$ ($M_\odot$pc$^{-3}$)')
plt.xlabel(r'$r$ (pc)')
# plt.savefig('./figures/density_NFW_rho_DCBH.pdf')
plt.show()

#%%

rho_after_DCBH = density_after_DCBH(r_array)
rho_mask = rho_after_DCBH > 0
rho_after_DCBH_smooth = smooth_Bspline(r_array[rho_mask], rho_after_DCBH[rho_mask], increasing = False)
logrho_after_DCBH_spline = UnivariateSpline(np.log(r_array[rho_mask]), np.log(rho_after_DCBH_smooth(r_array)[rho_mask]), k = 2, s = 0)
dlogrho_after_DCBH = logrho_after_DCBH_spline.derivative(n = 1)

#%%
plt.figure()
plt.semilogx(r_array, dlogrho_initial(np.log(r_array)), '.', c = rgb_palette_dict['dark sienna'], label = 'NFW')
# plt.semilogx(r_array[r_array > 4*r_S], dlogrho_GS(np.log(r_array[r_array > 4*r_S])), '.', c = rgb_palette_dict['amber'], label = r'GS, $m_{BH} = 10^5 \mathcal{M}_\odot$')
plt.semilogx(r_array, dlogrho_after_SMS(np.log(r_array)), '.', c = rgb_palette_dict['flickr pink'], label = 'after SMS formation')
plt.semilogx(r_array[r_array > 4*r_S], dlogrho_after_DCBH(np.log(r_array[r_array > 4*r_S])), '.', c = rgb_palette_dict['jazzberry jam'], label = 'after DCBH formation')
plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
plt.axvline(4*r_S, c = rgb_palette_dict['rich black'], ls = '--')
plt.ylim(-2.5, 0.5)
plt.xlim(1E-8, 1E-5)
plt.legend(facecolor = 'w', frameon = True, framealpha = 1, edgecolor = 'w')
plt.ylabel(r'd log$(\rho)$/d log($r$)')
plt.xlabel(r'$r$ (pc)')
# plt.savefig('./figures/density_NFW_dlogrho_DCBH.pdf')
plt.show()

#%%

mask_eaten = density_after_SMS.output_variables_dataframe['eaten'] != 0
mask_errored = density_after_SMS.output_variables_dataframe['errored'] == 2
mask = np.logical_or(mask_eaten, mask_errored)
r_i_samples = density_after_SMS.output_variables_dataframe['r_i']
weight_samples = density_after_SMS.output_variables_dataframe['global_weight']

r_i_samples_noteaten = r_i_samples[~mask].values
weight_samples_noteaten = weight_samples[~mask].values.reshape(len(r_i_samples_noteaten), 1)
rseries_noteaten = density_after_SMS.output_rseries_dataframe.loc[~mask].values
r_i_samples_all = r_i_samples[~mask_errored].values
weight_samples_all = weight_samples[~mask_errored].values.reshape(len(r_i_samples_all), 1)
rseries_all = density_after_SMS.output_rseries_dataframe.loc[~mask_errored].values

bins = r_array[::10]

p_r_ri_noteaten_grid = np.zeros((len(bins) - 1, len(r_array)))
p_r_ri_all_grid = np.zeros((len(bins) - 1, len(r_array)))

for i in range(len(bins) - 1):

    mask_r = np.logical_and(r_i_samples_noteaten >= bins[i], r_i_samples_noteaten < bins[i + 1])
    if np.sum(mask_r) != 0 and np.sum(rseries_noteaten[mask_r]) != 0:
        p_r_ri = np.sum(rseries_noteaten[mask_r]*weight_samples_noteaten[mask_r], axis = 0)
        p_r_ri_noteaten_grid[i] = p_r_ri

    mask_r = np.logical_and(r_i_samples_all >= bins[i], r_i_samples_all < bins[i + 1])
    if np.sum(mask_r) != 0 and np.sum(rseries_all[mask_r]) != 0:
        p_r_ri = np.sum(rseries_all[mask_r]*weight_samples_all[mask_r], axis = 0)
        p_r_ri_all_grid[i] = p_r_ri

#%%
p_r_ri_all_masked = np.ma.masked_equal(p_r_ri_all_grid.T, 0)
p_r_ri_all_normed = p_r_ri_all_masked/np.sum(p_r_ri_all_masked, axis = 1).reshape(len(r_array), 1)

lch_cmap = lch_colour_map([rgb_palette_dict['dark sienna'], rgb_palette_dict['flickr pink'], rgb_palette_dict['blond']], N_bins = 256)
plt.figure()
plt.pcolormesh(r_array[5::10][20:50], r_array[200:500], p_r_ri_all_normed[200:500, 20:50], cmap = lch_cmap, 
               norm = LogNorm(vmin = p_r_ri_all_normed[200:500, 20:50].min(), vmax = p_r_ri_all_normed[200:500, 20:50].max()))
plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
plt.axvline(4*r_S, c = rgb_palette_dict['rich black'], ls = '--')
plt.xlim(1E-8, 1E-5)
plt.ylim(4*r_S, 1E-5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$r_i$ (pc)')
plt.ylabel('$r$ (pc)')
plt.colorbar(label  = r'$p(r_i | r)$', location = 'top',
             fraction = 0.05, pad = 0)
plt.savefig('./figures/density_NFW_p_ri_r.pdf')
plt.show()

# p_r_ri_noteaten_masked = np.ma.masked_equal(p_r_ri_noteaten_grid.T, 0)
# p_r_ri_noteaten_normed = p_r_ri_noteaten_masked/np.sum(p_r_ri_noteaten_masked, axis = 1).reshape(len(r_array), 1)
#%%
fraction_noteaten = p_r_ri_noteaten_grid.T/p_r_ri_all_masked

luv_cmap = lch_colour_map([rgb_palette_dict['jazzberry jam'], rgb_palette_dict['blond']], N_bins = 256)

plt.figure()
plt.pcolormesh(r_array[5::10], r_array, fraction_noteaten, cmap = luv_cmap, 
               norm = Normalize(vmin = 0, vmax = 1))
plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
plt.axvline(4*r_S, c = rgb_palette_dict['rich black'], ls = '--')
plt.xlim(1E-8, 1E-5)
plt.ylim(4*r_S, 1E-5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$r_i$ (pc)')
plt.ylabel('$r$ (pc)')
# ax_divider = make_axes_locatable(ax)
# cax = ax_divider.append_axes("top", size = "7%", pad = "2%")
plt.colorbar(label  = r'contribution from uncaptured particles', 
             location = 'top', fraction = 0.05, pad = 0)
# cax.xaxis.set_ticks_position("top")
# cax.xaxis.set_label_position("top")
plt.savefig('./figures/density_NFW_fraction_eaten.pdf')
plt.show()

#%%
fraction = np.average(1 - fraction_noteaten, axis = 0)
mask_frac = fraction >= 0.99
r_cutoff = r_array[::10][:-1][mask_frac][-1]

plt.figure()
plt.semilogx(r_array[5::10], fraction, c = rgb_palette_dict['flickr pink'])
plt.axvline(4*r_S, c = rgb_palette_dict['rich black'], ls = '--')
plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
plt.xlabel(r'$r_i$ (pc)')
plt.ylabel(r'$\rho(r |$captured$)/\rho(r)$')
plt.show()
print(r_cutoff/r_core)

#%%

p_r_ri_noteaten_normed = p_r_ri_all_normed*fraction_noteaten

lch_cmap = lch_colour_map([rgb_palette_dict['dark sienna'], rgb_palette_dict['flickr pink'], rgb_palette_dict['blond']], N_bins = 256)
plt.figure()
plt.pcolormesh(r_array[5::10][20:50], r_array[200:500], p_r_ri_noteaten_normed[200:500, 20:50], cmap = lch_cmap, 
               norm = LogNorm(vmin = p_r_ri_all_normed[200:500, 20:50].min(), vmax = p_r_ri_all_normed[200:500, 20:50].max()))
plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
plt.axvline(4*r_S, c = rgb_palette_dict['rich black'], ls = '--')
plt.xlim(1E-8, 1E-5)
plt.ylim(4*r_S, 1E-5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$r_i$ (pc)')
plt.ylabel('$r$ (pc)')
plt.colorbar(label  = r'$p(r_i | r)$', location = 'top',
             fraction = 0.05, pad = 0)
# plt.savefig('./figures/density_NFW_p_ri_r_noteaten.pdf')
plt.show()
#%%



# %%
