# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 15:17:47 2025

@author: m1335
"""
import numpy as np
import pandas as pd
from scipy.integrate import trapz
import matplotlib.pyplot as plt


def load_carbon_flux(file):
    f = open(file, 'r')
    age = []
    slab_flux = []
    carbon_flux = []
    f.readline()
    for each_line in f.readlines():
        each_line = each_line.strip('\n')
        each_line = each_line.split()
        age.append(float(each_line[0]))
        slab_flux.append(float(each_line[1]))
        carbon_flux.append(float(each_line[6]))
    age = np.array(age)
    slab_flux = np.array(slab_flux)
    carbon_flux = np.array(carbon_flux)
    
    return age, slab_flux, carbon_flux


# load mean data
Models = ('TX2019slab', 'UU-P07', 'LLNL_G3D_JPS', 'MITP08', 'GLAD_M25')
carbon_flux_mean = 0
for i in range(len(Models)):
    file = '../Carbon_flux/Dismax1000_Dmax200_mean_newrate/flux_{}.txt'.format(Models[i])
    Age_flux, slab_flux, carbon_flux = load_carbon_flux(file)

    # delete outliers at 1 Ma for models MIT-P08 and TX2019slab
    if Models[i]=='MITP08' or Models[i]=='TX2019slab':
        slab_flux[0], carbon_flux[0] = 0, 0
    carbon_flux_mean += carbon_flux

    if Models[i]=='MITP08' or Models[i]=='TX2019slab':
        slab_flux[0], carbon_flux[0] = np.nan, np.nan

carbon_flux_mean[1:] = carbon_flux_mean[1:] / len(Models)
carbon_flux_mean[0] = carbon_flux_mean[0] / (len(Models)-2)


# =============================================================================
# # diff
# dflux_dt = np.gradient(carbon_flux_mean, Age_flux)
# plt.plot(Age_flux, dflux_dt)
# =============================================================================

# window-slide integral
window = 5
local_integral_5 = []
center_age_5 = []

for i in range(window, len(Age_flux)-window):
    t_local = Age_flux[i-window : i+window+1]
    f_local = carbon_flux_mean[i-window : i+window+1]
    integral = trapz(f_local, t_local)
    local_integral_5.append(integral)
    center_age_5.append(Age_flux[i])

window = 10
local_integral_10 = []
center_age_10 = []

for i in range(window, len(Age_flux)-window):
    t_local = Age_flux[i-window : i+window+1]
    f_local = carbon_flux_mean[i-window : i+window+1]
    integral = trapz(f_local, t_local)
    local_integral_10.append(integral)
    center_age_10.append(Age_flux[i])

window = 15
local_integral_15 = []
center_age_15 = []

for i in range(window, len(Age_flux)-window):
    t_local = Age_flux[i-window : i+window+1]
    f_local = carbon_flux_mean[i-window : i+window+1]
    integral = trapz(f_local, t_local)
    local_integral_15.append(integral)
    center_age_15.append(Age_flux[i])
    
    
fig, ax = plt.subplots()
ax.plot(center_age_5, local_integral_5, label='5 Myr window')
ax.plot(center_age_10, local_integral_10, label='10 Myr window')
ax.plot(center_age_15, local_integral_15, label='15 Myr window')

font={'family':'Arial', 'weight':'normal', 'size':9}
ax.legend(prop=font, loc='best')
ax.set_xlabel('Age (Ma)', fontproperties={'family':'Arial'}, fontsize=11)
ax.set_ylabel(r'Cumulative carbon flux (Mt)', fontproperties={'family':'Arial'}, fontsize=11)
ax.set_xlim([0, 100])
ax.invert_xaxis()

plt.savefig('integral.png', format='png', bbox_inches='tight')
