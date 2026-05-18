# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:27:38 2025

continuous wavelet tranform for geological data

@author: shenhao
@mail: shenhao@mail.iggcas.ac.cn
"""
import csv
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from obspy.signal.tf_misfit import cwt, plot_tfr
from obspy.imaging.cm import obspy_sequential




def load_CO2(file):
    f = open(file, 'r', encoding=u'utf-8', errors='ignore')
    reader = csv.reader(f)
    reader = list(reader)
    age = []
    CO2 = []
    for i in range(len(reader)):
        age.append(float(reader[i][0]))
        CO2.append(float(reader[i][1]))
    age = np.array(age)
    CO2 = np.array(CO2)
    
    # interpolate to 1~65 with interval of 1 Ma
    age_interp = np.arange(1,66)
    f = interpolate.interp1d(age, CO2)
    CO2_interp = f(age_interp)
    
    
    # normalization
    CO2_max = CO2_interp.max()
    CO2_min = CO2_interp.min()
    CO2_normalization = (CO2_interp-CO2_min) / (CO2_max-CO2_min)

    return age_interp, CO2_normalization



def load_carbon_flux(file):
    df = pd.read_csv(file)
    age = df['# time'][0:65]
    carbon_flux = df['total_subducted_mean  (Mt C/yr)'][0:65]

    # Normalization
    flux_max = carbon_flux.max()
    flux_min = carbon_flux.min()
    carbon_flux_normalized = (carbon_flux-flux_min) / (flux_max-flux_min)
    
    return np.array(age), np.array(carbon_flux_normalized)



def load_Sr_isotope(file):
    Age_Sr = []
    Sr_ratio = []
    with open(file, 'r') as f:
        for eachline in f.readlines():
            eachline = eachline.strip('\n')
            eachline = eachline.split()
            Age_Sr.append(float(eachline[0]))
            Sr_ratio.append(float(eachline[1]))
    
    Age_Sr = np.array(Age_Sr)
    Sr_ratio = np.array(Sr_ratio)


    # nomalization
    Sr_max = Sr_ratio.max()
    Sr_min = Sr_ratio.min()
    Sr_ratio_normalization = (Sr_ratio-Sr_min) / (Sr_max-Sr_min)
    return Age_Sr, Sr_ratio_normalization
    
    
    
# read geological data
file = 'CO2_CenCO2PIP_2023.csv'
Age_CO2, CO2 = load_CO2(file)

file = 'subducted_carbon.csv'
Age_flux, carbon_flux_mean = load_carbon_flux(file)

# load Sr isotope
Age_Sr, Sr_ratio = load_Sr_isotope('Sr_ratio_fit.txt')


# =============================================================================
# # generate a random signal with main frequency of 5 Myr
# freq = 0.2 # main frequency
# fs = 1 # sampling rate
# A = 0.2
# start_T = 1
# T = 65
# phi = 0 # initial phase
# t = np.linspace(start_T, T, fs*T, endpoint=True)
# noise = np.random.normal(0, 0.05, len(t))
# y = A * np.sin(2 * np.pi * freq * t + phi) + noise
# carbon_flux_noise = carbon_flux_mean + y
# 
# 
# # save random signal
# data = {}
# data['# time'] = t
# data['random_signal'] = y
# df = pd.DataFrame(data)
# df.to_csv('random_signal.csv', index=False)
# =============================================================================
 

# continuous wavelet transform 
dt = 1 # Myr
bandwidth_resolution = 10 # selected from Prokoph, 2008
f_max = 0.5 # corresponding to minimum wavelength 2 Myr
# f_min = 0.03125 # corresponding to maximum wavelength 32 Myr
f_min = 0.015625 # corresponding to maximum wavelength 64 Myr
spectrogram = cwt(st=CO2, dt=dt, w0=bandwidth_resolution, fmin=f_min, fmax=f_max, nf=100, wl='morlet')



# plot spectrogram 
fig = plt.figure()
ax = fig.add_subplot(111)

X, Y = np.meshgrid(
    Age_flux,
    np.logspace(np.log10(f_min), np.log10(f_max), spectrogram.shape[0])
    )


# plot_tfr(carbon_flux_mean, dt=dt, fmin=f_min, fmax=f_max)


cax = ax.pcolormesh(X, Y, np.abs(spectrogram), cmap=obspy_sequential, vmin=0, vmax=0.8, shading='gouraud')

ax.set_yscale('log')
# translate frequency axis to wavelength axis
yticks = [0.015625, 0.03125, 0.0625, 0.125, 0.25]
ylable = [64, 32, 16, 8, 4]
ax.set_yticks(yticks)
ax.set_yticklabels(ylable)
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel('Age (Ma)', fontproperties={'family':'Arial'}, fontsize=12)
ax.set_ylabel('Period (Myr)', fontproperties={'family':'Arial'}, fontsize=12)

cbar = fig.colorbar(cax, ax=ax)

plt.savefig('spectrogram_CO2.png', format='png', dpi=600)




# =============================================================================
# fig = plot_tfr(carbon_flux_noise, dt=dt, fmin=f_min, fmax=f_max, w0=10, show=False)
# axes = fig.get_axes()
# yticks = [0.015625, 0.03125, 0.0625, 0.125, 0.25]
# ylable = [64, 32, 16, 8, 4]
# axes[2].set_yticks(yticks)
# axes[2].set_yticklabels(ylable)
# axes[2].yaxis.set_minor_locator(NullLocator())
# axes[2].set_ylabel('Period (Myr)', fontproperties={'family':'Arial'}, fontsize=12)
# axes[0].set_xlabel('Age (Ma)', fontproperties={'family':'Arial'}, fontsize=12)
# 
# =============================================================================
