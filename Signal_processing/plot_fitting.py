# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:06:05 2025

@author: shenhao
@mail:shenhao@mail.iggcas.ac.cn
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def load_reg_curve(file):
    f = open(file, 'r')
    age = []
    reg_curve = []
    f.readline()
    for each_line in f.readlines():
        each_line = each_line.strip('\n').split()
        age.append(float(each_line[0]))
        reg_curve.append(float(each_line[1]))
    age = np.array(age)
    reg_curve = np.array(reg_curve)
    
    return age, reg_curve



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


# load regression curve
file = 'fitting_curve_this_study.txt'
Age_reg, reg_curve_this_study = load_reg_curve(file)

file = 'fitting_curve_noise.txt'
Age_reg, reg_curve_noise = load_reg_curve(file)
    
file = 'fitting_curve_noise_filt.txt'
Age_reg, reg_curve_noise_filt = load_reg_curve(file)

# load CO2
file = 'CO2_CenCO2PIP_2023.csv'
Age_CO2, CO2 = load_CO2(file)


# plot
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8.5, 11.8))
fig.subplots_adjust(hspace=0.1)

ax1.plot(Age_reg, reg_curve_this_study, label='original fitting curve (R=0.76)', color='k')
ax1.plot(Age_CO2, CO2, label=r'Atmospheric CO$_{\mathbf{2}}$', color='tab:blue', linestyle='--')
# x axis setting
ax1.set_xlim([0, 66])
ax1.set_ylim([-0.2, 1])
ax1.invert_xaxis()
ax1.set_xticks([])
font={'family':'Arial', 'weight':'normal'}
ax1.legend(prop=font, loc='best')

ax2.plot(Age_reg, reg_curve_noise, label='fitting curve with noise (R=0.65)', color='k')
ax2.plot(Age_CO2, CO2, label=r'Atmospheric CO$_{\mathbf{2}}$', color='tab:blue', linestyle='--')
# x axis setting
ax2.set_xlim([0, 66])
ax2.set_ylim([-0.2, 1.0])
ax2.invert_xaxis()
ax2.set_xticks([])
ax2.legend(prop=font, loc='best')

ax3.plot(Age_reg, reg_curve_noise_filt, label='fitting curve after filter (R=0.70)', color='k')
ax3.plot(Age_CO2, CO2, label=r'Atmospheric CO$_{\mathbf{2}}$', color='tab:blue', linestyle='--')
# x axis setting
ax3.set_xlim([0, 66])
ax3.set_ylim([-0.2, 1.0])
ax3.invert_xaxis()
ax3.legend(prop=font, loc='best')
ax3.set_xlabel('Age (Ma)', fontproperties={'family':'Arial'}, fontsize=12)










