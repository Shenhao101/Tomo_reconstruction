# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:50:36 2024

@author: shenhao
@mail: shenhao@mail.iggcas.ac.cn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
from scipy.optimize import curve_fit



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
models = ('TX2019slab', 'UU-P07', 'MITP08', 'LLNL_G3D_JPS', 'GLAD_M25')
slab_flux_models = []
carbon_flux_models = []
slab_flux_mean = 0
carbon_flux_mean = 0
for i in range(len(models)):
    file = '../Carbon_flux/Dis_sub1000_Dmax200_mean_newrate/flux_{}.txt'.format(models[i])
    age_flux, slab_flux, carbon_flux = load_carbon_flux(file)

    # delete outliers at 1 Ma for models MIT-P08 and TX2019slab
    if models[i]=='MITP08' or models[i]=='TX2019slab':
        slab_flux[0], carbon_flux[0] = 0, 0
    slab_flux_mean += slab_flux
    carbon_flux_mean += carbon_flux

    if models[i]=='MITP08' or models[i]=='TX2019slab':
        slab_flux[0], carbon_flux[0] = np.nan, np.nan
    slab_flux_models.append(slab_flux)
    carbon_flux_models.append(carbon_flux)

slab_flux_mean[1:] = slab_flux_mean[1:] / len(models)
slab_flux_mean[0] = slab_flux_mean[0] / (len(models)-2)
carbon_flux_mean[1:] = carbon_flux_mean[1:] / len(models)
carbon_flux_mean[0] = carbon_flux_mean[0] / (len(models)-2)
slab_flux_models = np.array(slab_flux_models)
carbon_flux_models = np.array(carbon_flux_models)

# Normalization
carbon_flux_normalized = carbon_flux_mean[0:65].copy()
carbon_flux_normalized /= carbon_flux_mean[0]


# save mean carbon flux
data = {}
data['# time'] = age_flux[0:65]
data['total_subducted_mean  (Mt C/yr)'] = carbon_flux_mean[0:65]



# load upper limit
slab_flux_upper_limit = 0
carbon_flux_upper_limit = 0
for i in range(len(models)):
    file = '../Carbon_flux/Dis_sub1200_Dmax200_max_newrate/flux_{}.txt'.format(models[i])
    age_flux, slab_flux, carbon_flux = load_carbon_flux(file)

    # delete outliers at 1 Ma for models MIT-P08 and TX2019slab
    if models[i]=='MITP08' or models[i]=='TX2019slab':
        slab_flux[0], carbon_flux[0] = 0, 0

    slab_flux_upper_limit += slab_flux 
    carbon_flux_upper_limit += carbon_flux

slab_flux_upper_limit[1:] = slab_flux_upper_limit[1:] / len(models)
slab_flux_upper_limit[0] = slab_flux_upper_limit[0] / (len(models)-2)
carbon_flux_upper_limit[1:] = carbon_flux_upper_limit[1:] / len(models)
carbon_flux_upper_limit[0] = carbon_flux_upper_limit[0] / (len(models)-2)

data['total_subducted_upper_limit  (Mt C/yr)'] = carbon_flux_upper_limit[0:65]


# load lower limit
slab_flux_lower_limit = 0
carbon_flux_lower_limit = 0
for i in range(len(models)):
    file = '../Carbon_flux/Dis_sub800_Dmax200_min_newrate/flux_{}.txt'.format(models[i])
    Age_flux, slab_flux, carbon_flux = load_carbon_flux(file)

    # delete outliers at 1 Ma for models MIT-P08 and TX2019slab
    if models[i]=='MITP08' or models[i]=='TX2019slab':
        slab_flux[0], carbon_flux[0] = 0, 0

    slab_flux_lower_limit += slab_flux 
    carbon_flux_lower_limit += carbon_flux

slab_flux_lower_limit[1:] = slab_flux_lower_limit[1:] / len(models)
slab_flux_lower_limit[0] = slab_flux_lower_limit[0] / len(models)
carbon_flux_lower_limit[1:] = carbon_flux_lower_limit[1:] / len(models)
carbon_flux_lower_limit[0] = carbon_flux_lower_limit[0] / len(models)
data['total_subducted_upper_limit  (Mt C/yr)'] = carbon_flux_upper_limit[0:65]


df = pd.DataFrame(data)
df.to_csv('subducted_carbon.csv', index=False)