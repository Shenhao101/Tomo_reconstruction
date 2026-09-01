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


def load_carbon_flux(file):
    f = open(file, 'r')
    age = []
    slab_flux = []
    carbon_flux = []
    carbon_flux_lithosphere = []
    carbon_flux_serpentinite = []
    carbon_flux_crust = []
    carbon_flux_sediment = []

    f.readline()
    for each_line in f.readlines():
        each_line = each_line.strip('\n')
        each_line = each_line.split()

        age.append(float(each_line[0]))
        slab_flux.append(float(each_line[1]))
        carbon_flux_lithosphere.append(float(each_line[2]))
        carbon_flux_serpentinite.append(float(each_line[3]))
        carbon_flux_crust.append(float(each_line[4]))
        carbon_flux_sediment.append(float(each_line[5]))
        carbon_flux.append(float(each_line[6]))

    age = np.array(age)
    slab_flux = np.array(slab_flux)
    carbon_flux = np.array(carbon_flux)

    carbon_flux_lithosphere = np.array(carbon_flux_lithosphere)
    carbon_flux_serpentinite = np.array(carbon_flux_serpentinite)
    carbon_flux_crust = np.array(carbon_flux_crust)
    carbon_flux_sediment = np.array(carbon_flux_sediment)
    carbon_flux_reservoirs = [carbon_flux_lithosphere, carbon_flux_serpentinite,
                              carbon_flux_crust, carbon_flux_sediment]
    carbon_flux_reservoirs = np.array(carbon_flux_reservoirs)
    return age, slab_flux, carbon_flux, carbon_flux_reservoirs


# load mean data
models = ('TX2019slab', 'UU-P07', 'MITP08', 'LLNL_G3D_JPS', 'GLAD_M25')
slab_flux_models = []
carbon_flux_models = []
slab_flux_mean = 0
carbon_flux_mean = 0
carbon_flux_reservoirs_mean = 0
model_num = 0 # the number of non-zoer values in models 

for i in range(len(models)):
    file = './Carbon_flux/Dis_sub1000_Dmax200_mean_correction/flux_{}.txt'.format(models[i])
    age_flux, slab_flux, carbon_flux, carbon_flux_reservoirs = load_carbon_flux(file)

    # delete outliers at 1 Ma for models MIT-P08 and TX2019slab
    if models[i]=='MITP08' or models[i]=='TX2019slab':
        slab_flux[0], carbon_flux[0] = 0, 0
        carbon_flux_reservoirs[:,0] = 0

    slab_flux_mean += slab_flux
    carbon_flux_mean += carbon_flux
    carbon_flux_reservoirs_mean += carbon_flux_reservoirs
    carbon_flux_reservoirs[carbon_flux_reservoirs>0] = 1
    model_num += carbon_flux_reservoirs

slab_flux_mean[1:] = slab_flux_mean[1:] / len(models)
slab_flux_mean[0] = slab_flux_mean[0] / (len(models)-2)
carbon_flux_mean[1:] = carbon_flux_mean[1:] / len(models)
carbon_flux_mean[0] = carbon_flux_mean[0] / (len(models)-2)
carbon_flux_reservoirs_mean = carbon_flux_reservoirs_mean / model_num

carbon_flux_lithosphere_mean = carbon_flux_reservoirs_mean[0]
carbon_flux_serpentinite_mean = carbon_flux_reservoirs_mean[1]
carbon_flux_crust_mean = carbon_flux_reservoirs_mean[2]
carbon_flux_sediment_mean = carbon_flux_reservoirs_mean[3]


# load upper limit
slab_flux_max = 0
carbon_flux_max = 0
carbon_flux_reservoirs_max = 0
model_num = 0 # the number of non-zoer values in models 

for i in range(len(models)):
    file = './Carbon_flux/Dis_sub1200_Dmax200_max_correction/flux_{}.txt'.format(models[i])
    age_flux, slab_flux, carbon_flux, carbon_flux_reservoirs = load_carbon_flux(file)

    # delete outliers at 1 Ma for models MIT-P08 and TX2019slab
    if models[i]=='MITP08' or models[i]=='TX2019slab':
        slab_flux[0], carbon_flux[0] = 0, 0
        carbon_flux_reservoirs[:,0] = 0

    slab_flux_max += slab_flux 
    carbon_flux_max += carbon_flux
    carbon_flux_reservoirs_max += carbon_flux_reservoirs
    carbon_flux_reservoirs[carbon_flux_reservoirs>0] = 1
    model_num += carbon_flux_reservoirs

slab_flux_max[1:] = slab_flux_max[1:] / len(models)
slab_flux_max[0] = slab_flux_max[0] / (len(models)-2)
carbon_flux_max[1:] = carbon_flux_max[1:] / len(models)
carbon_flux_max[0] = carbon_flux_max[0] / (len(models)-2)
carbon_flux_reservoirs_max = carbon_flux_reservoirs_max / model_num

carbon_flux_lithosphere_max = carbon_flux_reservoirs_max[0]
carbon_flux_serpentinite_max = carbon_flux_reservoirs_max[1]
carbon_flux_crust_max = carbon_flux_reservoirs_max[2]
carbon_flux_sediment_max = carbon_flux_reservoirs_max[3]


# load lower limit
slab_flux_min = 0
carbon_flux_min = 0
carbon_flux_reservoirs_min = 0
model_num = 0 # the number of non-zoer values in models 

for i in range(len(models)):
    file = './Carbon_flux/Dis_sub800_Dmax200_min_correction/flux_{}.txt'.format(models[i])
    Age_flux, slab_flux, carbon_flux, carbon_flux_reservoirs = load_carbon_flux(file)

    # delete outliers at 1 Ma for models MIT-P08 and TX2019slab
    if models[i]=='MITP08' or models[i]=='TX2019slab':
        slab_flux[0], carbon_flux[0] = 0, 0
        carbon_flux_reservoirs[:,0] = 0

    slab_flux_min += slab_flux 
    carbon_flux_min += carbon_flux
    carbon_flux_reservoirs_min += carbon_flux_reservoirs
    carbon_flux_reservoirs[carbon_flux_reservoirs>0] = 1
    model_num += carbon_flux_reservoirs

slab_flux_min[1:] = slab_flux_min[1:] / len(models)
slab_flux_min[0] = slab_flux_min[0] / len(models)
carbon_flux_min[1:] = carbon_flux_min[1:] / len(models)
carbon_flux_min[0] = carbon_flux_min[0] / len(models)
carbon_flux_reservoirs_min = carbon_flux_reservoirs_min / model_num

carbon_flux_lithosphere_min = carbon_flux_reservoirs_min[0]
carbon_flux_serpentinite_min = carbon_flux_reservoirs_min[1]
carbon_flux_crust_min = carbon_flux_reservoirs_min[2]
carbon_flux_sediment_min = carbon_flux_reservoirs_min[3]


# save
data = {}
data['# time'] = age_flux[0:65]
data['total_carbon_flux_mean (Mt C/yr)'] = carbon_flux_mean[0:65]
data['total_carbon_flux_max (Mt C/yr)'] = carbon_flux_max[0:65]
data['total_carbon_flux_min (Mt C/yr)'] = carbon_flux_min[0:65]
data['slab_flux (km3/yr)'] = slab_flux_mean[0:65]
data['slab_flux_max (km3/yr)'] = slab_flux_max[0:65]
data['slab_flux_min (km3/yr)'] = slab_flux_min[0:65]
data['lithosphere_carbon_flux_mean (Mt C/yr)'] = carbon_flux_lithosphere_mean[0:65]
data['lithosphere_carbon_flux_max (Mt C/yr)'] = carbon_flux_lithosphere_max[0:65]
data['lithosphere_carbon_flux_min (Mt C/yr)'] = carbon_flux_lithosphere_min[0:65]
data['serpentinite_carbon_flux_mean (Mt C/yr)'] = carbon_flux_serpentinite_mean[0:65]
data['serpentinite_carbon_flux_max(Mt C/yr)'] = carbon_flux_serpentinite_max[0:65]
data['serpentinite_carbon_flux_min (Mt C/yr)'] = carbon_flux_serpentinite_min[0:65]
data['crust_carbon_flux_mean (Mt C/yr)'] = carbon_flux_crust_mean[0:65]
data['crust_carbon_flux_max (Mt C/yr)'] = carbon_flux_crust_max[0:65]
data['crust_carbon_flux_min (Mt C/yr)'] = carbon_flux_crust_min[0:65]
data['crust_sediment_flux_mean (Mt C/yr)'] = carbon_flux_sediment_mean[0:65]
data['crust_sediment_flux_max (Mt C/yr)'] = carbon_flux_sediment_max[0:65]
data['crust_sediment_flux_min (Mt C/yr)'] = carbon_flux_sediment_min[0:65]
df = pd.DataFrame(data)
df.to_csv('subducted_carbon.csv', index=False)