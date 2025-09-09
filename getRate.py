# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:19:57 2020

@author: shenhao
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pykrige.ok import OrdinaryKriging


def upper_mantle():
    # load time-dependent subduction rate 
    subduction_data = pd.read_excel('shallow_vertical_rate.xlsx', engine='openpyxl')
    Age = subduction_data['Age']
    rate_mean = subduction_data['Subduction rate mean (cm/yr)']

    upper_rate = {}
    for i in range(len(Age)):
        upper_rate[str(Age[i])] = rate_mean[i] * 10 # unit: mm/yr
    
    return upper_rate # dict shape: age_length * 1



def lower_mantle():
    # load rate file
    slab_data = pd.read_excel('Lower_mantle_rate.xlsx', engine='openpyxl')
    Lon = slab_data['longitude']
    Lat = slab_data['latitude']
    Depth = slab_data['depth (km)']
    Age = slab_data['slab age']

    # calculate sinking rate in lower mantle
    upper_rate = upper_mantle()
    rate = []#lower rate of points
    for i in range (len(Lon)):

        # calculate the time needed to sink into the lower mantle
        slab_age = int(Age[i])
        if slab_age > 250:
            slab_age = 250
        
        subducted_depth = 0
        time = 0
        for j in range(slab_age):
            temp = subducted_depth + upper_rate[str(slab_age-j)]
            if temp <= 410:
                subducted_depth = temp
                time += 1
            else:
                break
        time += (410-subducted_depth) / upper_rate[str(slab_age-j)]
        
        # calculat slab sinking rate in the lower mantle 
        lower_rate = (Depth[i]-410) / (Age[i] - time)
        rate.append(lower_rate)
    
    # add background mean sinking rate 1.2 cm/yr at coarse grid 
    x = np.arange(-180, 181, 30)
    y = np.arange(-90, 91, 30)
    rate_mean = 12
    cutoff = 30
    Lon_coarse = []
    Lat_coarse = []
    rate_coarse = []
    for lon in x:
        for lat in y:
            flag = True
            for i in range(len(Lat)):
                distance = math.sqrt((lon-Lon[i])**2 + (lat-Lat[i])**2)
                if distance < cutoff:
                    flag = False
                    break
            if flag == True:
                Lon_coarse.append(lon)
                Lat_coarse.append(lat)
                rate_coarse.append(rate_mean)
    
    Lon_interp = np.concatenate((Lon, Lon_coarse))
    Lat_interp = np.concatenate((Lat, Lat_coarse))
    rate_interp = np.concatenate((rate, rate_coarse))


    # Kriging interpolation method
    x = np.array(range(-180,181))
    y = np.array(range(-90,91))

    OK = OrdinaryKriging(Lon_interp, Lat_interp, rate_interp, variogram_model='spherical',
                         variogram_parameters=[12.0, 50, 0.1], coordinates_type='geographic', exact_values=True)
    new_rate, ss = OK.execute('grid', x, y)
     
    return new_rate 


    
                
    
