# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:25:08 2024

Calculating the global carbon flux subducted into the mantle throughout geological time.

@author: m1335
"""
from netCDF4 import Dataset
import math
import numpy as np
import getRate
import multiprocessing
import os
import pandas as pd
from scipy.interpolate import interp1d


# parameters
EARTH_RADIUS = 6371
# optional model: TX2019slab, UU-P07, LLNL_G3D_JPS, MITP08, GLAD_M25
MODEL = 'LLNL_G3D_JPS'
VERSION = 'Dmax200'
LIMIT = 'min' # Selection of the carbon flux limit: mean, min, max
DIS_SUB= 800 # maximum distance between positive anomaly and subduction zone
OUTPUT_PATH = 'Carbon_flux/Dis_sub{}_{}_{}_correction'.format(DIS_SUB, VERSION, LIMIT)
UPPER_RATE = getRate.upper_mantle() # dict shape: age_length * 1
LOWER_RATE = getRate.lower_mantle() # shape: 181*361(-90~90, -180~180)
COMPRRESSION_CORRECTION = True


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def read_reconstructed_tomography(age):
    fname = 'Reconstructed_TomographyModel/{}_{}_newrate/{}_{}.nc'.format(MODEL, VERSION, MODEL, age)
    file = Dataset(fname)
    dV = file.variables['z'][:]
    # read mean positive velocity (MPV)
    mpv = file.variables['MPV'][:]
    return dV, mpv[0]


def read_subduction_zone_data(age):
    subduction_zone_data = []
    fname = 'Carbon_VolumeDensity_SubductionZone/mean/carbon_volume_density_{}.txt'.format(age)
    with open(fname, 'r') as file:
        file.readline()
        for each_line in file.readlines():
            if each_line.startswith('='):
                continue
            if each_line.startswith('*'):
                continue
            each_line = each_line.strip()
            each_line = each_line.split()
            for i in range(len(each_line)):
                each_line[i] = float(each_line[i])
            subduction_zone_data.append(each_line)
    return subduction_zone_data
                

def read_ak135(file):
    depth = []
    density = []
    subduction_data = pd.read_excel(file, engine='openpyxl')
    with open(file, 'r') as f:
        f.readline()
        for line in f.readlines():
            line = line.strip().split()
            depth.append(float(line[0]))
            density.append(float(line[1]))

    return np.array(depth), np.array(density)


def haversine_distance(depth, lat1, lon1, lat2, lon2):
    delta_lat = lat1 - lat2
    delta_lon = lon1 - lon2
    
    a = math.sin(math.radians(delta_lat/2))**2 +\
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *\
        math.sin(math.radians(delta_lon/2))**2
    d = 2 * (EARTH_RADIUS - depth) * math.asin(math.sqrt(a))
    
    return d


def time_depth(age):
    """
    calculating corresponding depth of subducted slabs at specific age.

    Parameters:
    -----------
    age: int
        Geological age in Ma.
    
    Returns:
    --------
    depth : ndarray
        2D array of global slab depths (km).
    depth_mean: float
        Mean depth (km).
    """

    # calculate the time needed for the slab to sink into the lower mantle
    subducted_depth = 0
    time_upper = 0
    
    for j in range(age):
        temp = subducted_depth + UPPER_RATE[str(age-j)]
        if temp <= 410:
            flag = False 
            subducted_depth = temp
            time_upper += 1
        else:
            flag = True
            break
    time_upper += (410-subducted_depth) / UPPER_RATE[str(age-j)]
    
    # calculate slab depth at specified age 
    if flag == False: # slab in the upper mantle
        depth = np.full(LOWER_RATE.shape, subducted_depth)
    else: # slab in the lower mantle
        depth = 410 + (age - time_upper) * LOWER_RATE

    return depth


def search_nearest_subduction_zone(depth, latitude, longitude, data): 
    """
    Find the nearest subduction-zone point to a given location.

    Parameters
    ----------
    depth : float
        Depth of the target point, in km.
    latitude : float
        Latitude of the target point, in degrees.
    longitude : float
        Longitude of the target point, in degrees.
    data : list
        List of subduction-zone carbon data.

    Returns
    -------
    data_min : list
        The nearest subduction-zone point to the target location.
    flag : int (0 or 1)
        Flag indicating whether the nearest point satisfies the
        specified distance criterion.
    """

    dis_min = haversine_distance(depth, latitude, longitude, data[0][0], data[0][1])
    data_min = data[0]
    for i in range(len(data)):
        distance = haversine_distance(depth, latitude, longitude, data[i][0], data[i][1])
        if distance <= dis_min:
            dis_min = distance 
            data_min = data[i]
    if dis_min < DIS_SUB:
        flag = 1
    else:
        flag = 0

    return data_min, flag
    

def calculate_flux(age, dv_limit):
    # read reconstructed tomography model
    dv, mpv = read_reconstructed_tomography(age)
    if age == 1:
        interp_dep_last = np.zeros((181, 361))
    else:
        interp_dep_last = time_depth(age-1)
    interp_dep = time_depth(age)

    # read subduction zone carbon data 
    subduction_zone_data = read_subduction_zone_data(age)

    # calculate velocity anomaly limit that define the slab
    # using the MPV(mean positive velocity) (Shephard et al., 2017)
    if LIMIT == 'mean':
        dv_slab = mpv
    # lower limit of the flux    
    elif LIMIT == 'min':
        if interp_dep.mean() < 410:
            dv_slab = dv_limit[0]
        else:
            dv_slab = dv_limit[2]

    # upper limit of the flux
    elif LIMIT == 'max':
        if interp_dep.mean() < 410:
            dv_slab = dv_limit[1]
        else:
            dv_slab = dv_limit[3]
    
    print('Working at %s Ma. MPV= %s. dv_slab= %s'% (age, mpv, dv_slab))


    if COMPRRESSION_CORRECTION == True:

        ak135_model = pd.read_excel('AK135_density.xlsx', engine='openpyxl')
        ak135_depth = ak135_model['Depth(km)'].astype(float)
        ak135_density = ak135_model['Rho(kg/m^3)'].astype(float)
        density_func = interp1d(
            ak135_depth,
            ak135_density,
            kind='linear',
        )

        # Using density at 120 km as the reference of subducted oceanic lithosphere
        # exclude the effect from crust or sediment in the shallow depth
        density_ref = 3.371


    slab_flux = 0
    lithosphere_carbon_flux = 0
    serpentinite_carbon_flux = 0
    crust_carbon_flux = 0
    sediment_carbon_flux = 0
    total_carbon_flux = 0
    for i in range(181): # latitude:-90~90
        for j in range(361): # longitude:-180~180
            if dv[i][j] > dv_slab:
                
                # search for the nearest subduction zone 
                latitude = i - 90
                longitude = j - 180

                # calculate flux in the Caribbean area
                # if latitude < 0 or latitude > 30:
                #     continue
                # if longitude < -100 or longitude > -50:
                #     continue

                subduction_zone_nearest_data, flag = search_nearest_subduction_zone(
                    interp_dep[i][j], latitude, longitude, subduction_zone_data
                )
                if flag == 0:
                    # the distance between positive anomaly and subduction zone exceed the cutoff
                    continue
                
                # calculate slab area
                # scale the distance according to the depth and latitude at each point 
                lat_d = (2 * math.pi * (EARTH_RADIUS - interp_dep[i][j])) / 360 
                lon_d = lat_d * math.cos(math.radians(i-90))
                area = lat_d * lon_d

                # 1D self-compression correction using AK135 model (van der meer et al., 2014)
                if COMPRRESSION_CORRECTION == True:
                    if interp_dep[i][j] > 120:
                        density_dep = density_func(interp_dep[i][j])
                        compression_factor = density_dep / density_ref
                        area *= compression_factor**(2 / 3) # isotropic compression assumption
                
                # calculate slab volume
                if interp_dep_last[i][j] < 410 and interp_dep[i][j] > 410:
                    delta_dep = LOWER_RATE[i][j]
                else:
                    delta_dep = interp_dep[i][j] - interp_dep_last[i][j]
                volume = area * delta_dep

                # convert km^3/Myr to km^3/yr
                volume *= 1e-6 
                slab_flux += volume 
                
                #calculate carbon flux, unit: Mt/yr
                lithosphere_carbon_flux += volume * subduction_zone_nearest_data[4]
                serpentinite_carbon_flux += volume * subduction_zone_nearest_data[5]
                crust_carbon_flux += volume * subduction_zone_nearest_data[6]
                sediment_carbon_flux += volume * subduction_zone_nearest_data[7]
                total_carbon_flux += volume * subduction_zone_nearest_data[8]
    flux = [age, slab_flux, lithosphere_carbon_flux, serpentinite_carbon_flux,
            crust_carbon_flux, sediment_carbon_flux, total_carbon_flux] 
    print('%s Ma has finished.'%age)
    return flux


if __name__ == '__main__':

    reconstruction_age = np.arange(1, 101)

    # Calculate the upper limit and lower limit of dv slab
    mpv_max_upper_mantle = -float('inf')
    mpv_min_upper_mantle = float('inf')
    mpv_max_lower_mantle = -float('inf')
    mpv_min_lower_mantle = float('inf')

    for age in range(1,66):
        interp_dep = time_depth(age)
        dv, mpv = read_reconstructed_tomography(age)
        if interp_dep.mean() < 410:
            if mpv < mpv_min_upper_mantle:
                mpv_min_upper_mantle = mpv
            if mpv > mpv_max_upper_mantle:
                mpv_max_upper_mantle = mpv
        else:
            if mpv < mpv_min_lower_mantle:
                mpv_min_lower_mantle = mpv
            if mpv > mpv_max_lower_mantle:
                mpv_max_lower_mantle = mpv

    if MODEL == 'MITP08':
        # Ignore extremely high anpmalies of MITP08 model in the crust layer
        # The reason of these amonalies may result from their inversion method?
        mpv_max_upper_mantle = -float('inf')
        for age in range(2,10):
            interp_dep = time_depth(age)
            dv, mpv = read_reconstructed_tomography(age)
            if interp_dep.mean() < 410:
                if mpv > mpv_max_upper_mantle:
                    mpv_max_upper_mantle = mpv

    dv_limit = [mpv_max_upper_mantle, mpv_min_upper_mantle, mpv_max_lower_mantle, mpv_min_lower_mantle]


    results = []
    cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(processes=cores)
    for age in reconstruction_age:
        results.append(p.apply_async(calculate_flux, args=(age, dv_limit)))
    p.close()
    p.join()
    
    
    all_flux_data = []
    for result in results:
        all_flux_data.append(result.get())
    all_flux_data = sorted(all_flux_data, key=lambda x: x[0])
    
    # save to file 
    mkdir(OUTPUT_PATH)
    fname = '{}/flux_{}.txt'.format(OUTPUT_PATH, MODEL)
    with open(fname, 'w') as file:
        # write header
        string = 'Age(Ma)'  + ' ' * 4 # length: 11
        file.write(string)
        string = 'Slab_Flux(km^3/yr)'  + ' ' * 4 # length: 22
        file.write(string)
        string = 'Lithosphere_Carbon_Flux(Mt/yr)'  + ' ' * 4 # length: 34
        file.write(string)
        string = 'Serpentinite_Carbon_Flux(Mt/yr)'  + ' ' * 4 # length: 35
        file.write(string)
        string = 'Crust_Carbon_Flux(Mt/yr)'  + ' ' * 4 # length: 28
        file.write(string)
        string = 'Sediment_Carbon_Flux(Mt/yr)'  + ' ' * 4 # length: 31
        file.write(string)
        string = 'Total_Carbon_Flux(Mt/yr)'  + ' ' * 4 # length: 28
        file.write(string + '\n')
    
        # write data
        for i in range(len(all_flux_data)):
            string = '%s' % all_flux_data[i][0]
            file.write(string + ' '*(11 - len(string)))
            
            string = '%.2f' % all_flux_data[i][1]
            file.write(string + ' '*(22 - len(string)))
        
            string = '%.2f' % all_flux_data[i][2]
            file.write(string + ' '*(34 - len(string)))
        
            string = '%.2f' % all_flux_data[i][3]
            file.write(string + ' '*(35 - len(string)))
        
            string = '%.2f' % all_flux_data[i][4]
            file.write(string + ' '*(28 - len(string)))
        
            string = '%.2f' % all_flux_data[i][5]
            file.write(string + ' '*(31 - len(string)))
        
            string = '%.2f' % all_flux_data[i][6]
            file.write(string + ' '*(28 - len(string)))
        
            file.write('\n')



    
            


