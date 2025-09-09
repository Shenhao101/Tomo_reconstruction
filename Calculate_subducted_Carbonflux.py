# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:25:08 2024

@author: m1335
"""
from netCDF4 import Dataset
import math
import numpy as np
import getRate
import multiprocessing
import os

# parameters
R = 6371
# optional model: TX2019slab, UU-P07, LLNL_G3D_JPS, MITP08, GLAD_M25
model = 'GLAD_M25'
version = 'Dmax200'
limit = 'mean'
Dis_max = 1000 # maximum distance between positive anomaly and subduction zone
output_path = 'Carbon_flux/Dismax{}_{}_{}_newrate_without_subduction_zones'.format(Dis_max, version, limit)

upper_rate = getRate.upper_mantle() # dict shape: age_length * 1
lower_rate = getRate.lower_mantle() # shape: 181*361(-90~90, -180~180)



def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def read_reconstructed_tomography(age):
    fname = 'Reconstructed_TomographyModel/{}_{}_newrate/{}_{}.nc'.format(model, version, model, age)
    file = Dataset(fname)
    dV = file.variables['z'][:]
    # read mean positive velocity (MPV)
    MPV = file.variables['MPV'][:]
    return dV, MPV[0]


def read_SubductionZone_data(age):
    SubductionZone_data = []
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
            SubductionZone_data.append(each_line)
    return SubductionZone_data
                

def haversine_distance(depth, lat1, lon1, lat2, lon2):
    delta_lat = lat1 - lat2
    delta_lon = lon1 - lon2
    
    a = math.sin(math.radians(delta_lat/2))**2 +\
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *\
        math.sin(math.radians(delta_lon/2))**2
    d = 2 * (R-depth) * math.asin(math.sqrt(a))
    
    return d


def time_depth(age):

    # calculate the time needed to sink into the lower mantle
    subducted_depth = 0
    time = 0
    for j in range(age):
        temp = subducted_depth + upper_rate[str(age-j)]
        if temp <= 410:
            flag = False 
            subducted_depth = temp
            time += 1
        else:
            flag = True
            break
    
    time += (410-subducted_depth) / upper_rate[str(age-j)]
    
    # calculate slab depth at specified age 
    if flag == False: # slab in the upper mantle
        depth = np.full(lower_rate.shape, subducted_depth)
    else: # slab in the lower mantle
        depth = 410 + (age - time) * lower_rate

    depth_mean = np.average(depth, axis=None, weights=None)
    return depth



def search_nearest_SubductionZone(depth, latitude, longitude, data): 


    dis_min = haversine_distance(depth, latitude, longitude, data[0][0], data[0][1])
    data_min = data[0]
    for i in range(len(data)):
        distance = haversine_distance(depth, latitude, longitude, data[i][0], data[i][1])
        if distance <= dis_min:
            dis_min = distance 
            data_min = data[i]
    if dis_min < Dis_max:
        flag = 1
    else:
        flag = 1

    return data_min, flag
    

def calculate_flux(age, dv_limit):
    # read reconstructed tomography model
    dV, MPV = read_reconstructed_tomography(age)
    if age == 1:
        Interpdep_last = np.zeros((181, 361))
    else:
        Interpdep_last = time_depth(age-1)
    Interpdep = time_depth(age)

    # read subduction zone carbon data 
    SubductionZone_data = read_SubductionZone_data(age)

    # # calculate velocity anomaly limit that define the slab
    # # using the MPV(mean positive velocity) (Shephard et al., 2017)
    if limit == 'mean':
        dv_slab = MPV

    # lower limit of the flux    
    elif limit == 'min':
        if Interpdep.mean() < 410:
            dv_slab = dv_limit[0]
        else:
            dv_slab = dv_limit[2]

    # upper limit of the flux
    elif limit == 'max':
        if Interpdep.mean() < 410:
            dv_slab = dv_limit[1]
        else:
            dv_slab = dv_limit[3]
    
    print('Working at %s Ma. MPV= %s. dv_slab= %s'% (age, MPV, dv_slab))
    
    slab_flux = 0
    lithosphere_carbon_flux = 0
    serpentinite_carbon_flux = 0
    crust_carbon_flux = 0
    sediment_carbon_flux = 0
    total_carbon_flux = 0
    for i in range(181): # latitude:-90~90
        for j in range(361): # longitude:-180~180
            if dV[i][j] > dv_slab:
                
                # search for the nearest subduction zone 
                latitude = i - 90
                longitude = j - 180


                # # calculate flux in the Caribbean area
                # if latitude < 0 or latitude > 30:
                #     continue
                # if longitude < -100 or longitude > -50:
                #     continue



                SubductionZone_nearest_data, flag = search_nearest_SubductionZone(
                    Interpdep[i][j], latitude, longitude, SubductionZone_data
                )
                if flag == 0:
                    # the distance between positive anomaly and subduction zone exceed the cutoff
                    continue
                
                # calculate slab area
                # scale the distance according to the depth and latitude at each point 
                lat_d = (2 * math.pi * (R-Interpdep[i][j])) / 360 
                lon_d = lat_d * math.cos(math.radians(i-90))
                area = lat_d * lon_d

                
                # calculate slab volume
                if Interpdep_last[i][j] < 410 and Interpdep[i][j] > 410:
                    delta_dep = lower_rate[i][j]
                else:
                    delta_dep = Interpdep[i][j] - Interpdep_last[i][j]
                volume = area * delta_dep
                # convert km^3/Myr to km^3/yr
                volume *= 1e-6 
                slab_flux += volume 
                
                #calculate carbon flux, unit: Mt/yr
                lithosphere_carbon_flux += volume * SubductionZone_nearest_data[4]
                serpentinite_carbon_flux += volume * SubductionZone_nearest_data[5]
                crust_carbon_flux += volume * SubductionZone_nearest_data[6]
                sediment_carbon_flux += volume * SubductionZone_nearest_data[7]
                total_carbon_flux += volume * SubductionZone_nearest_data[8]
    flux = [age, slab_flux, lithosphere_carbon_flux, serpentinite_carbon_flux,
            crust_carbon_flux, sediment_carbon_flux, total_carbon_flux] 
    print('%s Ma has finished.'%age)
    return flux


if __name__ == '__main__':

    reconstruction_age = np.arange(1, 101)

    # calculate dv slab for the upper limit and lower limit
    MPV_max_upper_mantle = 0
    MPV_min_upper_mantle = 999
    MPV_max_lower_mantle = 0
    MPV_min_lower_mantle = 999

    for age in range(1,66):
        Interpdep = time_depth(age)
        dv, MPV = read_reconstructed_tomography(age)
        if Interpdep.mean() < 410:
            if MPV < MPV_min_upper_mantle:
                MPV_min_upper_mantle = MPV
            if MPV > MPV_max_upper_mantle:
                MPV_max_upper_mantle = MPV
        else:
            if MPV < MPV_min_lower_mantle:
                MPV_min_lower_mantle = MPV
            if MPV > MPV_max_lower_mantle:
                MPV_max_lower_mantle = MPV

    if model == 'MITP08':
        MPV_max_upper_mantle = 0
        for age in range(2,10):
            Interpdep = time_depth(age)
            dv, MPV = read_reconstructed_tomography(age)
            if Interpdep.mean() < 410:
                if MPV > MPV_max_upper_mantle:
                    MPV_max_upper_mantle = MPV

    dv_limit = [MPV_max_upper_mantle, MPV_min_upper_mantle, MPV_max_lower_mantle, MPV_min_lower_mantle]


    results = []
    Cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(processes=Cores)
    for age in reconstruction_age:
        results.append(p.apply_async(calculate_flux, args=(age, dv_limit)))
    p.close()
    p.join()
    
    
    all_flux_data = []
    for result in results:
        all_flux_data.append(result.get())
    all_flux_data = sorted(all_flux_data, key=lambda x: x[0])
    
    # save to file 
    mkdir(output_path)
    fname = '{}/flux_{}.txt'.format(output_path, model)
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



    
            


