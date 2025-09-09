# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:42:38 2024

1. extract the location of past subduction zone from plate motion model
2. calculate plate thickness at subduction zone
3. calculate carbonate volume density at subduction zone
@author: shenhao
@email: shenhao@mail.iggcas.ac.cn
"""
import glob
import math
import numpy as np
from scipy import interpolate, special
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gplately
from gplately import pygplates
import multiprocessing


def SubductionZone(rotation_model, topology_features, age):
    # Resolve our topological plate polygons (and deforming networks) to the current 'time'.
    resolved_topologies = []
    pygplates.resolve_topologies(topology_features, rotation_model, resolved_topologies, age)
    
    subduction = []
    for resolved_topology in resolved_topologies:
        boundary_sub_segments = resolved_topology.get_boundary_sub_segments()
        for boundary_sub_segment in boundary_sub_segments:
            if boundary_sub_segment.get_resolved_feature().get_feature_type() == pygplates.FeatureType.gpml_subduction_zone:
                sub_segment_points = boundary_sub_segment.get_resolved_geometry().to_lat_lon_list()
                subduction.append(sub_segment_points)

    # refine the subduction zone to the specified interval
    interval = 1.0
    subduction_new = []
    for i in range(len(subduction)):
        subduction_temp = []
        for j in range(len(subduction[i])-1):
            lat1=subduction[i][j][0]; lon1=subduction[i][j][1]
            lat2=subduction[i][j+1][0]; lon2=subduction[i][j+1][1]
            lat_diff = lat1 - lat2
            lon_diff = lon1 - lon2
            diff = max(abs(lat_diff), abs(lon_diff))
            
            if diff > interval and diff < 100:
                if abs(lat_diff) >= abs(lon_diff):
                    x = [lat1, lat2]; y = [lon1, lon2]
                    f = interpolate.interp1d(x, y, kind='linear')
                    num = int(abs(lat_diff)) + 2 
                    x_interp = np.linspace(lat1, lat2, num)
                    for k in range(len(x_interp)-1):
                        y_interp = float(f(x_interp[k]))
                        subduction_temp.append((x_interp[k], y_interp))
                else:
                    x = [lon1, lon2]; y = [lat1, lat2]
                    f = interpolate.interp1d(x, y, kind='linear')
                    num = int(abs(lon_diff)) + 2 
                    x_interp = np.linspace(lon1, lon2, num)
                    for k in range(len(x_interp)-1):
                        y_interp = float(f(x_interp[k]))
                        subduction_temp.append((y_interp, x_interp[k],))  
                        
            else:
                subduction_temp.append(subduction[i][j])
                
        # add the last point at each subduciton zone
        subduction_temp.append(subduction[i][j+1])
        
        subduction_new.append(subduction_temp)            
    return subduction_new


# nearest interpolation for the value at subduction 
def interpolation(lat_grid, lon_grid, data_grid, subduction):
    f = interpolate.RegularGridInterpolator((lat_grid, lon_grid), data_grid, method='linear')
    increment = 0.2
    
    # linear interpolation
    data_subduction = []
    for i in range(len(subduction)):
        data_temp = f(subduction[i])
        # check the point out of the data range
        for j in range(len(data_temp)):
            if math.isnan(data_temp[j]):
                # search the nearest data point
                latitude = subduction[i][j][0]
                longitude =subduction[i][j][1]
                k = 1
                while math.isnan(data_temp[j]):
                            
                    # case 1: western border
                    longitude_new = longitude - k * increment
                    if longitude_new > -180.0:
                        temp = f((latitude, longitude_new))
                        if not math.isnan(temp):
                            data_temp[j] = temp
                            break
                        else:
                            for ii in range(k):
                                latitude_new = latitude + (ii+1) * increment
                                if latitude_new < 90.0:
                                    temp = f((latitude_new, longitude_new))
                                    if not math.isnan(temp):
                                        data_temp[j] = temp
                                        break
                                latitude_new = latitude - (ii+1) * increment
                                if latitude_new > -90.0:
                                    temp = f((latitude_new, longitude_new))
                                    if not math.isnan(temp):
                                        data_temp[j] = temp
                                        break
                            if not math.isnan(data_temp[j]):
                                break

                    # case 2: eastern border
                    longitude_new = longitude + k * increment
                    if longitude_new < 180.0:
                        temp = f((latitude, longitude_new))
                        if not math.isnan(temp):
                            data_temp[j] = temp
                            break
                        else:
                            for ii in range(k):
                                latitude_new = latitude + (ii+1) * increment
                                if latitude_new < 90.0:
                                    temp = f((latitude_new, longitude_new))
                                    if not math.isnan(temp):
                                        data_temp[j] = temp
                                        break
                                latitude_new = latitude - (ii+1) * increment
                                if latitude_new > -90.0:
                                    temp = f((latitude_new, longitude_new))
                                    if not math.isnan(temp):
                                        data_temp[j] = temp
                                        break
                            if not math.isnan(data_temp[j]):
                                break
                        
                    # case 3: northern border
                    latitude_new = latitude + k * increment
                    if latitude_new < 90.0:
                        temp = f((latitude_new, longitude))
                        if not math.isnan(temp):
                            data_temp[j] = temp
                            break
                        else:
                            for ii in range(k):
                                longitude_new = longitude + (ii+1) * increment
                                if longitude_new < 180.0:
                                    temp = f((latitude_new, longitude_new))
                                    if not math.isnan(temp):
                                        data_temp[j] = temp
                                        break
                                longitude_new = longitude - (ii+1) * increment
                                if longitude_new > -180.0:
                                    temp = f((latitude_new, longitude_new))
                                    if not math.isnan(temp):
                                        data_temp[j] = temp
                                        break
                            if not math.isnan(data_temp[j]):
                                break
                       
                    # case 4: southern point
                    latitude_new = latitude - k * increment   
                    if latitude_new > -90.0:
                        temp = f((latitude_new, longitude))
                        if not math.isnan(temp):
                            data_temp[j] = temp
                            continue
                        else:
                            for ii in range(k):
                                longitude_new = longitude + (ii+1) * increment
                                if longitude_new < 180.0:
                                    temp = f((latitude_new, longitude_new))
                                    if not math.isnan(temp):
                                        data_temp[j] = temp
                                        break
                                longitude_new = longitude - (ii+1) * increment
                                if longitude_new > -180.0:
                                    temp = f((latitude_new, longitude_new))
                                    if not math.isnan(temp):
                                        data_temp[j] = temp
                                        break
                            if not math.isnan(data_temp[j]):
                                break
                    k += 1
        data_subduction.append(data_temp)
    return data_subduction


def plate_thickness(subduction, agegrid_file, age):

    # seafloor age grid
    fname = agegrid_file.format(age)
    file = Dataset(fname)
    lon_grid = file.variables['lon'][:]
    lat_grid = file.variables['lat'][:]
    age_grid = file.variables['z'][:]
    # mantle temperature
    Tm = 1350.0
    # surface temperature
    T0 = 0.0
    # lithosphere isotherm 
    T1 = 1150.0
    # thermal diffusivity, unit: m2s-1
    kappa = 0.804e-6
    # cutoff for plate thickness
    cutoff1 = 10.0
    cutoff2 = 125.0
    
    # interpolate seafloor age at subduction zone, unit: Myr
    age_subduction = interpolation(lat_grid, lon_grid, age_grid, subduction)
    
    # calculate oceanic lithosphere thickness at subduction zone
    factor = special.erfinv((T1-T0)/(Tm-T0)) * 2 * math.sqrt(kappa)
    Myr2sec=1e6*365*24*60*60
    thickness_subduction = []
    for i in range(len(age_subduction)):
        thickness_subduction_each = []
        for j in range(len(age_subduction[i])):
            thickness = factor * math.sqrt(age_subduction[i][j]*Myr2sec) * 1e-3
            if thickness > cutoff2:
                thickness = cutoff2
            if thickness < cutoff1:
                thickness = cutoff1
            thickness_subduction_each.append(thickness)
        thickness_subduction.append(thickness_subduction_each)
    return age_subduction, thickness_subduction
    
    
def carbon_volume_density(subduction, grid_file, thickness_subduction, age):
    fname = grid_file.format(age)
    file = Dataset(fname)
    lon_grid = file.variables['x'][:]
    lat_grid = file.variables['y'][:]
    carbon_grid = file.variables['z'][:] # unit: Mt C/m^2
    carbon_grid = carbon_grid.filled(np.nan) 
    carbon_subduction = interpolation(lat_grid, lon_grid, carbon_grid, subduction)
    
    for i in range(len(carbon_subduction)):
        for j in range(len(carbon_subduction[i])):
            # convert Mt/m^2 to Mt/km^2
            carbon_subduction[i][j] *= 1e6
            # convert Mt/km^2 to Mt/km^3
            carbon_subduction[i][j] /= thickness_subduction[i][j] 
    return carbon_subduction
    

def carbon_volume_desity_lithosphere(subduction, grid_file, thickness_subduction, age):
    fname = grid_file.format(age)
    file = Dataset(fname)
    lon_grid = file.variables['x'][:]
    lat_grid = file.variables['y'][:]
    carbon_grid = file.variables['z'][:]
    carbon_grid = carbon_grid.filled(np.nan)
    # delete extremely high values
    for i in range(len(carbon_grid)):
        for j in range(len(carbon_grid[i])):
            if carbon_grid[i][j] > 1e-4:
                carbon_grid[i][j] = np.nan

    carbon_subduction = interpolation(lat_grid, lon_grid, carbon_grid, subduction)
    
    for i in range(len(carbon_subduction)):
        for j in range(len(carbon_subduction[i])):
            # convert Mt/m^2 to Mt/km^2
            carbon_subduction[i][j] *= 1e6
            # convert Mt/km^2 to Mt/km^3
            carbon_subduction[i][j] /= thickness_subduction[i][j] 
    return carbon_subduction


def save_to_txt(fname, subduction, age, thickness, lithosphere, serpentinite, crust, sediment, total):
    with open(fname, 'w') as file:
        # write header
        string = 'Latitude'  + ' ' * 4 # length: 12
        file.write(string)
        string = 'Longitude'  + ' ' * 4 # length: 13
        file.write(string)
        string = 'Seafloor_Age(Myr)'  + ' ' * 4 # length: 21
        file.write(string)
        string = 'Plate_Thickness(km)'  + ' ' * 4 # length: 23
        file.write(string)
        string = 'lithosphere_carbon(Mt/km^3)'  + ' ' * 4 # length: 31
        file.write(string)
        string = 'Serpentinite_carbon(Mt/km^3)'  + ' ' * 4 # length: 32
        file.write(string)
        string = 'Crust_carbon(Mt/km^3)'  + ' ' * 4 # length: 25
        file.write(string)
        string = 'Sediment_carbon(Mt/km^3)'  + ' ' * 4 # length: 28
        file.write(string)
        string = 'Total_carbon(Mt/km^3)'  + ' ' * 4 # length: 25
        file.write(string + '\n')
        
        # write data
        for i in range(len(subduction)):
            header = '='*20 + 'Subduction_Zone_{}'.format(i) + '='*20
            file.write(header + '\n')
            for j in range(len(subduction[i])):
                # mark the anomaly value at some points
                # especially in the mediterranean region, there are several points 
                # with extremly young seafloor age but very high sediment carbon
                if total[i][j] > 10:
                    file.write('*')

                string = '%.2f' % subduction[i][j][0]
                file.write(string + ' '*(12 - len(string)))

                string = '%.2f' % subduction[i][j][1]
                file.write(string + ' '*(13 - len(string)))
                
                string = '%.2f' % age[i][j]
                file.write(string + ' '*(21 - len(string)))
                
                string = '%.2f' % thickness[i][j]
                file.write(string + ' '*(23 - len(string)))
                
                string = '%f' % lithosphere[i][j]
                file.write(string + ' '*(31 - len(string)))
                
                string = '%f' % serpentinite[i][j]
                file.write(string + ' '*(32 - len(string)))
                
                string = '%f' % crust[i][j]
                file.write(string + ' '*(25 - len(string)))
                
                string = '%f' % sediment[i][j]
                file.write(string + ' '*(28 - len(string)))
                
                string = '%f' % total[i][j]
                file.write(string + ' '*(25 - len(string)))
                file.write('\n')
    
def calculate_carbon_subduction(age, subduction, output_path):
    print('Working at %s Ma' % age)

    # step2: calculate plate thickness at subduction zone
    agegrid_file = 'Muller_etal_2019_Tectonics_v2.0_netCDF/Muller_etal_2019_Tectonics_v2.0_AgeGrid-{}.nc'# seafloor Agegrid files
    age_subduction, thickness_subduction = plate_thickness(subduction, agegrid_file, age)

    # step3: calculate carbonate volume density at subduction zone

    # carbon in the lithosphere
    lithosphere_file = 'Data_carbon_Muller2022/Lithosphere/mean/carbon_lithosphere_grid_{}.nc'
    lithosphere_carbon_subduction = carbon_volume_desity_lithosphere(
        subduction, lithosphere_file, thickness_subduction, age
    )
    # carbon in the serpentinite
    serpentinite_file = 'Data_carbon_Muller2022/Serpentinite/mean/carbon_serpentinite_grid_{}.nc'
    serpentinite_carbon_subduction = carbon_volume_density(
        subduction, serpentinite_file, thickness_subduction, age
    )

    # carbon in the crust
    crust_file = 'Data_carbon_Muller2022/Crust/mean/carbon_crust_grid_{}.nc'
    crust_carbon_subduction = carbon_volume_density(
        subduction, crust_file, thickness_subduction, age
    )

    # carbon in the sediment
    sediment_file = 'Data_carbon_Muller2022/Sediment/mean/carbon_sediment_grid_{}.nc'
    sediment_carbon_subduction = carbon_volume_density(
        subduction, sediment_file, thickness_subduction, age
    )
    # total carbon
    total_carbon_subduction = []
    for i in range(len(subduction)):
        total_carbon_subduction_each = lithosphere_carbon_subduction[i] +\
                                       serpentinite_carbon_subduction[i] +\
                                       crust_carbon_subduction[i] +\
                                       sediment_carbon_subduction[i]
        total_carbon_subduction.append(total_carbon_subduction_each)

    # step4: save to file
    output_file = output_path + 'carbon_volume_density_{}.txt'.format(age)
    save_to_txt(
        output_file, subduction, age_subduction, thickness_subduction,
        lithosphere_carbon_subduction, serpentinite_carbon_subduction,
        crust_carbon_subduction, sediment_carbon_subduction, total_carbon_subduction
    )
    print('%s Ma has finished.'%age)


import os
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def test_plot(z):
    fig, ax = plt.subplots()
    im = ax.imshow(z, origin='lower', cmap=cm.cividis, extent=[-180, 180, -90, 90], )
    fig.colorbar(im, ax=ax)
    
    plt.show()


if __name__=='__main__':
    # load plate motion model
    use_local_files = True
    # download plate reconstruction data
    if not use_local_files:
        gdownload = gplately.download.DataServer("Muller2019")
        rotation_model, topology_features, static_polygons = gdownload.get_plate_reconstruction_files()
    #loading local files
    if use_local_files:
        input_directory = "./Muller_etal_2019_PlateMotionModel_v2.0_Tectonics_Updated/"
        
        # Locate rotation files and set up the RotationModel object
        rotation_filenames = glob.glob(os.path.join(input_directory, '*.rot'))
        rotation_model = pygplates.RotationModel(rotation_filenames)
        
        # Locate topology feature files and set up a FeatureCollection object 
        topology_filenames = glob.glob(os.path.join(input_directory, '*.gpml'))
        topology_features = pygplates.FeatureCollection()
        for topology_filename in topology_filenames:
            # (omit files with the string "inactive" in the filepath)
            if "Inactive" not in topology_filename:
                topology_features.add( pygplates.FeatureCollection(topology_filename) )
            else:
                topology_filenames.remove(topology_filename)
    
    
    output_path = 'Carbon_VolumeDensity_SubductionZone/mean/'
    mkdir(output_path)


    # calculate the carbon at subduction zone 
    Cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(processes=Cores)
    reconstruction_age = np.arange(0, 100)
    for age in reconstruction_age:
        # step1: extract the location of past subduction zone from plate motion model
        subduction = SubductionZone(rotation_model, topology_features, age)
        p.apply_async(calculate_carbon_subduction, args=(age, subduction, output_path))
    p.close()
    p.join()


    

