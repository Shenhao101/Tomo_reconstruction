# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:05:54 2024

@author: shenhao (adopted from Muller 2022)
"""
import os 
import glob
import warnings
import ptt
import pygplates
import numpy as np
import pandas as pd
from pygplates_helper import *


# common data files
agegrid_filename = "../Muller_etal_2019_Tectonics_v2.0_netCDF/Muller_etal_2019_Tectonics_v2.0_AgeGrid-{:.0f}.nc"

# common variables
extent_globe = [-180, 180, -90, 90]
earth_radius = 6371.0e3
tessellation_threshold_radians = np.radians(0.01)

# output grid resolution - should be identical to input grid resolution!
spacingX, spacingY = 0.2, 0.2
lon_grid = np.arange(extent_globe[0], extent_globe[1]+spacingX, spacingX)
lat_grid = np.arange(extent_globe[2], extent_globe[3]+spacingY, spacingY)
lonq, latq = np.meshgrid(lon_grid,lat_grid)

# 0-250Ma from Muller 2022
input_directory = "../Muller_etal_2019_PlateMotionModel_v2.0_Tectonics_Updated/"


rotation_filenames = glob.glob(os.path.join(input_directory, '*.rot'))
rotation_model = pygplates.RotationModel(rotation_filenames)


topology_filenames = glob.glob(os.path.join(input_directory, '*.gpml'))
topology_features = pygplates.FeatureCollection()
for topology_filename in topology_filenames:
    if "Inactive" not in topology_filename:
        topology_features.add( pygplates.FeatureCollection(topology_filename) )


# We define a custom interpolator that optionally returns indices and distances
# The grid is unchanged throughout time, so we define this once and leave it alone.
interpolator = RegularGridInterpolator((lat_grid, lon_grid), np.empty((lat_grid.size, lon_grid.size)), 'nearest')

def sample_grid(lon, lat, grid, extent=[-180,180,-90,90], return_indices=False, return_distances=False):
    interpolator.values = grid
    return interpolator(np.c_[lat, lon], return_indices=return_indices, return_distances=return_distances)

def My2s(Ma):
    return Ma*3.1536e13

def plate_temp(age, z, PLATE_THICKNESS) :
    "Computes the temperature in a cooling plate for age = t\
    and at a depth = z."

    KAPPA = 0.804e-6
    T_MANTLE = 1350.0
    T_SURFACE = 0.0

    sine_arg = np.pi * z / PLATE_THICKNESS
    exp_arg = -KAPPA * np.pi * np.pi * age / (PLATE_THICKNESS * PLATE_THICKNESS)
    k = np.ones_like(age)*np.arange(1, 20).reshape(-1,1)
    cumsum = ( np.sin(k * sine_arg) * np.exp(k*k*exp_arg)/k ).sum(axis=0)

    return T_SURFACE + 2.0 * cumsum * (T_MANTLE - T_SURFACE)/np.pi + (T_MANTLE - T_SURFACE) * z/PLATE_THICKNESS

def plate_isotherm_depth(age, temp=1250.0):
    "Computes the depth to the temp - isotherm in a cooling plate mode.\
    Solution by iteration. By default the plate thickness is 125 km as\
    in Parsons/Sclater."

    PLATE_THICKNESS = 125e3
    
    z = 0.0 # starting depth is 0
    rtol = 0.001 # error tolerance
    
    z_too_small = np.atleast_1d(np.zeros_like(age, dtype=float))
    z_too_big = np.atleast_1d(np.full_like(age, PLATE_THICKNESS, dtype=float))
    
    for i in range(20):
        zi = 0.5 * (z_too_small + z_too_big)
        ti = plate_temp (age, zi, PLATE_THICKNESS)
        t_diff = temp - ti
        z_too_big[t_diff < -rtol] = zi[t_diff < -rtol]
        z_too_small[t_diff > rtol] = zi[t_diff > rtol]
        
        if (np.abs(t_diff) < rtol).all():
            break
            
    # protect against negative ages
    zi[age <= 0] = 0
    return np.squeeze(zi)


"""

Calculate some tectonic parameters for the plate reconstruction model,
reference from Muller 2022

"""
def plate_tectonic_stats(reconstruction_time):
    
        
    # calculate subduction convergence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        subduction_data = ptt.subduction_convergence.subduction_convergence(
            rotation_model,
            topology_features,
            tessellation_threshold_radians,
            reconstruction_time,
            anchor_plate_id=0)
        subduction_data = np.vstack(subduction_data)
        subduction_lon  = subduction_data[:,0]
        subduction_lat  = subduction_data[:,1]
        subduction_len  = np.radians(subduction_data[:,6]) * 1e3 * pygplates.Earth.mean_radius_in_kms
    
        
    # protect against "negative" subduction
    subduction_data[:,2] = np.clip(subduction_data[:,2], 0.0, 1e99)
    # horizental component (normal to trench) of the subduction velocity (relative to trench in cm/yr) 
    subduction_vel = np.fabs(subduction_data[:,2])*1e-2 * np.cos(np.radians(subduction_data[:,3])) 
    subd_vel_mean, subd_vel_std = np.mean(subduction_vel), np.std(subduction_vel)
    
    # sample age grid
    age_grid = read_netcdf_grid(agegrid_filename.format(reconstruction_time), resample=(spacingX,spacingY))
    age_grid_filled = fill_ndimage(age_grid)
    age_interp = sample_grid(subduction_lon, subduction_lat, age_grid_filled)
    thickness = plate_isotherm_depth(My2s(age_interp))
    
    # how can get this relationship?
    slab_dip = 1.88e-3*subduction_vel*thickness + 21.9
    slab_dip_mean, slab_dip_std = np.median(slab_dip), np.std(slab_dip)

    # calculate vertical subduction velocity
    vertical_vel = subduction_vel * np.sin(np.radians(slab_dip))
    vert_vel_mean, vert_vel_std = np.mean(vertical_vel), np.std(vertical_vel)

    del subduction_data, age_grid, age_grid_filled # clean up

    return(subd_vel_mean, subd_vel_std, slab_dip_mean, slab_dip_std, vert_vel_mean, vert_vel_std)


if __name__ == '__main__':
    Ages = np.arange(251)
    stats_data = {}
    stats_data['Age'] = []
    stats_data['Subduction rate mean (cm/yr)'] = []
    stats_data['Shallow slab dip mean (deg)'] = []
    stats_data['Vertical subduction rate mean (cm/yr)'] = []
    for age in Ages:
        print(f'working at {age} Ma...')
        stats_Ma = plate_tectonic_stats(age)
        stats_data['Age'].append(age)
        stats_data['Subduction rate mean (cm/yr)'].append(stats_Ma[0]*1e2)
        stats_data['Shallow slab dip mean (deg)'].append(stats_Ma[2])
        stats_data['Vertical subduction rate mean (cm/yr)'].append(stats_Ma[4]*1e2)

    df = pd.DataFrame(stats_data)
    df.to_excel('shallow_vertical_rate.xlsx', index=False, engine='openpyxl')
     
    
    
    
    