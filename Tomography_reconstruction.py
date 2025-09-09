# -*- coding: utf-8 -*-
"""
Created on Feb 28 17:33:21 2024

Reconstruct the global tomography model 

@author: m1335
"""

from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
import getRate
import math
import os
import glob
import multiprocessing
R = 6371


# optional model: TX2019slab, UU-P07, LLNL_G3D_JPS, MITP08，DETOX-P3, GLAD_M25
Tomo_model = 'TX2019slab'
# reconstruction parameters
# The maximum distance between points in the tomography model and subduction zone above 410km
Dmax = 200
#processing depth of the residual tomography model
Dep_pro = 410
output_path = 'Reconstructed_TomographyModel/{}_Dmax{}_newrate/'.format(Tomo_model, Dmax)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def load_TX2019slab(fname):
    file = Dataset(fname)
    Depth = file.variables['depth'][:]#22 * 1
    dV = file.variables['dvp'][:]#22*181*361
    return Depth, dV


def load_UUP07(fname):
    data = []
    f = open(fname,'r')
    temp = []
    depth = 5.0
    for each_line in f:
        temp1 = each_line.strip('\n')
        temp1 = temp1.split()
        if temp1 != []:
            for i in range(4):
                temp1[i] = float(temp1[i])
            # if in the same depth 
            if temp1[2] == depth:
                temp.append(temp1)
            else:
                data.append(temp)
                temp = []
                depth = temp1[2]
                temp.append(temp1)
    data.append(temp)
    f.close()
    #resampling data to k*180*360
    dV = []
    Depth = []
    for i in range(len(data)):
        dV_temp1 = []
        dV_temp2 = []
        m = 0
        n = 0
        l = 0
        for j in range(len(data[i])):
            n += 1
            if l != 180:
                if n > 720 and n != 1440: continue
                if n == 1440:
                    n = 0
                    continue
            if j % 2 == 0:
                dV_temp1.append(data[i][j][3])
                m += 1
                if m == 360:
                    l += 1
                    m = 0
                    dV_temp1.append(data[i][j+1][3])
                    dV_temp2.append(dV_temp1)
                    dV_temp1 = []
        dV_temp2.reverse()
        dV.append(dV_temp2)
        Depth.append(data[i][j][2])
    Depth = np.array(Depth)
    dV = np.array(dV)
    return Depth, dV # depth:28 ; dv:28*181*361


def load_LLNL_G3D_JPS(fname):
    depth = []
    dv = []
    temp11 = []
    for num in range(17, 59 + 1):
        if num > 17:
            depth_mean_last = temp11.mean()
        file = fname.format(num)
        f = open(file, 'r')
        temp1 = [] # depth
        temp11 = []
        temp2 = []  # dvp
        temp22 = []
        m = 0
        for each_line in f:
            temp = each_line.strip('\n')
            temp = temp.split()
            if temp != []:
                temp1.append(float(temp[1])) # depth
                temp2.append(float(temp[3])) # dVp
                m = m + 1
                if m == 361:
                    m = 0
                    temp11.append(temp1)
                    temp22.append(temp2)
                    temp1 = []
                    temp2 = []
                    
        temp11 = np.array(temp11)
        depth_mean = temp11.mean()
        # the case in 410 km and 660 km velocity discontinuity layers
        # there will be two layers with same depth, which corresponds to the upper
        # and the lower velocity layers in the velocity discontinuity layers 
        if num > 17:
            if int(depth_mean) <= int(depth_mean_last): 
                depth_mean = depth_mean + 1
        depth.append(depth_mean)
        dv.append(temp22)
        f.close()
    depth = np.array(depth)
    dv = np.array(dv)
    return depth, dv # depth: k; dv:k*181*361


def load_LLNL_G3Dv3(directory):
    depth = []
    dv = []
    fmin = 15
    fmax = 57
    temp11 = []
    for num in range(fmin, fmax + 1):
        if num > 17:
            depth_mean_last = temp11.mean()
        
        fname = 'LLNL_G3Dv3.Interpolated.Layer{}.txt'.format(num)
        f = open(directory+fname, 'r')
        temp1 = []
        temp11 = []
        temp2 = []
        temp22 = []
        m = 0
        for each_line in f:
            temp = each_line.strip('\n')
            temp = temp.split()
            if temp != []:
                temp1.append(6371 - float(temp[0]))#depth
                temp2.append(float(temp[2]))#dVp
                m = m + 1
                if m == 361:
                    m = 0
                    temp11.append(temp1)
                    temp22.append(temp2)
                    temp1 = []
                    temp2 = []
                    
        temp11 = np.array(temp11)
        depth_mean = temp11.mean()
        if num > 17:
            if int(depth_mean) <= int(depth_mean_last): 
                depth_mean = depth_mean + 1
        depth.append(depth_mean)
        dv.append(temp22)
        f.close()
        
    depth = np.array(depth)
    dv = np.array(dv)
    
    return depth, dv # k*181*361


def load_GYPSUM(fname):
    file = Dataset(fname)
    depth = file.variables['depth'][:]
    dv = file.variables['dvp'][:]     
    depth = np.array(depth)
    dv = np.array(dv)        
    return depth, dv # depth: 25*1; dv:k*181*361


def load_MITP08(fname):
    f = open(fname, 'r')
    depth = [22.6,]
    dv_origin = []
    dv_layer = []
    dv_temp = []
    lat_origin = []
    lon_origin = []
    
    f.readline()
    for each_line in f:
        each_line = each_line.strip('\n')
        each_line = each_line.split()
        
        # read original coordinates of MITP08 model
        if depth[-1] == 22.6:
            if float(each_line[0]) not in lat_origin:
                lat_origin.append(float(each_line[0]))
            if float(each_line[1]) not in lon_origin:
                lon_origin.append(float(each_line[1]))
        
        # read data
        if float(each_line[2]) != depth[-1]:
            depth.append(float(each_line[2]))
            dv_origin.append(dv_layer)
            dv_layer = []
        
        dv_temp.append(float(each_line[3]))
        if float(each_line[0]) == 89.65:
            dv_layer.append(dv_temp)
            dv_temp = []
    dv_origin.append(dv_layer)
    
    # interpolate to the node grid of 1*1
    dv_interp = []
    x, y = lon_origin, lat_origin
    x[0], x[-1] = 0, 360.0
    y[0], y[-1] = -90.0, 90.0
    
    x_interp = np.concatenate([np.arange(181,361,1), np.arange(0,181,1)])
    y_interp = np.arange(-90, 91, 1)
    X, Y = np.meshgrid(x_interp, y_interp, indexing='xy')
    
    
    for i in range(len(depth)):
        interp = RegularGridInterpolator((x,y), dv_origin[i])
        dv_layer = interp((X, Y))
        dv_interp.append(dv_layer)
    
    depth = np.array(depth)
    dv_interp = np.array(dv_interp)
    
    return depth, dv_interp


def load_DETOXP3(directory):
    # this model is downloaded from SubMachine website
    # get the depth of each tomography slice
    pattern = directory + '*.txt'
    files = glob.glob(pattern)
    Depth = []
    for file in files:
        file = file.split('_')
        depth = int(file[4].split('.')[0])
        Depth.append(depth)
    Depth = sorted(Depth)

    # define the coordinate of the node grid
    lat_origin = np.arange(-90, 90.5, 0.5)
    lon_origin = np.arange(0, 360.5, 0.5)
    m, n = len(lat_origin), len(lon_origin)
    
    # read the original tomography model
    dv_origin = []
    for i in range(len(Depth)):
        fname = 'SubMachine_depth_slice_{}.txt'.format(Depth[i])
        dv_layer = []
        with open(directory+fname, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            for each_line in f:
                each_line = each_line.strip('\n').split()
                dv_layer.append(float(each_line[2]))
        dv_layer = np.array(dv_layer)
        dv_layer = dv_layer.reshape(m,n)
        dv_origin.append(dv_layer)
        
    # interpolate to the node grid of 1*1
    dv_interp = []
    x, y = lat_origin, lon_origin

    x_interp = np.arange(-90, 91, 1)
    y_interp = np.concatenate([np.arange(181, 361, 1), np.arange(0, 181, 1)])
    X, Y = np.meshgrid(x_interp, y_interp, indexing='ij')

    for i in range(len(Depth)):
        interp = RegularGridInterpolator((x,y), dv_origin[i])
        dv_layer = interp((X, Y))
        dv_interp.append(dv_layer)
    
    Depth = np.array(Depth)
    dv_interp = np.array(dv_interp)
    
    return Depth, dv_interp


def load_GLAD_M25(fname):
    file = Dataset(fname)
    Depth = file.variables['depth'][30:] # 342 * 1
    latitude = file.variables['latitude'][:]
    longitude = file.variables['longitude'][:]
    vpv = file.variables['vpv'][30:]
    vph = file.variables['vph'][30:]

    # # calculate isotropic vp (Tao et al., 2018)
    vpv2 = vpv**2
    vph2 = vph**2
    vp2 = (vpv2 + 4*vph2) / 5
    vp = vp2**(1/2)

    # calculate dlnvp 
    dvp_origin = []
    for i in range(len(vp)):
        layer_mean = vp[i].mean()
        dvp_origin.append((vp[i] - layer_mean) / layer_mean)
    # convert to %
    dvp_origin = np.array(dvp_origin)
    dvp_origin = dvp_origin * 100

    # interpolate to 1*1 grid
    dv_interp = []
    x, y = latitude, longitude
    x_interp = np.arange(-90, 91, 1)
    y_interp = np.arange(-180, 181, 1)
    X, Y = np.meshgrid(x_interp, y_interp, indexing='ij')

    for i in range(len(Depth)):
        interp = RegularGridInterpolator((x,y), dvp_origin[i])
        dv_layer = interp((X, Y))
        dv_interp.append(dv_layer)

    Depth = np.array(Depth)
    dv_interp = np.array(dv_interp)
    
    return Depth, dv_interp


def read_SubductionZone_coordinate(age):
    SubductionZone = []
    fname = 'Carbon_VolumeDensity_SubductionZone/mean/carbon_volume_density_{}.txt'.format(age)
    with open(fname, 'r') as file:
        file.readline()
        for each_line in file.readlines():
            if each_line.startswith('='):
                continue
            if each_line.startswith('*'):
                each_line = each_line[1:]
            each_line = each_line.strip()
            each_line = each_line.split()
            temp = (float(each_line[0]), float(each_line[1]))
            SubductionZone.append(temp)

    # delete repeated points
    SubductionZone = set(SubductionZone)
    SubductionZone = list(SubductionZone)
    return SubductionZone
    

# calculating time-depth correlation
def time_depth(age):

    upper_rate = getRate.upper_mantle() # dict shape: age_length * 1
    lower_rate = getRate.lower_mantle() # shape: 181*361(-90~90, -180~180)

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
    return depth, depth_mean


def haversine_distance(depth, lat1, lon1, lat2, lon2):
    delta_lat = lat1 - lat2
    delta_lon = lon1 - lon2
    
    a = math.sin(math.radians(delta_lat/2))**2 +\
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *\
        math.sin(math.radians(delta_lon/2))**2
    d = 2 * (R-depth) * math.asin(math.sqrt(a))
    
    return d


def search_nearest_SubductionZone(depth, latitude, longitude, data): 
    dis_min = haversine_distance(depth, latitude, longitude, data[0][0], data[0][1])
    data_min = data[0]
    for i in range(len(data)):
        distance = haversine_distance(depth, latitude, longitude, data[i][0], data[i][1])
        if distance <= dis_min:
            dis_min = distance 
            data_min = data[i]
    if dis_min < Dmax:
        flag = 1
    else:
        flag = 0
    return data_min, flag


# interpolate velocity anomaly at the given depth
def interpolation(depth, dV, interpdep, SubductionZone):
    m, n = interpdep.shape
    # store the interpolated velocity anomaly values
    value = np.zeros((m, n))
    original_value = np.zeros((m, n))
    for i in range(m):# latitude -90~90°
        for j in range(n):# Lontitude -180~180°
            y = [] # store dv of each point
            for k in range(len(depth)):
                y.append(dV[k][i][j])
            max_depth = depth.max()
            min_depth = depth.min()
            y = np.array(y)
            f = interp1d(depth, y, kind = 'slinear')
            if interpdep[i][j] >= min_depth and interpdep[i][j] <= max_depth:
                value[i][j] = f(interpdep[i][j])
                original_value[i][j] = f(interpdep[i][j])
            #else:
                #print(f'mindepth={min_depth}; maxdepth={max_depth}; interp_depth = {interpdep[i][j]}')

            # get residual tomography model based on subduction zone
            if interpdep[i][j] < Dep_pro and value[i][j] > 0:
                latitude = i - 90
                longitude = j - 180
                dis_min, flag = search_nearest_SubductionZone(
                    interpdep[i][j], latitude, longitude, SubductionZone
                    )
                if flag == 0:
                    value[i][j] = 0

            
    # calculate mean positive veolocity (MPV)
    MPV = original_value[original_value>0].mean()

    return value, MPV # shape: 181 * 361(-90~90, -180~180)


def reconstruction(age, Depth, dV):
    print('The %d Ma begin!' % age)
    Interpdep, Depth_mean = time_depth(age)
    print(f'Depth_mean at {age} Ma is {Depth_mean} km.')
    # read subduction zone coordinate extracted from plate motion model
    SubductionZone = read_SubductionZone_coordinate(age)

    # reconstruction 
    each_dV, MPV = interpolation(Depth, dV, Interpdep, SubductionZone)
    # save reconstructed model to file
    fname = output_path + '{}_{}.nc'.format(Tomo_model, age)
    lon_grid = np.arange(-180, 181)
    lat_grid = np.arange(-90, 91)
    with Dataset(fname, 'w') as file:
        file.createDimension('lon', lon_grid.size)
        file.createDimension('lat', lat_grid.size)
        file.createDimension('scalar', 1)
        longitude = file.createVariable('lon', lon_grid.dtype, ('lon',), zlib=True)
        latitude = file.createVariable('lat', lat_grid.dtype, ('lat',), zlib=True)
        longitude[:] = lon_grid
        latitude[:] = lat_grid
        longitude.units = 'degrees'
        latitude.units = 'degrees'
        
        data = file.createVariable('z', each_dV.dtype, ('lat', 'lon'), zlib=True)
        data[:,:] = each_dV
        mean_positive_velocity = file.createVariable('MPV', np.float64, ('scalar',))
        mean_positive_velocity[:] = MPV

    print('The %d Ma completed!' % age)
    


if __name__ == '__main__':
    
    # read global tomography model
    if Tomo_model == 'TX2019slab':
        fname = 'Original_TomographyModel/TX2019slab_percent.nc'
        Depth, dV = load_TX2019slab(fname)
    elif Tomo_model == 'UU-P07':
        fname = 'Original_TomographyModel/UU-P07_lon_lat_depth_%dVp_cell_depth_midpoint.txt'
        Depth, dV = load_UUP07(fname)
    elif Tomo_model == 'LLNL_G3D_JPS':
        fname = 'Original_TomographyModel/LLNL_G3D_JPS/LLNL_G3D_JPS.Interpolated.{}.txt'
        Depth, dV = load_LLNL_G3D_JPS(fname)
    elif Tomo_model == 'LLNL_G3Dv3':
        directory = 'Original_TomographyModel/LLNL_G3Dv3/LLNL_G3Dv3_interpolated/'
        Depth, dV = load_LLNL_G3Dv3(directory)
    elif Tomo_model == 'GYPSUM':
        fname = 'Original_TomographyModel/GYPSUM_percent.nc'
        Depth, dV = load_GYPSUM(fname)
    elif Tomo_model == 'MITP08':
        fname = 'Original_TomographyModel/MITP08.txt'
        Depth, dV = load_MITP08(fname)
    elif Tomo_model == 'DETOX-P3':
        directory = 'Original_TomographyModel/DETOX-P3/'
        Depth, dV = load_DETOXP3(directory)
    elif Tomo_model == 'GLAD_M25':
        fname = 'Original_TomographyModel/GLAD_M25/glad-m25-vp-0.0-n4.nc'
        Depth, dV = load_GLAD_M25(fname)


    # reconstruction
    mkdir(output_path)
    Cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(processes=Cores)
    Age = np.arange(1, 101)
    for i in range(len(Age)):
        p.apply_async(reconstruction, args=(Age[i], Depth, dV))
    p.close()
    p.join()
    


