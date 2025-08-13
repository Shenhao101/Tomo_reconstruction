import numpy as np
import pygplates
from scipy.interpolate import RegularGridInterpolator as RGI


class RegularGridInterpolator(RGI):

    def __init__(self, points, values, method="linear", bounds_error=False, fill_value=np.nan):

        super(RegularGridInterpolator, self).__init__(points, values, method, bounds_error, fill_value)

    def __call__(self, xi, method=None, return_indices=False, return_distances=False):
        """
        Interpolation at coordinates
        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".
        """
        from scipy.interpolate.interpnd import _ndim_coords_from_arrays
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        indices, norm_distances = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices,
                                           norm_distances)
        elif method == "nearest":
            result = self._evaluate_nearest(indices,
                                            norm_distances)
        # if not self.bounds_error and self.fill_value is not None:
        #     result[out_of_bounds] = self.fill_value
            
        interp_output = result.reshape(xi_shape[:-1] + self.values.shape[ndim:])
        output_tuple = [interp_output]

        if return_indices:
            output_tuple.append(indices)
        if return_distances:
            output_tuple.append(norm_distances)
        
        if return_distances or return_indices:
            return tuple(output_tuple)
        else:
            return output_tuple[0]

def update_progress(progress):
    from IPython.display import clear_output
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    
    bar_length = 20
    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)
        
def fill_ndimage(data,invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell
    """
    from scipy.ndimage import distance_transform_edt
    masked_array = hasattr(data, "fill_value")
    if masked_array:
        mask_fill_value = data.data == data.fill_value
        data = data.data.copy()
        data[mask_fill_value] = np.nan
    else:
        data = data.copy()

    if invalid is None:
        invalid = np.isnan(data)
        if masked_array:
            invalid += mask_fill_value
    ind = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def read_netcdf_grid(filename, return_grids=False, resample=None):
    """
    Read in a netCDF file and re-align from -180 to 180 degrees
    
    Parameters
    ----------
    filename : str
        path to netCDF file
    return_grids : bool
        optionally return lon, lat arrays associated with grid
    resample : tuple
        optionally resample grid, pass spacing in X and Y direction as a tuple
        e.g. resample=(spacingX, spacingY)
    """
    import netCDF4
    
    # open netCDF file and re-align from -180, 180 degrees
    with netCDF4.Dataset(filename, 'r') as cdf:
        cdf_grid = cdf["z"]
        try:
            cdf_lon = cdf['lon'][:]
            cdf_lat = cdf['lat'][:]
        except:
            cdf_lon = cdf['x'][:]
            cdf_lat = cdf['y'][:]
            
        cdf_lon_mask = cdf_lon[:] > 180
        
        if cdf_lon_mask.any():
            cdf_grid_z = np.hstack([cdf_grid[:,cdf_lon_mask], cdf_grid[:,~cdf_lon_mask]])
            cdf_lon = np.hstack([cdf_lon[cdf_lon_mask], cdf_lon[~cdf_lon_mask]])
        else:
            cdf_grid_z = cdf_grid[:]

    # resample
    if resample is not None:
        spacingX, spacingY = resample
        lon_grid = np.arange(cdf_lon.min(), cdf_lon.max()+spacingX, spacingX)
        lat_grid = np.arange(cdf_lat.min(), cdf_lat.max()+spacingY, spacingY)
        lonq, latq = np.meshgrid(lon_grid, lat_grid)
        interp = RegularGridInterpolator((cdf_lat, cdf_lon), cdf_grid_z, method='nearest', bounds_error=False)
        cdf_grid_z = interp((latq, lonq))
        cdf_lon = lon_grid
        cdf_lat = lat_grid
            
    if return_grids:
        return cdf_grid_z, cdf_lon, cdf_lat
    else:
        return cdf_grid_z
    
def write_netcdf_grid(filename, grid, extent=[-180,180,-90,90]):
    import netCDF4
    
    nrows, ncols = np.shape(grid)
    
    lon_grid = np.linspace(extent[0], extent[1], ncols)
    lat_grid = np.linspace(extent[2], extent[3], nrows)
    
    with netCDF4.Dataset(filename, 'w') as cdf:
        cdf.createDimension('x', lon_grid.size)
        cdf.createDimension('y', lat_grid.size)
        cdf_lon = cdf.createVariable('x', lon_grid.dtype, ('x',), zlib=True)
        cdf_lat = cdf.createVariable('y', lat_grid.dtype, ('y',), zlib=True)
        cdf_lon[:] = lon_grid
        cdf_lat[:] = lat_grid
        cdf_lon.units = "degrees"
        cdf_lat.units = "degrees"

        cdf_data = cdf.createVariable('z', grid.dtype, ('y','x'), zlib=True)
        cdf_data[:,:] = grid
        

def lonlat2xyz(lon, lat):
    """
    Convert lon / lat (radians) for the spherical triangulation into x,y,z
    on the unit sphere
    """
    cosphi = np.cos(lat)
    xs = cosphi*np.cos(lon)
    ys = cosphi*np.sin(lon)
    zs = np.sin(lat)
    return xs, ys, zs

def xyz2lonlat(x,y,z):
    """
    Convert x,y,z representation of points *on the unit sphere* of the
    spherical triangulation to lon / lat (radians).

    Notes:
        no check is made here that (x,y,z) are unit vectors
    """
    lons = np.arctan2(ys, xs)
    lats = np.arcsin(zs)
    return lons, lats

def haversine_distance(lon1, lon2, lat1, lat2):
    """
    from  https://en.wikipedia.org/wiki/Haversine_formula
    https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
    """
    R = 6378.137 # radius of earth in km
    dLat = lat2*np.pi/180 - lat1*np.pi/180
    dLon = lon2*np.pi/180 - lon1*np.pi/180
    a = np.sin(dLat/2)**2 + np.cos(lat1*np.pi/180)*np.cos(lat2*np.pi/180) * np.sin(dLon/2)**2
    c = 2.0*np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    d = R*c
    return d*1000


def get_point_velocities(lons, lats, topology_features, rotation_model, reconstruction_time, delta_time=1.0):
    """ function to make a velocity mesh nodes at an arbitrary set of points defined in Lat
    Lon and Lat are assumed to be 1d arrays. """
    # Add points to a multipoint geometry

    reconstruction_time = float(reconstruction_time)

    multi_point = pygplates.MultiPointOnSphere([(float(lat),float(lon)) for lat, lon in zip(lats,lons)])

    # Create a feature containing the multipoint feature, and defined as MeshNode type
    meshnode_feature = pygplates.Feature(pygplates.FeatureType.create_from_qualified_string('gpml:MeshNode'))
    meshnode_feature.set_geometry(multi_point)
    meshnode_feature.set_name('Velocity Mesh Nodes from pygplates')

    velocity_domain_features = pygplates.FeatureCollection(meshnode_feature)
    
    # NB: at this point, the feature could be written to a file using
    # output_feature_collection.write('myfilename.gpmlz')
    
    
    # All domain points and associated (magnitude, azimuth, inclination) velocities for the current time.
    all_domain_points = []
    all_velocities = []

    # Partition our velocity domain features into our topological plate polygons at the current 'time'.
    plate_partitioner = pygplates.PlatePartitioner(topology_features, rotation_model, reconstruction_time)

    for velocity_domain_feature in velocity_domain_features:
        # A velocity domain feature usually has a single geometry but we'll assume it can be any number.
        # Iterate over them all.
        for velocity_domain_geometry in velocity_domain_feature.get_geometries():

            for velocity_domain_point in velocity_domain_geometry.get_points():

                all_domain_points.append(velocity_domain_point)

                partitioning_plate = plate_partitioner.partition_point(velocity_domain_point)
                if partitioning_plate:

                    # We need the newly assigned plate ID
                    # to get the equivalent stage rotation of that tectonic plate.
                    partitioning_plate_id = partitioning_plate.get_feature().get_reconstruction_plate_id()

                    # Get the stage rotation of partitioning plate from 'time + delta_time' to 'time'.
                    equivalent_stage_rotation = rotation_model.get_rotation(reconstruction_time,
                                                                            partitioning_plate_id,
                                                                            reconstruction_time + delta_time)

                    # Calculate velocity at the velocity domain point.
                    # This is from 'time + delta_time' to 'time' on the partitioning plate.
                    velocity_vectors = pygplates.calculate_velocities(
                        [velocity_domain_point],
                        equivalent_stage_rotation,
                        delta_time)

                    # Convert global 3D velocity vectors to local (magnitude, azimuth, inclination) tuples
                    # (one tuple per point).
                    velocities =pygplates.LocalCartesian.convert_from_geocentric_to_north_east_down(
                            [velocity_domain_point],
                            velocity_vectors)
                    all_velocities.append((velocities[0].get_x(), velocities[0].get_y()))

                else:
                    all_velocities.append((0,0))
                    
    return np.array(all_velocities)

def get_valid_geometries(shape_filename):
    """ only return valid geometries """
    import cartopy.io.shapereader as shpreader
    shp_geom = shpreader.Reader(shape_filename).geometries()
    geometries = []
    for record in shp_geom:
        if record.is_valid:
            geometries.append(record.buffer(0.0))
    return geometries


def get_reconstructed_polygons(feature, rotation_model, time):
    import shapely

    reconstructed_feature = []
    pygplates.reconstruct(feature, rotation_model, reconstructed_feature, float(time))

    all_geometries = []
    for feature in reconstructed_feature:

        # get geometry in lon lat order
        geometry = feature.get_reconstructed_geometry().to_lat_lon_array()[::-1,::-1]

        # construct shapely geometry
        geom = shapely.geometry.Polygon(geometry)

        # we need to make sure the exterior coordinates are ordered anti-clockwise
        # and the geometry is valid otherwise it will screw with cartopy
        if not geom.exterior.is_ccw:
            geom.exterior.coords = list(geometry[::-1])
        if geom.is_valid:
            all_geometries.append(geom)

    return all_geometries


def get_reconstructed_lines(feature, rotation_model, time):
    import shapely

    reconstructed_feature = []
    pygplates.reconstruct(feature, rotation_model, reconstructed_feature, float(time))

    all_geometries = []
    for feature in reconstructed_feature:

        # get geometry in lon lat order
        geometry = feature.get_reconstructed_geometry().to_lat_lon_array()[::-1,::-1]

        # construct shapely geometry
        geom = shapely.geometry.LineString(geometry)

        # we need to make sure the exterior coordinates are ordered anti-clockwise
        # and the geometry is valid otherwise it will screw with cartopy
        if not geom.exterior.is_ccw:
            geom.exterior.coords = list(geometry[::-1])
        if geom.is_valid:
            all_geometries.append(geom)

    return all_geometries


def shapelify_feature_polygons(features):
    import shapely

    date_line_wrapper = pygplates.DateLineWrapper()

    all_geometries = []
    for feature in features:

        rings = []
        wrapped_polygons = date_line_wrapper.wrap(feature.get_reconstructed_geometry())
        for poly in wrapped_polygons:
            ring = np.array([(p.get_longitude(), p.get_latitude()) for p in poly.get_exterior_points()])
            ring[:,1] = np.clip(ring[:,1], -89, 89) # anything approaching the poles creates artefacts
            ring_polygon = shapely.geometry.Polygon(ring)

            # we need to make sure the exterior coordinates are ordered anti-clockwise
            # and the geometry is valid otherwise it will screw with cartopy
            if not ring_polygon.exterior.is_ccw:
                ring_polygon.exterior.coords = list(ring[::-1])

            rings.append(ring_polygon)

        geom = shapely.geometry.MultiPolygon(rings)

        # we need to make sure the exterior coordinates are ordered anti-clockwise
        # and the geometry is valid otherwise it will screw with cartopy
        all_geometries.append(geom.buffer(0.0)) # add 0.0 buffer to deal with artefacts

    return all_geometries

def shapelify_feature_lines(features):
    import shapely

    all_geometries = []
    for feature in features:

        # get geometry in lon lat order
        geometry = feature.get_reconstructed_geometry().to_lat_lon_array()[::-1,::-1]

        # construct shapely geometry
        geom = shapely.geometry.LineString(geometry)

        # we need to make sure the exterior coordinates are ordered anti-clockwise
        # and the geometry is valid otherwise it will screw with cartopy
        if not geom.exterior.is_ccw:
            geom.exterior.coords = list(geometry[::-1])
        if geom.is_valid:
            all_geometries.append(geom)

    return all_geometries



def get_ridge_transforms(topology_features, rotation_model, reconstruction_time):
    reconstruction_time = float(reconstruction_time)
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(topology_features,
                                 rotation_model,
                                 resolved_topologies,
                                 reconstruction_time,
                                 shared_boundary_sections)

    # Iterate over the resolved topologies.
    resolved_topology_features = []
    for resolved_topology in resolved_topologies:
        resolved_topology_features.append(resolved_topology.get_resolved_feature())

    boundary_MOR = pygplates.FeatureType.create_gpml('MidOceanRidge')
    ridge_transform_boundary_section_features = []


    # Iterate over the shared boundary sections.
    for shared_boundary_section in shared_boundary_sections:
        boundary_type = shared_boundary_section.get_feature().get_feature_type()

        # Get all the geometries of the current boundary section.
        boundary_section_features = [shared_sub_segment.get_resolved_feature()
                for shared_sub_segment in shared_boundary_section.get_shared_sub_segments()]

        if boundary_type == boundary_MOR:
            ridge_transform_boundary_section_features.extend(boundary_section_features)

    return ridge_transform_boundary_section_features


def save_ridge_transforms(filename, topology_features, rotation_model, reconstruction_time):
    ridge_transform_boundary_section_features = get_ridge_transforms(
        topology_features, rotation_model, reconstruction_time)

    ridge_transform_boundary_section_feature_collection = pygplates.FeatureCollection(
        ridge_transform_boundary_section_features)
    
    ridge_transform_boundary_section_feature_collection.write(filename)


# shp_subdL.geometries()


# subduction teeth
def tesselate_triangles(shapefilename, tesselation_radians, triangle_base_length, triangle_aspect=1.0):
    """
    Place subduction teeth along line segments within a MultiLineString shapefile
    
    Parameters
    ----------
        shapefilename  : str  path to shapefile
        tesselation_radians : float
        triangle_base_length : float  length of base
        triangle_aspect : float  aspect ratio
        
    Returns
    -------
        X_points : (n,3) array of triangle x points
        Y_points : (n,3) array of triangle y points
    """

    import shapefile
    shp = shapefile.Reader(shapefilename)

    tesselation_degrees = np.degrees(tesselation_radians)
    triangle_pointsX = []
    triangle_pointsY = []

    for i in range(len(shp)):
        pts = np.array(shp.shape(i).points)

        cum_distance = 0.0
        for p in range(len(pts) - 1):

            A = pts[p]
            B = pts[p+1]

            AB_dist = B - A
            AB_norm = AB_dist / np.hypot(*AB_dist)
            cum_distance += np.hypot(*AB_dist)

            # create a new triangle if cumulative distance is exceeded.
            if cum_distance >= tesselation_degrees:

                C = A + triangle_base_length*AB_norm

                # find normal vector
                AD_dist = np.array([AB_norm[1], -AB_norm[0]])
                AD_norm = AD_dist / np.linalg.norm(AD_dist)

                C0 = A + 0.5*triangle_base_length*AB_norm

                # project point along normal vector
                D = C0 + triangle_base_length*triangle_aspect*AD_norm

                triangle_pointsX.append( [A[0], C[0], D[0]] )
                triangle_pointsY.append( [A[1], C[1], D[1]] )

                cum_distance = 0.0

    shp.close()
    return np.array(triangle_pointsX), np.array(triangle_pointsY)

def tesselate_arrows(shapefilename, tesselation_radians, arrow_length, buffer=0.0, near_points=None, **kwargs):
    
    import shapefile
    shp = shapefile.Reader(shapefilename)
    
    if near_points is None:
        d = np.array(0.0)
    else:
        from scipy.spatial import cKDTree
        tree = cKDTree(near_points)
        
    d_tol = 0.5

    tesselation_degrees = np.degrees(tesselation_radians)
    arrow_pointsX = []
    arrow_pointsY = []
    
    for i in range(len(shp)):
        pts = np.array(shp.shape(i).points)
        
        cum_distance = 0.0
        for p in range(len(pts) - 1):
            
            A = pts[p]
            B = pts[p+1]
            
            AB_dist = B - A
            AB_norm = AB_dist / np.linalg.norm(AB_dist)
            cum_distance += np.linalg.norm(AB_dist)
            
            if near_points is not None:
                d, idx = tree.query(A)
            
            if cum_distance >= tesselation_degrees and d < d_tol:
                
                AD_dist = np.array([AB_norm[1], -AB_norm[0]])
                AD_norm = AD_dist / np.linalg.norm(AD_dist)
                
                dC = arrow_length*AD_norm
                
                arrow_pointsX.append( [A[0], dC[0]] )
                arrow_pointsY.append( [A[1], dC[1]] )
                
    return np.array(arrow_pointsX), np.array(arrow_pointsY)