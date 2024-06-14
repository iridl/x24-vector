import xarray as xr
import os
import re


ds_zarr = '/data/remic/mydatafiles/zarr/test/hurs'
ds_nc_dir = '/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/MPI-ESM1-2-HR/'

#only using NetCDF files that contain 'hurs' variable
def contains_hurs(filename):
    return 'hurs' in filename

#extracting years from nc file
def extract_years(filename):
    match = re.search(r'(\d{4})_(\d{4})\.nc$', filename)
    if match:
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        return start_year, end_year
    else:
        raise ValueError("Unable to find years")

#function that compares nc files(hurs) vs zarred file
def compare_datasets(nc_file_path, ds_zarr):
    #returns true if datasets are identical, false otherwise

    #open datasets
    zarr_dSet = xr.open_zarr(ds_zarr)
    nc_dSet = xr.open_dataset(nc_file_path)[['hurs']]


    #extracting start and end years from nc file
    start_year, end_year = extract_years(os.path.basename(nc_file_path))

    time_start = f'{start_year}-01-01'
    time_end = f'{end_year}-12-31'

    #check if datasets are empty
    if len(nc_dSet) == 0 or len(zarr_dSet) == 0:
        print("Datasets are empty")
        return False

    
    #mapping from zar to nc dimensions (X,Y,T) vs (lon,lat,time)
    coord_map = {'lon': 'X', 'lat': 'Y', 'time': 'T'}

    #xararay rename function to reset nc labels to zarr
    nc_dSet = nc_dSet.rename({orig_dim: new_dim for orig_dim, new_dim in coord_map.items() if orig_dim in nc_dSet.dims})
    nc_dSet = nc_dSet.rename({orig_coord: new_coord for orig_coord, new_coord in coord_map.items() if orig_coord in nc_dSet.coords})

    nc_dSet_sliced = nc_dSet.sel(T=slice(time_start, time_end))
    zarr_dSet_sliced = zarr_dSet.sel(T=slice(time_start, time_end))
    
    
    #testing.assert_equal is an xarray tool to compare datasets (see if they are identical)
    try:
        xr.testing.assert_equal(nc_dSet_sliced, zarr_dSet_sliced)
        print('Datasets are identical')
        
    except AssertionError as e:
        print("Datasets are not identical.")
        print(e)  # Print the error message associated with the AssertionError
        

    #find max hurs values for nc and zarr files
    max_nc = nc_dSet_sliced['hurs'].max().values
    max_zarr = zarr_dSet_sliced['hurs'].max().values

    #find min hurs values for nc and zarr files
    min_nc = nc_dSet_sliced['hurs'].min().values
    min_zarr = zarr_dSet_sliced['hurs'].min().values
    
    #all instances where max is found. gives location
    max_nc_loc = nc_dSet_sliced['hurs'] == max_nc
    max_zarr_loc = zarr_dSet_sliced['hurs'] == max_zarr
    min_nc_loc = nc_dSet_sliced['hurs'] == min_nc
    min_zarr_loc = zarr_dSet_sliced['hurs'] == min_zarr

    #coordinates for the max and mins
    max_nc_coords = nc_dSet_sliced.where(max_nc_loc, drop=True).coords
    max_zarr_coords = zarr_dSet_sliced.where(max_zarr_loc, drop=True).coords
    min_nc_coords = nc_dSet_sliced.where(min_nc_loc, drop=True).coords
    min_zarr_coords = zarr_dSet_sliced.where(min_zarr_loc, drop=True).coords

    print(f"Max value in NetCDF file {os.path.basename(nc_file_path)}: {max_nc}")
    print(f"Max value in Zarr dataset for the same period: {max_zarr}")
    print(f"Coordinates of max values in NetCDF file: {max_nc_coords}")
    print(f"Coordinates of max values in Zarr dataset: {max_zarr_coords}")

    print(f"Min value in NetCDF file {os.path.basename(nc_file_path)}: {min_nc}")
    print(f"Min value in Zarr dataset for the same period: {min_zarr}")
    print(f"Coordinates of min values in NetCDF file: {min_nc_coords}")
    print(f"Coordinates of min values in Zarr dataset: {min_zarr_coords}")

    #checking if max of nc and zarr are the same
    try:
        xr.testing.assert_equal(xr.DataArray(max_nc), xr.DataArray(max_zarr))
        print("Max values are identical.")
        max_identical = True
       
    except AssertionError as e:
        print("Max values are not identical.")
        print(e)  # Print the error message associated with the AssertionError

    #checking if min of nc and zarr are the same
    try:
        xr.testing.assert_equal(xr.DataArray(min_nc), xr.DataArray(min_zarr))
        print("Min values are identical.")
        min_identical = True
        
    except AssertionError as e:
        print("Min values are not identical.")
        print(e)  # Print the error message associated with the AssertionError
        
    return max_identical and min_identical

#list of all NetCDF files
nc_files = [os.path.join(ds_nc_dir, f) for f in os.listdir(ds_nc_dir) if f.endswith('.nc')]

#filtered list of NetCDF files with hurs
nc_files_with_hurs = [f for f in nc_files if contains_hurs(f)]

print('Comparing datasets and min/max values...')

#comparing datasets
all_identical = True
for nc_file in nc_files_with_hurs:
    if not compare_datasets(nc_file, ds_zarr):
        all_identical = False

#testing data
if all_identical:
    print("NC and Zarr files are identical")
else:
    print("Datasets are not the same ")








