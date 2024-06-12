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


    
    #testing.assert_equal is an xarray tool to compare datasets (see if they are identical)
    try:
        xr.testing.assert_equal(nc_dSet, zarr_dSet.sel(T=slice(time_start, time_end)))
        print('Datasets are identical')
        return True
    except AssertionError as e:
        print("Datasets are not identical.")
        print(e)  # Print the error message associated with the AssertionError
        return False

#list of all NetCDF files
nc_files = [os.path.join(ds_nc_dir, f) for f in os.listdir(ds_nc_dir) if f.endswith('.nc')]

#filtered list of NetCDF files with hurs
nc_files_with_hurs = [f for f in nc_files if contains_hurs(f)]

print('Comparing...')

all_identical = True
for nc_file in nc_files_with_hurs:
    if not compare_datasets(nc_file, ds_zarr):
        all_identical = False

#testing data
if all_identical:
    print("NC and Zarr files are identical")
else:
    print("Datasets are not the same ")

