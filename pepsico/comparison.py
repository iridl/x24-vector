import xarray as xr
import os


ds_zarr = '/data/remic/mydatafiles/zarr/test/hurs'
ds_nc_dir = '/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/MPI-ESM1-2-HR/'


#only using NetCDF files that contain 'hurs' variable
def contains_hurs(filename):
    return 'hurs' in filename



#function that compares nc files(hurs) vs zarred file
def compare_datasets(nc_file_path, ds_zarr):
    #returns true if datasets are identical, false otherwise

    #open datasets
    zarr_dSet = xr.open_zarr(ds_zarr)
    nc_dSet = xr.open_dataset(nc_file_path)[['hurs']]

    #mapping from zar to nc dimensions (X,Y,T) vs (lon,lat,time)
    coord_map = {'lon': 'X', 'lat': 'Y', 'time': 'T'}

    #xararay rename function to reset nc labels to zarr
    nc_dSet = nc_dSet.rename({orig_dim: new_dim for orig_dim, new_dim in coord_map.items() if orig_dim in nc_dSet.dims})
    nc_dSet = nc_dSet.rename({orig_coord: new_coord for orig_coord, new_coord in coord_map.items() if orig_coord in nc_dSet.coords})

    

    #align the zarr chunk with overlapping nc data
    time_start = max(zarr_dSet['T'].min().values, nc_dSet['T'].min().values) #start time = compare earliest time of both files and choose latest between the 2
    time_end = min(zarr_dSet['T'].max().values, nc_dSet['T'].max().values) #minimimum of the 2 files max time
    nc_dSet = nc_dSet.sel(T=slice(time_start, time_end)) #slicing to overlapping part in nc file
    zarr_dSet = zarr_dSet.sel(T=slice(time_start, time_end)) #overlapping section for zarr file


    #renaming zarr variables to match nc
    #testing.assert_equal is an xarray tool to compare datasets
    try:
        xr.testing.assert_equal(nc_dSet, zarr_dSet)
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

