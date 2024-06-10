import xarray as xr
#from pathlib import Path 

# defining the path to file
ds_zarr = '/data/remic/mydatafiles/zarr/test/hurs'
print(ds_zarr)
ds_nc = '/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/MPI-ESM1-2-HR/mpi-esm1-2-hr_r1i1p1f1_w5e5_historical_hurs_global_daily_1951_1960.nc'
#only 9 years (need to compare ALL the years)
#renamed variables

#function that compares nc vs zarred datasets
def compare_datasets(ds_nc, ds_zarr):
    #returns true if datasets are identical, false otherwise

    #open datasets
    zarr_dSet = xr.open_zarr(ds_zarr)

    nc_dSet = xr.open_dataset(ds_nc)

    #mapping from zar to nc dimensions (X,Y,T) vs (lon,lat,time)
    coord_map = {'X': 'lon', 'Y': 'lat', 'T': 'time'}

    #xararay rename function to reset zarr dimensions to nc labels 
    zarr_dSet = zarr_dSet.rename({orig_dim: new_dim for orig_dim, new_dim in coord_map.items() if orig_dim in zarr_dSet.dims})
    zarr_dSet = zarr_dSet.rename({orig_coord: new_coord for orig_coord, new_coord in coord_map.items() if orig_coord in zarr_dSet.coords})

    #align the zarr chunk with overlapping nc data
    time_start = max(nc_dSet['time'].min().values, zarr_dSet['time'].min().values)
    time_end = min(nc_dSet['time'].max().values, zarr_dSet['time'].max().values)
    nc_dSet = nc_dSet.sel(time=slice(time_start, time_end))
    zarr_dSet = zarr_dSet.sel(time=slice(time_start, time_end))


    #renaming zarr variables to match nc
    #testing.assert_equal is an xarray tool to compare datasets
    try:
        xr.testing.assert_equal(nc_dSet, zarr_dSet)
        return True
    except AssertionError as e:
        print("Datasets are not identical.")
        print(e)  # Print the error message associated with the AssertionError
        return False

#testing data
if compare_datasets(ds_nc, ds_zarr):
    print("NC and Zarr files are the same")
else:
    print("Datasets are not the same ")

