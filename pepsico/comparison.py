import xarray as xr
#from pathlib import Path 

# defining the path to file
ds_zarr = '/data/remic/mydatafiles/zarr/test/hurs'
print(ds_zarr)
ds_nc = '/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/MPI-ESM1-2-HR/mpi-esm1-2-hr_r1i1p1f1_w5e5_historical_hurs_global_daily_1951_1960.nc'

#function that compares nc vs zarred datasets
def compare_datasets(ds_nc, ds_zarr):
    #returns true if datasets are identical, false otherwise

    #open datasets:q
    zarr_dSet = xr.open_zarr(ds_zarr)

    nc_dSet = xr.open_dataset(ds_nc)

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

