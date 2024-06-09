import xarray as xr
    
# defining the path to file  
ds_zarr = '../data/remic/mydatafiles/zarr/test/'

ds_nc = "/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/MPI-ESM1-2-HR/mpi-esm1-2-hr_r1i1p1f1_w5e5_historical_hurs_global_daily_1951_1960.nc" 
  
# using the Dataset() function  
zarr_dSet = xr.open_zarr(ds_zarr)

nc_dSet = xr.open_dataset(ds_nc)

#compare nc vs zarred datasets
#comparison = nc_dSet.compare(zarr_dSet, cd 
