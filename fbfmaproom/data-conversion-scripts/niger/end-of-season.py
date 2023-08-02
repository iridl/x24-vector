import cftime
import xarray as xr

ds = xr.open_dataset('cos.nc')
ds['time'] = [cftime.Datetime360Day(x, 8, 16) for x in range(1991, 2023)]
ds = ds.sortby('latitude')
print(ds)
ds.to_zarr('niger-end-of-season.zarr')
