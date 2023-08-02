import cftime
import xarray as xr

ds = xr.open_dataset('enacts_cos.nc')
ds = ds.rename(Time='time')
ds['time'] = [cftime.Datetime360Day(x, 8, 16) for x in ds['time']]
ds = ds.sel(time=ds.time > cftime.Datetime360Day(1991, 1, 1))
ds = ds.sortby('latitude')
print(ds)
ds.to_zarr('niger-end-of-season-enacts.zarr')
