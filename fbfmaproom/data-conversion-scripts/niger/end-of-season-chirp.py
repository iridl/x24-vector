import cftime
import xarray as xr

ds = xr.open_dataset('cos_chirp.nc').rename(Time='time')
ds['time'] = [cftime.Datetime360Day(x, 8, 16) for x in ds['time']]
ds = ds.sortby('latitude')
print(ds)
ds.to_zarr('niger-end-of-season-chirp.zarr')
