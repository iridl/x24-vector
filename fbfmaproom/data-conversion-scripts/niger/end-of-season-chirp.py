import cftime
import xarray as xr

ds = xr.open_dataset('cos_chirp.nc').rename(Time='time')
ds['time'] = [cftime.Datetime360Day(x, 8, 16) for x in ds['time']]
print(ds)
ds.to_zarr('niger-end-of-season-chirp.zarr')
