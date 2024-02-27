import cftime
from datetime import timedelta
import xarray as xr

jas = xr.open_zarr('/data/aaron/fbf/djibouti/bad-years.zarr')
jas['T'] =jas['T'] - timedelta(days=15)
print(jas)
jas.to_zarr('djibouti-bad-years-jjas.zarr')
