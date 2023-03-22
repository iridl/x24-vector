import cftime
from datetime import timedelta
import xarray as xr

jas = xr.open_zarr('/data/aaron/fbf/djibouti/bad-years.zarr')
jas['T'] =jas['T'] - timedelta(days=4 * 30)
jas.to_zarr('djibouti-bad-years-mam.zarr')
