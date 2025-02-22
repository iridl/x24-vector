# Drop S from the forecast files for compatibility with hindcasts
# generated with an earlier version of pycpt. Includes a workaround
# for an xarray bug.

import xarray as xr

ds = xr.open_dataset('MME_forecast_prediction_error_variance_2023.nc')
ds = ds.drop_vars('S')
ds['prediction_error_variance'].variable.encoding['coordinates'] = 'Tf Ti'
ds.to_netcdf('var.nc')
print(xr.open_dataset('var.nc'))

ds = xr.open_dataset('MME_deterministic_forecast_2023.nc')
ds = ds.drop_vars('S')
ds['deterministic'].variable.encoding['coordinates'] = 'Tf Ti'
ds.to_netcdf('mu.nc')
print(xr.open_dataset('mu.nc'))
