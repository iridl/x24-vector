import cftime
import numpy as np
import pandas as pd
import xarray as xr

x = np.arange(41.625, 43.4, .25)
y = np.arange(10.875, 12.9, .25)
s = [cftime.Datetime360Day(y, 5, 1) for y in range(1981, 2024)]
p = np.arange(5, 100, 5)

shape = len(p), len(s), len(y), len(x)
data = np.full(shape, 0.0)

da = xr.DataArray(
    data=data,
    coords={'P': p, 'S': s, 'Y': y, 'X': x},
    name='pnep',
)
ds = xr.Dataset().merge(da)
ds.to_zarr('djibouti-pnep-jjas-dummy.zarr')
