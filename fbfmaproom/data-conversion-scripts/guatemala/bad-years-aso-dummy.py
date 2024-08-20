import cftime
import datetime
import pandas as pd
import xarray as xr

bad = pd.read_csv('bad_years.csv')

years = [ cftime.Datetime360Day(y + 1, 9, 16) for y in bad.iloc[:,0].to_list() ]
ranks = bad.iloc[:,1].to_list()

ds = xr.Dataset(data_vars={"bad": xr.DataArray(ranks, coords={"T": years})})

ds.to_zarr('bad-years-aso-dummy.zarr')
