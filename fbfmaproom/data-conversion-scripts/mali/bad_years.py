import cftime
import datetime
import pandas as pd
import xarray as xr

bad = pd.read_csv('bad_years_emdat.csv')

years = [ cftime.Datetime360Day(y, 8, 16) for y in bad.iloc[:,0].to_list() ]
ranks = bad.iloc[:,1].to_list()

ds = xr.Dataset(data_vars={"bad": xr.DataArray(ranks, coords={"T": years})})

ds.to_zarr('/data/aaron/fbf-candidate/mali/bad-years-emdat.zarr', mode='w')
