import cftime
import datetime
import pandas as pd
import xarray as xr

bad = pd.read_csv('MADBadYear4ToolDec2022Workshop.csv')

years = [ cftime.Datetime360Day(y + 1, 1, 16) for y in bad.iloc[:,0].to_list() ]
ranks = bad.iloc[:,1].to_list()

ds = xr.Dataset(data_vars={"rank": xr.DataArray(ranks, coords={"T": years})})

ds.to_zarr('bad-years-v3-djf.zarr')
