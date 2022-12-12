import pandas as pd
import xarray as xr
import cftime

fname = 'central_madagascar_onset_Demise_LRS.xlsx'
df = pd.read_excel(fname, usecols='A,B,D', nrows=40)
df['time'] = df['Year'].apply(lambda x: cftime.Datetime360Day(x+1, 1, 16))
df = df.set_index('time')[['onset', 'LORS']]
ds = xr.Dataset(df)
print(ds)
ds.to_zarr('onset.zarr')
