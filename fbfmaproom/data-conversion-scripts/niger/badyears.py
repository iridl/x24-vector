import cftime
import pandas as pd

df = pd.read_csv('NIGERBadYear4ToolDec22Workshop.csv')
df['T'] = list(map(lambda x: cftime.Datetime360Day(x, 8, 16), df['year']))
df = df.set_index('T')
df = df.drop('year', axis='columns')
ds = df.to_xarray()
print(ds)
ds.to_zarr('niger-bad-years-2.zarr')
