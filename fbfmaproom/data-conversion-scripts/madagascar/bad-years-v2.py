import cftime
import pandas as pd

bad = pd.read_csv('MadaBadYear4Tool0622Workshp - MadaBadYear4Tool0622Workshp.csv')

df = bad.set_index(bad['Year'].apply(lambda x: cftime.Datetime360Day(x, 11, 16)).rename('T'))
df = df.drop('Year', axis='columns')
df = df.rename({'Rank': 'rank'}, axis='columns')
ds = df.to_xarray()
ds.to_zarr('bad-years-ond.zarr')

df = bad.set_index(bad['Year'].apply(lambda x: cftime.Datetime360Day(x + 1, 1, 16)).rename('T'))
df = df.drop('Year', axis='columns')
df = df.rename({'Rank': 'rank'}, axis='columns')
ds = df.to_xarray()
ds.to_zarr('bad-years-djf.zarr')
