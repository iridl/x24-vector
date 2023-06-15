import cftime
import openpyxl
import pandas as pd
import xarray as xr

wb = openpyxl.load_workbook('rankings_ni.xlsx')
sheet = wb['Sheet1']
cols = list(sheet.columns)

years_col = [x.value for x in cols[0]]
assert years_col[0] == 'Years'
years = [cftime.Datetime360Day(x, 8, 16) for x in years_col[1:]]

ranking_col = [x.value for x in cols[1]]
assert ranking_col[0] == 'Ranking'

df = pd.DataFrame(index=years, data={'rank': ranking_col[1:]})
df.index.name = 'T'

ds = df.to_xarray()
print(ds)
ds.to_zarr('niger-bad-years-v3.zarr')
