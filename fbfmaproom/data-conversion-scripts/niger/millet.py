import cftime
import openpyxl
import pandas as pd
import xarray as xr

wb = openpyxl.load_workbook('millet yield.xlsx')
sheet = wb[wb.sheetnames[0]]
cols = list(sheet.columns)

year_col = [x.value for x in cols[0]]
assert year_col[0] == 'year'
years = [cftime.Datetime360Day(x, 8, 16) for x in year_col[1:]]

yield_col = [x.value for x in cols[1]]
assert yield_col[0] == 'millet_national_yield'

df = pd.DataFrame(index=years, data={'yield': yield_col[1:]})
df.index.name = 'T'

ds = df.to_xarray()
print(ds)
ds.to_zarr('niger-millet.zarr')
