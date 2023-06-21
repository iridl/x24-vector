import cftime
import openpyxl
import pandas as pd
import xarray as xr

wb = openpyxl.load_workbook('ipc_nigertotal.xlsx')
sheet = wb[wb.sheetnames[0]]
cols = list(sheet.columns)

year_col = [x.value for x in cols[0]]
assert year_col[0] == 'Year'
years = [cftime.Datetime360Day(x, 8, 16) for x in year_col[1:]]

insecure_col = [x.value for x in cols[1]]
assert insecure_col[0] == 'total population food insecure'
insecure_vals = insecure_col[1:]

total_col = [x.value for x in cols[2]]
assert total_col[0] == 'total population'
total_vals = total_col[1:]

df = pd.DataFrame(
    index=years,
    data={
        'insecure': insecure_vals,
        'total': total_vals,
    }
)
df.index.name = 'T'

df['fraction_insecure'] = df['insecure'] / df['total']
df = df.drop(['insecure', 'total'], axis='columns')

ds = df.to_xarray()
print(ds)
ds.to_zarr('niger-insecure-national.zarr')
