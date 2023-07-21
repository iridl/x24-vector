import cftime
import datetime
import pandas as pd
import xarray as xr

season_month1 = 4.5

df0 = pd.read_csv('ethiopia_aa_trigger_foraaron_adm0.csv')
df0['geom_key'] = 'ET05'

months = df0['month'].unique()
severities = df0['severity'].unique()

df3 = pd.read_csv('ethiopia_aa_trigger_foraaron_adm3.csv').rename(columns={'key_combined': 'geom_key'})
assert set(df3['month'].unique()) == set(months)
assert set(df3['severity'].unique()) == set(severities)

df = pd.concat([df0, df3])

YEAR = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def monthnum(s):
    return YEAR.index(s)

def season_date(row):
    return (
        cftime.Datetime360Day(row['year'], 1, 1) +
        datetime.timedelta(days=(season_month1 - 1) * 30)
    )

df['time'] = df.apply(season_date, axis=1)

for month in months:
    for severity in severities:
        subdf = df[(df['month'] == month) & (df['severity'] == severity)]
        subdf = subdf.set_index(['geom_key', 'time'])
        name = f'triggers-ond-{month}-{severity}.zarr'
        ds = subdf['trig_adj_yn'].to_xarray().to_dataset()
        print(name)
        print(ds)
        ds.to_zarr(name)

