import cftime
import openpyxl
import pandas as pd
import xarray as xr
import sys

dirout='/data/aaron/fbf-candidate/'

if  len(sys.argv) < 3:
    print("\033[91m"+"Two arguments is expected. (month, country)"+"\033[0m")
    print("\033[1m"+"Example: python bad-years_3Season.py Jan djibouti"+"\033[0m")
    sys.exit()
else:
    midmonth = sys.argv[1]
    country = f"{sys.argv[2]}".lower()
    dirout=dirout+country+"/"

countriesdic = {
    "djibouti": {"file": "/data/aaron/fbf-candidate/original-data/djibouti/Djibouti_bad-years_JAS.xlsx","column":"JAS"}
}

midmonthdic = {
    "Jan": {"month": 1, "season": "DJF"},
    "Feb": {"month": 2, "season": "JFM"},
    "Mar": {"month": 3, "season": "FMA"},
    "Apr": {"month": 4, "season": "MAM"},
    "May": {"month": 5, "season": "AMJ"},
    "Jun": {"month": 6, "season": "MJJ"},
    "Jul": {"month": 7, "season": "JJA"},
    "Aug": {"month": 8, "season": "JAS"},
    "Sep": {"month": 9, "season": "ASO"},
    "Oct": {"month": 10, "season": "SON"},
    "Nov": {"month": 11, "season": "OND"},
    "Dec": {"month": 12, "season": "NDJ"}
}
if not midmonth in midmonthdic:
    print("\033[91m"+"\033[1m"+f"{midmonth}"+"\033[0m"+ "\033[91m"+ " does not exist in the dictionary."+"\033[0m")
    print("\033[1m"+"Months available: "+"\033[0m"+', '.join(midmonthdic.keys()))
    print("\033[1m"+"Example: python bad-years_3Season.py Jan djibouti"+"\033[0m")
    sys.exit()
elif not country in countriesdic:
    print("\033[91m"+"\033[1m"+f"{country}"+"\033[0m"+ "\033[91m"+ " does not exist in the dictionary."+"\033[0m")
    print("\033[1m"+"Countries available: "+"\033[0m"+', '.join(countriesdic.keys()))
    print("\033[1m"+"Example: python bad-years_3Season.py Jan djibouti"+"\033[0m")
    sys.exit()


wb = openpyxl.load_workbook(countriesdic[country]['file'])
sheet = wb['Sheet1']
cols = list(sheet.columns)

years_col = cols[0]
assert years_col[0].value == 'Years'
years = list(map(lambda x: cftime.Datetime360Day(x.value, midmonthdic[midmonth]['month'], 16), years_col[1:]))

jas_col = cols[1]
assert jas_col[0].value == countriesdic[country]['column']
colors = {
    'FFFF0000': 1,  # red (driest)
    'FFFFC000': 2,  # orange
    'FFFFFF00': 3,  # yellow
    'FFBDD6EE': 4,  # light blue
    'FF9CC2E5': 4,  # another light blue
    'FF92D050': 5,  # light green
    'FF00B050': 6,  # dark green (wettest)
}
severities = list(map(lambda x: colors.get(x.fill.fgColor.rgb), jas_col[1:]))

df = pd.DataFrame(index=years, data={'bad': severities})
df.index.name = 'T'

ds = df.to_xarray()
print(ds)
ds.to_zarr(dirout+'bad-years-'+f"{midmonthdic[midmonth]['season']}".lower()+'.zarr')

