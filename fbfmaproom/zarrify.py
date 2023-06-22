import cptio
import datetime
import glob
import numpy as np
from pathlib import Path
import re
import scipy.stats
import xarray as xr

SOURCE_ROOT = Path('/')
DEST_ROOT = Path('/data/aaron/fbf-test')
datasets = [
    [
        'data/aaron/NigerFBF2023/prcp/jas'
        'niger/prcp/jas-v4'
    ],
]

hindcast_file = 'niger-jun-hindcasts-rewritten.tsv'
obs_file = SOURCE_ROOT / 'data/aaron/NigerFBFlate2022/JAS/obs_PRCP_Jul-Sep.tsv'
mu_files =  sorted(glob.glob(str(SOURCE_ROOT / 'data/aaron/NigerFBFlate2022/JAS/Forecast_mu/June/NextGen_PRCPPRCP_CCAFCST_mu_Jul-Sep_Jun*.tsv')))
var_files = sorted(glob.glob(str(SOURCE_ROOT / 'data/aaron/NigerFBFlate2022/JAS/Forecast_var/June/NextGen_PRCPPRCP_CCAFCST_var_Jul-Sep_Jun*.tsv')))

hindcasts = cptio.open_cptdataset(hindcast_file)
hindcasts = hindcasts.assign_coords(
    {
        'S': (
            'T',
            [
                datetime.datetime(d.dt.year.item(), 6, 1)
                for d in hindcasts['T']
            ]
        )
    }
)
hindcasts = hindcasts.swap_dims(T='S')
obs = cptio.open_cptdataset(obs_file)

def fixyear_date(t, year):
    return datetime.datetime(year, t.dt.month.item(), t.dt.day.item())

def fixyear_slice(ds, year):
    for coord in 'T', 'Ti', 'Tf', 'S':
        ds = ds.assign_coords(
            {
                coord: (
                    'T',
                    [fixyear_date(t, year) for t in ds[coord]]
                )
            }
        )
    return ds

def load(filenames):
    slices = [
        (cptio.open_cptdataset(fname), extract_year(fname))
        for fname in filenames
    ]
    fixed_slices = [fixyear_slice(ds, year) for ds, year in slices]
    return xr.concat(fixed_slices, 'T').swap_dims(T='S')

filename_re = re.compile('(\d\d\d\d).tsv$')
def extract_year(filename):
    return int(filename_re.search(filename).group(1))

mu = load(mu_files)
var = load(var_files)

# np.arange(.05, 1, .05) yields quantiles that are slightly off, e.g. .9500000001
quantiles = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
thresholds = hindcasts.sel(S=slice('1991-06-01', '2016-06-01')).quantile(quantiles, 'S')
ntrain = 2016 - 1991 + 1
dof = ntrain - 1

climo_mu = hindcasts.mean('S')
climo_var = hindcasts.var('S')
climo_scale = np.sqrt((dof - 1) / dof * climo_var)
def func(thresh, mu, scale, *, dof):
    return scipy.stats.t.cdf(thresh, dof, mu, scale)
pne = xr.apply_ufunc(func, thresholds, climo_mu, climo_scale, kwargs={'dof': dof})
print(pne.sel(quantile=.3).isel(X=0, Y=0))
