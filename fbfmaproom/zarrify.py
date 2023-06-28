import calendar
import cftime
import cptio
import datetime
import glob
import numpy as np
from pathlib import Path
import re
import scipy.stats
import xarray as xr

SOURCE_ROOT = Path('/home/aaron/scratch/iri')
#SOURCE_ROOT = Path('/')
DEST_ROOT = Path('/data/aaron/fbf-test')

def niger_v1_test():
    '''Open a pycpt v1 dataset that has already been set up in Ingrid, for the purpose
    of comparing our results to those of the Ingrid version.'''
    hindcast_file = 'niger-jun-hindcasts-rewritten.tsv'
    obs_file = SOURCE_ROOT / 'data/aaron/NigerFBFlate2022/JAS/obs_PRCP_Jul-Sep.tsv'
    mu_files =  sorted(glob.glob(str(SOURCE_ROOT / 'data/aaron/NigerFBFlate2022/JAS/Forecast_mu/June/NextGen_PRCPPRCP_CCAFCST_mu_Jul-Sep_Jun*.tsv')))
    var_files = sorted(glob.glob(str(SOURCE_ROOT / 'data/aaron/NigerFBFlate2022/JAS/Forecast_var/June/NextGen_PRCPPRCP_CCAFCST_var_Jul-Sep_Jun*.tsv')))

    hindcasts = cptio.open_cptdataset(hindcast_file)['prec']
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
    obs = cptio.open_cptdataset(obs_file)['prcp']

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

    def load(filenames, var):
        slices = [
            (cptio.open_cptdataset(fname)[var], extract_year(fname))
            for fname in filenames
        ]
        fixed_slices = [fixyear_slice(ds, year) for ds, year in slices]
        return xr.concat(fixed_slices, 'T')

    filename_re = re.compile('(\d\d\d\d).tsv$')
    def extract_year(filename):
        return int(filename_re.search(filename).group(1))

    forecast_mu = load(mu_files, 'prec')
    forecast_var = load(var_files, 'prec')
    forecasts = xr.Dataset().merge({'mu': forecast_mu, 'var': forecast_var})

    pne = calc_pne(obs, hindcasts, forecasts, dof=34, quantile_first_year=1991, quantile_last_year=2016)

    #print(pne.sel(quantile=.3).isel(X=0, Y=-1))


def sqrt(x):
    return xr.apply_ufunc(np.sqrt, x)

def tcdf(x, dof, loc=0, scale=1):
    return xr.apply_ufunc(scipy.stats.t.cdf, x, dof, loc, scale)

def calc_pne(obs, hindcasts, forecasts, dof=None, quantile_first_year=None, quantile_last_year=None):
    if dof is None:
        dof = len(hindcasts['T']) - 1
    if quantile_first_year is None:
        quantile_first_year = hindcasts['T'].dt.year[0].item()
    if quantile_last_year is None:
        quantile_last_year = hindcasts['T'].dt.year[-1].item()

    assert len(obs.groupby('T.month')) == 1
    assert len(hindcasts.groupby('T.month')) == 1
    assert len(forecasts.groupby('T.month')) == 1

    # np.arange(.05, 1, .05) yields quantiles that are slightly off, e.g. .9500000001
    quantiles = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]

    thresholds = (
        obs
        .sel(T=slice(f'{quantile_first_year}-01-01', f'{quantile_last_year}-12-31'))
        .quantile(quantiles, 'T')
    )

    hindcast_pne = tcdf(thresholds, dof, hindcasts['mu'], sqrt(hindcasts['var']))
    forecast_pne = tcdf(thresholds, dof, forecasts['mu'], sqrt(forecasts['var']))
    concat_pne = xr.concat([hindcast_pne, forecast_pne], 'T')
    concat_pne = concat_pne.rename(obs='pne')

    return concat_pne

abbrevs = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def read_v2(path):
    issues = []
    for monthno in range(1, 13):
        abbrev = abbrevs[monthno]
        month_path = path / abbrev
        if month_path.exists():
            issue = read_v2_one_issue(month_path)
            #issue.attrs['S'] = ...
            issues.append(issue)
    return xr.concat(issues, 'S')

def convert_ds(ds, issue_month_abbrev):
    # Replace Gregorian T coordinate with the 360-day calendar used by FbF
    ds = to_360_ds(ds)
    
    # Reconstruct the issue date and add it as a non-dimension coordinate.
    # TODO: pycpt has this info at some point; make it save that in the netcdf
    # so we don't have to reconstruct it here.
    issue_month = abbrevs.index(issue_month_abbrev)
    target_month = ds['T'].dt.month[0].item()
    target_day = ds['T'].dt.day[0].item()
    if target_day == 1:
        pass
    elif target_day == 16:
        target_month += .5
    else:
        assert False, f"Unexpected target day {target_day}"
    lead = target_month
    lead = datetime.timedelta(days=((target_month - issue_month) % 12) * 30)
    ds.coords['S'] = ('T', ds['T'].values - lead)
    return ds

def read_v2_one_issue(path):
    hindcasts = xr.Dataset(
        dict(
            mu=xr.open_dataarray(path / 'MME_deterministic_hindcasts.nc'),
            var=xr.open_dataarray(path / 'MME_hindcast_prediction_error_variance.nc'),
        ),
        attrs={'issue_month': path.name}
    )
    forecasts = xr.Dataset(
        dict(
            mu=xr.open_mfdataset(path.glob('MME_deterministic_forecast_*.nc')).load().data_vars.values().__iter__().__next__(),
            var=xr.open_mfdataset(path.glob('MME_forecast_prediction_error_variance_*.nc')).load().data_vars.values().__iter__().__next__(),
        ),
        attrs={'issue_month': path.name}
    )
    obs = xr.Dataset(dict(
        obs=xr.open_dataarray(path / 'UCSB.PRCP.nc') # TODO don't hard-code obs name
    ))
    return hindcasts, forecasts, obs

def to_360_date(year, month, day):
    isleap = calendar.isleap(year)
    if day == 1:
        newday = 1
    elif day in (15, 16): # todo be as pedantic as for the last day?
        newday = 16
    elif (
            day == 28 and month == 2 and not isleap or
            day == 29 and month == 2 and isleap or
            day == 30 and month in (2, 4, 6, 9, 11) or
            day == 31 and month in (1, 3, 5, 7, 8, 10, 12)
    ):
        newday = 30
    else:
        assert False, f"Bad date {year}-{month}-{day}"

    return cftime.Datetime360Day(year, month, newday)

def to_360_coord(c):
    return [
        to_360_date(year, month, day)
        for year, month, day in zip(
                c.dt.year.values,
                c.dt.month.values,
                c.dt.day.values
        )
    ]
                
def to_360_ds(ds):
    # Just dropping Ti and Tf for now. If we need them we can convert them.
    ds = ds.drop_vars(['Ti', 'Tf'])
    ds['T'] = to_360_coord(ds['T'])
    return ds

def load_pne(path):
    hindcasts, forecasts, obs = read_v2_one_issue(path)
    pne = calc_pne(obs, hindcasts, forecasts)
    pne = convert_ds(pne, hindcasts.attrs['issue_month'])
    pne = pne.swap_dims(T='S')
    return pne

ROOT = Path('/home/aaron/scratch/iri/data/aaron/fbf-candidate')

def zarrify(inpath, outpath, shift):
    pne = load_pne(ROOT / inpath)
    pne = pne.drop_vars('T') # xr.where doesn't like the non-dimension coord?
    pne['quantile'] = (pne['quantile'] * 100).astype(int)
    pne['pne'] = pne['pne'] * 100
    print(pne)
    pne.to_zarr(ROOT / outpath)

#zarrify('niger/pnep-jja/May', 'niger/pnep-jja.zarr', 1)
zarrify('niger/pnep-aso/Jul', 'niger/pnep-aso.zarr', 1)

