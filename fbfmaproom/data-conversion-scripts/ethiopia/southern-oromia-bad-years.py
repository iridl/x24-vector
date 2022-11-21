from cftime import Datetime360Day as DT
import xarray as xr

seasons = {
    'mam': lambda year: DT(year, 4, 16),
    'ond': lambda year: DT(year, 11, 16),
}

badyears = {
    1997,
    1998,
    2000,
    2008,
    2011,
    2015,
    2017,
    2019,
    2022,
}


def makedata(season, firstyear, lastyear):
    ds = xr.Dataset(
        {
            'bad': (
                'T',
                [
                    1. if y in badyears else 0.
                    for y in range(firstyear, lastyear+1)
                ]
            ),
        },
        {
            'T': [seasons[season](y) for y in range(firstyear, lastyear+1)],
        },
    )
    print(season)
    print(ds)
    ds.to_zarr(f'southern-oromia-bad-years-{season}.zarr')

makedata('mam', 1990, 2022)
makedata('ond', 1990, 2021)
