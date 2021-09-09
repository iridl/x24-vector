import numpy as np
import pandas as pd
import xarray as xr

# Date Reading functions


def read_zarr_data(zarr_path):
    zarr_data = xr.open_zarr(zarr_path)
    return zarr_data


# Growing season functions


def onset_date(
    daily_rain,
    early_start_day,
    early_start_month,
    search_days,
    rainy_day,
    running_days,
    running_total,
    min_rainy_days,
    dry_days,
    dry_spell,
    time_coord="T",
):
    """Function reproducing Ingrid onsetDate function
    http://iridl.ldeo.columbia.edu/dochelp/Documentation/details/index.html
    with the exception that:
    output is a random deltatime rather than an actual onset
    earlyStart input is now 2 arguments: day and month (as opposed to 1)
    output is now the timedelta with each year's earlyStart
    (as opposed to the date itself)
    """
    onset_date = daily_rain[
        (daily_rain[time_coord].dt.day == early_start_day)
        & (daily_rain[time_coord].dt.month == early_start_month)
    ]
    onset_date = xr.DataArray(
        data=np.random.randint(0, high=search_days, size=onset_date.shape).astype(
            "timedelta64[D]"
        ),
        dims=onset_date.dims,
        coords=onset_date.coords,
        attrs=dict(
            description="Onset Date",
        ),
    )
    # Tip to get dates from timedelta  early_start_day
    #  onset_date = onset_date[time_coord] + onset_date
    return onset_date


# Time functions


def strftimeb2int(strftimeb):
    strftimeb_all = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    strftimebint = strftimeb_all[strftimeb]
    return strftimebint


def daily_tobegroupedby_season(
    daily_data, start_day, start_month, end_day, end_month, time_coord="T"
):
    """Returns dataset ready to be grouped by with:
    the daily data where all days not in season of interest are dropped
    season_starts:
      an array where the non-dropped days are indexed by the first day of their season
      -- to use to groupby
    seasons_ends: an array with the dates of the end of the seasons
    Can then apply groupby on daily_data against seasons_starts,
    and preserving seasons_ends for the record
    If starting day-month is 29-Feb, uses 1-Mar.
    If ending day-month is 29-Feb, uses 1-Mar and uses < rather than <=
    That means that the last day included in the season will be 29-Feb in leap years
    and 28-Feb otherwise
    """
    # Deal with leap year cases
    if start_day == 29 and start_month == 2:
        start_day = 1
        start_month = 3
    # Find seasons edges
    start_edges = daily_data[time_coord].where(
        lambda x: (x.dt.day == start_day) & (x.dt.month == start_month),
        drop=True,
    )
    if end_day == 29 and end_month == 2:
        end_edges = daily_data[time_coord].where(
            lambda x: (
                (x + np.timedelta64(1, "D")).dt.day
                == 1 
            )
            & (
                (x + np.timedelta64(1, "D")).dt.month
                == 3
            ),
            drop=True,
        )
    else:
        end_edges = daily_data[time_coord].where(
            lambda x: (x.dt.day == end_day) & (x.dt.month == end_month),
            drop=True,
        )
    # Drop dates outside very first and very last edges
    #  -- this ensures we get complete seasons with regards to edges, later on
    daily_data = daily_data.sel(**{time_coord: slice(start_edges[0], end_edges[-1])})
    start_edges = start_edges.sel(**{time_coord: slice(start_edges[0], end_edges[-1])})
    end_edges = end_edges.sel(
        **{time_coord: slice(start_edges[0], end_edges[-1])}
    ).assign_coords(**{time_coord: start_edges[time_coord]})
    # Drops daily data not in seasons of interest
    days_in_season = (
        daily_data[time_coord] >= start_edges.rename({time_coord: "group"})
    ) & (daily_data[time_coord] <= end_edges.rename({time_coord: "group"}))
    days_in_season = days_in_season.sum(dim="group")
    daily_data = daily_data.where(days_in_season == 1, drop=True)
    # Creates seasons_starts that will be used for grouping
    # and seasons_ends that is one of the outputs
    seasons_groups = (daily_data[time_coord].dt.day == start_day) & (
        daily_data[time_coord].dt.month == start_month
    )
    seasons_groups = seasons_groups.cumsum() - 1
    seasons_starts = (
        start_edges.rename({time_coord: "toto"})[seasons_groups]
        .drop_vars("toto")
        .rename("seasons_starts")
    )
    seasons_ends = end_edges.rename({time_coord: "group"}).rename("seasons_ends")
    # Dataset output
    daily_tobegroupedby_season = xr.merge([daily_data, seasons_starts, seasons_ends])
    return daily_tobegroupedby_season


# Seasonal Functions


def seasonal_sum(
    daily_data,
    start_day,
    start_month,
    end_day,
    end_month,
    min_count=None,
    time_coord="T",
):
    """Calculates seasonal totals of daily data in season defined by day-month edges"""
    grouped_daily_data = daily_tobegroupedby_season(
        daily_data, start_day, start_month, end_day, end_month
    )
    seasonal_data = (
        grouped_daily_data[daily_data.name]
        .groupby(grouped_daily_data["seasons_starts"])
        .sum(dim=time_coord, skipna=True, min_count=min_count)
        .rename({"seasons_starts": time_coord})
    )
    seasons_ends = grouped_daily_data["seasons_ends"].rename({"group": time_coord})
    summed_seasons = xr.merge([seasonal_data, seasons_ends])
    return summed_seasons


# Testing Functions


def run_test_season_stuff():
    import pyaconf
    import os
    from pathlib import Path

    CONFIG = pyaconf.load(os.environ["CONFIG"])
    DR_PATH = CONFIG["daily_rainfall_path"]
    RR_MRG_ZARR = Path(DR_PATH)
    rr_mrg = read_zarr_data(RR_MRG_ZARR)
    rr_mrg = rr_mrg.sel(T=slice("2000", "2005-02-28"))

    print("the inputs to grouping")
    print(daily_tobegroupedby_season(rr_mrg.precip, 29, 11, 29, 2))

    print("the outputs of seasonal_sum")
    print(seasonal_sum(rr_mrg.precip, 29, 11, 29, 2, min_count=0))

    print("some data")
    print(
        seasonal_sum(rr_mrg.precip, 29, 11, 29, 2, min_count=0)
        .precip.isel(X=150, Y=150)
        .values
    )
    print(
        rr_mrg.precip.sel(T=slice("2000-11-29", "2001-02-28"))
        .sum("T")
        .isel(X=150, Y=150)
        .values
    )
    print(
        rr_mrg.precip.sel(T=slice("2001-11-29", "2002-02-28"))
        .sum("T")
        .isel(X=150, Y=150)
        .values
    )
    print(
        rr_mrg.precip.sel(T=slice("2002-11-29", "2003-02-28"))
        .sum("T")
        .isel(X=150, Y=150)
        .values
    )
    print(
        rr_mrg.precip.sel(T=slice("2003-11-29", "2004-02-29"))
        .sum("T")
        .isel(X=150, Y=150)
        .values
    )


run_test_season_stuff()
