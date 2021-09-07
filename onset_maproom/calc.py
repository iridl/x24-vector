import numpy as np
import pandas as pd
import xarray as xr

#Date Reading functions

def read_zarr_data(zarr_path):
    zarr_data = xr.open_zarr(zarr_path)
    return zarr_data

#Growing season functions

def onset_date(daily_rain, early_start_day, early_start_month, search_days, rainy_day, running_days, running_total, min_rainy_days, dry_days, dry_spell, time_coord="T"):
    """Function reproducing Ingrid onsetDate function
    http://iridl.ldeo.columbia.edu/dochelp/Documentation/details/index.html?func=onsetDate
    with the exception that:
    output is a random deltatime rather than an actual onset
    earlyStart input is now 2 arguments: day and month (as opposed to 1)
    output is now the timedelta with each year's earlyStart (as opposed to the date itself)
    """
    onset_date = daily_rain[
      (daily_rain[time_coord].dt.day==early_start_day)
      &
      (daily_rain[time_coord].dt.month==early_start_month)
    ]
    onset_date = xr.DataArray(
      data=np.random.randint(0, high=search_days, size=onset_date.shape).astype('timedelta64[D]'),
      dims=onset_date.dims,
      coords=onset_date.coords,
      attrs=dict(
          description="Onset Date",
      ),
    )
   #Tip to get dates from timedelta  early_start_day
   #  onset_date = onset_date[time_coord] + onset_date  
    return onset_date

# Time functions

def daily_tobegroupedby_season(daily_data, start_day, start_month, end_day, end_month, time_coord="T"):
    """Returns dataset ready to be grouped by with:
    the daily data where all days not in season of interest are dropped
    season_starts: an array where the non-dropped days are indexed by the first day of their season -- to use to groupby
    seasons_ends: an array with the dates of the end of the seasons
    Can then apply groupby on daily_data against seasons_starts, and preserving seasons_ends for the record
    If starting day-month is 29-Feb, uses 1-Mar.
    If ending day-month is 29-Feb, uses 1-Mar and triggers the option to open the right edge of the Interval.
    That means that the last day included in the season will be 29-Feb in leap years and 28-Feb otherwise
    """
    #Deal with leap year cases
    if start_day == 29 and start_month == 2 :
      start_day = 1
      start_month = 3
    end_day2 = end_day
    end_month2 = end_month
    if end_day == 29 and end_month == 2 :
      end_day2 = 1
      end_month2 = 3
    start_edges = daily_data[time_coord].where(
      ((daily_data[time_coord].dt.day==start_day) & (daily_data[time_coord].dt.month==start_month)),
      drop=True
    )
    end_edges = daily_data[time_coord].where(
      ((daily_data[time_coord].dt.day==end_day2) & (daily_data[time_coord].dt.month==end_month2)),
      drop=True
    )
    #Drop date outside very first and very last edges -- this ensures we get complete seasons with regards to edges, later on
    daily_data = daily_data.sel(**{time_coord: slice(start_edges[0],end_edges[-1])})
    start_edges = daily_data[time_coord].where(
      ((daily_data[time_coord].dt.day==start_day) & (daily_data[time_coord].dt.month==start_month)),
      drop=True
    )
    end_edges = daily_data[time_coord].where(
      ((daily_data[time_coord].dt.day==end_day2) & (daily_data[time_coord].dt.month==end_month2)),
      drop=True
    )
    #Creates array of edges of the season that will form the bins
    seasons_edges = xr.concat([start_edges, end_edges], "T_out", join="override")
    #Creates seasons_starts that will be used for grouping
    #and seasons_ends that is one of the outputs
    if end_day == 29 and end_month == 2 :
      days_in_season = (
        daily_data[time_coord] >= seasons_edges.isel(T_out=0).rename({time_coord: "group"})
      ) & (
        daily_data[time_coord] < seasons_edges.isel(T_out=1).rename({time_coord: "group"})
      )
    else:
      days_in_season = (
        daily_data[time_coord] >= seasons_edges.isel(T_out=0).rename({time_coord: "group"})
      ) & (
        daily_data[time_coord] <= seasons_edges.isel(T_out=1).rename({time_coord: "group"})
      )
    seasons_starts = daily_data[time_coord].where(days_in_season)
    seasons_starts = (
      xr.where(seasons_starts >= seasons_starts["group"], seasons_starts["group"], seasons_starts)
      .min(dim="group", skipna=True)
      .dropna(dim=time_coord)
      .rename("seasons_starts")
    )
    seasons_ends = (
      daily_data[time_coord].where(days_in_season)
      .max(dim="T", skipna=True)
      .rename("seasons_ends")
    )
    #Drops daily data not in seasons of interest
    daily_data = daily_data.where(seasons_starts, drop=True)
    #Group by season with groups labeled by seasons_starts
    daily_tobegroupedby_season = xr.merge([daily_data, seasons_starts, seasons_ends])
    return daily_tobegroupedby_season

def seasonal_sum(daily_data, start_day, start_month, end_day, end_month, min_count=None, time_coord="T"):
    """Calculates seasonal totals of daily data in season defined by day-month edges
    """
    grouped_daily_data = daily_tobegroupedby_season(daily_data, start_day, start_month, end_day, end_month)
    seasonal_data = (
      grouped_daily_data[daily_data.name]
      .groupby(grouped_daily_data["seasons_starts"])
      .sum(dim=time_coord, skipna=True, min_count=min_count)
      .rename({"seasons_starts": time_coord})
    )
    seasons_ends = (
      grouped_daily_data["seasons_ends"]
      .rename({"group": time_coord})
    )
    summed_seasons = xr.merge([seasonal_data, seasons_ends])
    return summed_seasons

def run_test_season_stuff():
    import pyaconf
    import os
    from pathlib import Path

    CONFIG = pyaconf.load(os.environ["CONFIG"])
    DR_PATH = CONFIG["daily_rainfall_path"]
    RR_MRG_ZARR = Path(DR_PATH)
    rr_mrg = read_zarr_data(RR_MRG_ZARR)
    rr_mrg = rr_mrg.sel(T=slice("2000", "2004"))

    print("the inputs to grouping")
    print(
      daily_tobegroupedby_season(rr_mrg.precip, 29, 11, 5, 2)
    )

    print("the outputs of seasonal_sum")
    print(
      seasonal_sum(rr_mrg.precip, 29, 11, 5, 2, min_count=0)
    )

    print("some data")
    print(
      seasonal_sum(rr_mrg.precip, 29, 11, 5, 2, min_count=0).precip.isel(X=150, Y=150).values
    )

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
      "Dec": 12
    }
    strftimebint=strftimeb_all[strftimeb]
    return strftimebint

run_test_season_stuff()

