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

def daily_groupby_season(daily_data, start_day, start_month, end_day, end_month, time_coord="T"):
  """Groups daily data by yearly seasons from day-month edges of season
  Can then apply functions on each group to create yearly time series of seasonal quantities
  If starting day-month is 29-Feb, uses 1-Mar.
  If ending day-month is 29-Feb, uses 1-Mar and triggers the option to open the right edge of the Interval.
  That means that the last day included in the season will be 29-Feb in leap years and 28-Feb otherwise
  """
  #Deal with leap year cases
  if start_day == 29 and start_month == 2 :
    start_day = 1
    start_month = 3
  right_bool = True
  if end_day == 29 and end_month == 2 :
    end_day = 1
    end_month = 3
    right_bool = False
  #Creates array of edges of the season that will form the bins
  season_bins = daily_data[time_coord].where(
    ((daily_data[time_coord].dt.day==start_day) & (daily_data[time_coord].dt.month==start_month))
    |
    ((daily_data[time_coord].dt.day==end_day) & (daily_data[time_coord].dt.month==end_month)),
    drop=True
  )
  #First valid bin can be the 2nd one depending of order of first time_coord point, start/end day-month
  first_valid_bin = ~((season_bins[0].dt.day == start_day) & (season_bins[0].dt.month == start_month)).values*1
  #First pass to group and then concat only every other season that we want to keep
  daily_groubedby_season = xr.concat(
    [daily_data.groupby_bins(time_coord, season_bins, right=right_bool)[g] for g in list(daily_data.groupby_bins(time_coord, season_bins, right=right_bool).groups.keys())[first_valid_bin::2]],
    time_coord
  )
  #Second pass to recreate the group labels that concat lost
  daily_groubedby_season = daily_groubedby_season.groupby_bins(time_coord, season_bins, right=right_bool)
  return daily_groubedby_season

def seasonal_sum(daily_data, start_day, start_month, end_day, end_month, min_count=None, time_coord="T"):
  """Calculates seasonal totals of daily data in season defined by day-month edges
  """
  #It turns out that having daily_groupby_season concatenate only every other group is not enough to drop entired the undesired seasons
  #sum will return NaN for these so need to use dropna to clean up
  summed_seasons = daily_groupby_season(daily_data, start_day, start_month, end_day, end_month).sum(dim=time_coord, skipna=True, min_count=min_count).dropna(time_coord + "_bins")
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

  #here, one can see clearly that we retain 4 labeled groups, the other every other groups are just not labeled
  #that's why we need to use dropna later on
  print(
    daily_groupby_season(rr_mrg, 29, 11, 5, 2)
  )

  #Here we see we have the groups we want
  print(
    daily_groupby_season(rr_mrg, 29, 11, 5, 2).groups
  )

  print(
    seasonal_sum(rr_mrg, 29, 11, 5, 2, min_count=0)
  )

  print(
    seasonal_sum(rr_mrg, 29, 11, 5, 2, min_count=0).isel(X=150, Y=150).precip.values
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
