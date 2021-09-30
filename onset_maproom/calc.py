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
    wet_thresh,
    wet_spell_length,
    wet_spell_thresh,
    min_wet_days,
    dry_spell_length,
    dry_spell_search,
    time_coord="T",
):
    """Finds the first wet spell of wet_spell_length days
    where cumulative rain exceeds wet_spell_thresh,
    with at least min_wet_days count of wet days (greater than wet_thresh),
    not followed by a dry spell of dry_spell_length days of dry days (not wet),
    for the following dry_spell_search days
    returns the time delta rom the first day of daily_rain
    to the first wet day in that wet spell
    """
    # Find wet days
    wet_day = daily_rain > wet_thresh

    # Find 1st wet day in wet spells length
    first_wet_day = wet_day * 1
    first_wet_day = (
        first_wet_day.rolling(**{time_coord: wet_spell_length})
        .construct("wsl")
        .argmax("wsl")
    )

    # Find wet spells
    wet_spell = (
        daily_rain.rolling(**{time_coord: wet_spell_length}).sum() >= wet_spell_thresh
    ) & (wet_day.rolling(**{time_coord: wet_spell_length}).sum() >= min_wet_days)

    # Find dry spells following wet spells
    dry_day = ~wet_day
    dry_spell = (
        dry_day.rolling(**{time_coord: dry_spell_length}).sum() == dry_spell_length
    )
    # Note that rolling assigns to the last position of the wet_spell
    dry_spell_ahead = (
        dry_spell.rolling(**{time_coord: dry_spell_search})
        .sum()
        .shift(**{time_coord: dry_spell_search * -1})
        != 0
    )

    # Create a mask of 1s and nans where onset conditions are met
    onset_mask = (wet_spell & ~dry_spell_ahead) * 1
    onset_mask = onset_mask.where((onset_mask == 1))

    # Find onset date (or rather last day of 1st valid wet spell)
    # Note it doesn't matter to use idxmax or idxmin,
    # it finds the first max thus the first onset date since we have only 1s and nans
    # all nans returns nan
    onset_delta = onset_mask.idxmax(dim=time_coord)
    onset_delta = (
        onset_delta
        # offset relative position of first wet day
        # note it doesn't matter to apply max or min
        # per construction all values are nan but 1
        - (
            wet_spell_length
            - 1
            - first_wet_day.where(first_wet_day[time_coord] == onset_delta).max(
                dim=time_coord
            )
        ).astype("timedelta64[D]")
        # delta from 1st day of time series
        - daily_rain[time_coord][0]
    ).rename("onset_delta")
    return onset_delta


def run_test_onset_date():
    import pyaconf
    import os
    from pathlib import Path

    CONFIG = pyaconf.load(os.environ["CONFIG"])
    DR_PATH = CONFIG["daily_rainfall_path"]
    RR_MRG_ZARR = Path(DR_PATH)
    rr_mrg = read_zarr_data(RR_MRG_ZARR)
    rr_mrg = rr_mrg.sel(T=slice("2000-01-01", "2000-12-31"))
    print(
        onset_date(rr_mrg.precip, 1, 3, 20, 1, 7, 21, time_coord="T")
        .isel(X=150, Y=150)
        .values
    )
    print(
        (
            onset_date(rr_mrg.precip, 1, 3, 20, 1, 7, 21, time_coord="T")
            + onset_date(rr_mrg.precip, 1, 3, 20, 1, 7, 21, time_coord="T")["T"]
        )
        .isel(X=150, Y=150)
        .values
    )


# run_test_onset_date()

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
            lambda x: ((x + np.timedelta64(1, "D")).dt.day == 1)
            & ((x + np.timedelta64(1, "D")).dt.month == 3),
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


def seasonal_onset_date(
    daily_rain,
    search_start_day,
    search_start_month,
    search_days,
    wet_thresh,
    wet_spell_length,
    wet_spell_thresh,
    min_wet_days,
    dry_spell_length,
    dry_spell_search,
    time_coord="T",
):
    """Function reproducing Ingrid onsetDate function
    http://iridl.ldeo.columbia.edu/dochelp/Documentation/details/index.html?func=onsetDate
    combining a function that groups data by season
    and a function that search for an onset date
    """

    # Deal with leap year cases
    if search_start_day == 29 and search_start_month == 2:
        search_start_day = 1
        search_start_month = 3

    # Find an acceptable end_day/_month
    first_end_date = daily_rain[time_coord].where(
        lambda x: (x.dt.day == search_start_day) & (x.dt.month == search_start_month),
        drop=True,
    )[0] + np.timedelta64(
        search_days
        # search_start_day is part of the search
        - 1 + dry_spell_search
        # in case this first season covers a non-leap year 28 Feb
        # so that if leap years involve in the process, we have enough days
        # and if not, then we add 1 more day which should not cause trouble
        # unless that pushes us to a day that is not part of the data
        # that would make the whole season drop -- acceptable?
        + 1,
        "D",
    )

    end_day = first_end_date.dt.day.values

    end_month = first_end_date.dt.month.values

    # Apply daily grouping by season
    grouped_daily_data = daily_tobegroupedby_season(
        daily_rain, search_start_day, search_start_month, end_day, end_month
    )
    # Apply onset_date
    seasonal_data = (
        grouped_daily_data[daily_rain.name]
        .groupby(grouped_daily_data["seasons_starts"])
        .map(
            onset_date,
            wet_thresh=wet_thresh,
            wet_spell_length=wet_spell_length,
            wet_spell_thresh=wet_spell_thresh,
            min_wet_days=min_wet_days,
            dry_spell_length=dry_spell_length,
            dry_spell_search= dry_spell_search,
        )
        # This was not needed when applying sum
        .drop_vars(time_coord)
        .rename({"seasons_starts": time_coord})
    )
    # Get the seasons ends
    seasons_ends = grouped_daily_data["seasons_ends"].rename({"group": time_coord})
    seasonal_onset_date = xr.merge([seasonal_data, seasons_ends])

    # Tip to get dates from timedelta search_start_day
    # seasonal_onset_date = seasonal_onset_date[time_coord]
    # + seasonal_onset_date.onset_delta
    return seasonal_onset_date


def run_test_season_onset():
    import pyaconf
    import os
    from pathlib import Path

    CONFIG = pyaconf.load(os.environ["CONFIG"])
    DR_PATH = CONFIG["daily_rainfall_path"]
    RR_MRG_ZARR = Path(DR_PATH)
    rr_mrg = read_zarr_data(RR_MRG_ZARR)
    rr_mrg = rr_mrg.sel(T=slice("2000", "2005"))

    print("the output of onset date")
    print(
        seasonal_onset_date(rr_mrg.precip, 1, 3, 90, 1, 3, 20, 1, 7, 21, time_coord="T")
    )
    print(
        seasonal_onset_date(rr_mrg.precip, 1, 3, 90, 1, 3, 20, 1, 7, 21, time_coord="T")
        .onset_delta.isel(X=150, Y=150)
        .values
    )
    print(
        (
            seasonal_onset_date(
                rr_mrg.precip, 1, 3, 90, 1, 3, 20, 1, 7, 21, time_coord="T"
            ).onset_delta
            + seasonal_onset_date(
                rr_mrg.precip, 1, 3, 90, 1, 3, 20, 1, 7, 21, time_coord="T"
            )["T"]
        )
        .isel(X=150, Y=150)
        .values
    )


#run_test_season_onset()


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
        #        .rename({"seasons_starts": time_coord})
    )
    seasons_ends = grouped_daily_data["seasons_ends"].rename({"group": time_coord})
    summed_seasons = xr.merge([seasonal_data, seasons_ends])
    return summed_seasons

def probExceed(onsetMD, search_start):
    onsetDiff = onsetMD.onset - search_start
    onsetDiff_df = onsetDiff.to_frame()
    counts = onsetDiff_df['onset'].value_counts()
    countsDF = counts.to_frame().sort_index()
    cumsum = countsDF.cumsum()
    onset = onsetDiff_df.onset.dt.total_seconds() / (24 * 60 * 60)
    onset_unique = list(set(onset))
    cumsum['Days'] = onset_unique
    cumsum['probExceed'] = (1 - cumsum.onset / cumsum.onset[-1])
    return cumsum
    

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

    #    print("the inputs to grouping")
    #    print(daily_tobegroupedby_season(rr_mrg.precip, 29, 11, 29, 2))

    print("the outputs of seasonal_sum")
    print(seasonal_sum(rr_mrg.precip, 29, 11, 29, 2, min_count=0))


#    print("some data")
#    print(
#        seasonal_sum(rr_mrg.precip, 29, 11, 29, 2, min_count=0)
#        .precip.isel(X=150, Y=150)
#        .values
#    )
#    print(
#        rr_mrg.precip.sel(T=slice("2000-11-29", "2001-02-28"))
#        .sum("T")
#        .isel(X=150, Y=150)
#        .values
#    )
#    print(
#        rr_mrg.precip.sel(T=slice("2001-11-29", "2002-02-28"))
#        .sum("T")
#        .isel(X=150, Y=150)
#        .values
#    )
#    print(
#        rr_mrg.precip.sel(T=slice("2002-11-29", "2003-02-28"))
#        .sum("T")
#        .isel(X=150, Y=150)
#        .values
#    )
#    print(
#        rr_mrg.precip.sel(T=slice("2003-11-29", "2004-02-29"))
#        .sum("T")
#        .isel(X=150, Y=150)
#        .values
#    )


# run_test_season_stuff()
