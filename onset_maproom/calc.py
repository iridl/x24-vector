import numpy as np
import pandas as pd
import xarray as xr

# Date Reading functions
def read_zarr_data(zarr_path):
    zarr_data = xr.open_zarr(zarr_path)
    return zarr_data


# Growing season functions


def water_balance(
    daily_rain,
    et,
    taw,
    sminit,
    time_coord="T",
):
    """Estimates soil moisture from
    rainfall,
    evapotranspiration,
    total available water and
    intial soil moisture value, knowing that:
    sm(t) = sm(t-1) + rain(t) - et(t)
    with roof and floor respectively at taw and 0 at each time step.
    """
    # Get time_coord info
    time_coord_size = daily_rain[time_coord].size
    # Get all the rain-et deltas:
    delta_rain_et = daily_rain - et
    # Intializing sm
    soil_moisture = xr.DataArray(
        data=np.empty(daily_rain.shape),
        dims=daily_rain.dims,
        coords=daily_rain.coords,
        name="soil_moisture",
        attrs=dict(description="Soil Moisture", units="mm"),
    )
    soil_moisture[{time_coord: 0}] = (
        sminit + delta_rain_et.isel({time_coord: 0})
    ).clip(0, taw)
    # Looping on time_coord
    for i in range(1, time_coord_size):
        soil_moisture[{time_coord: i}] = (
            soil_moisture.isel({time_coord: i - 1})
            + delta_rain_et.isel({time_coord: i})
        ).clip(0, taw)
    water_balance = xr.Dataset().merge(soil_moisture)
    return water_balance


def longest_run_length(flagged_data, dim):
    """ Find the length of the longest run of flagged (0/1) data along a dimension.
    
    A run is a series of 1s not interrupted by 0s.
    The result is expressed in the units of `dim`, that is assumed evenly spaced.
    
    Parameters
    ----------
    flagged_data : DataArray
        Array of flagged data (0s or 1s)
    dim : str
        dimension of `flagged_data` along which to search for runs
        
    Returns
    -------
    DataArray
        Array of length of longest run along `dim`
        
    See Also
    --------
    
    Notes
    -----
    The longest run is the maximum value of the discrete difference
    of cumulative flags keeping only the unflagged data.
    Because diff needs at least 2 points,
    we need to keep (where) the unflagged and first and last
    with the cumulative value for last and 0 for first.
    Cumulative flags, where kept, need be propagated by bfill
    so that diff returns 0 or the length of runs.
    
    I believe that it works for unevenly spaced `dim`,
    only we then don't know what the units of the result are.
    
    Examples
    --------
    >>> t = pd.date_range(start="2000-05-01", end="2000-05-29", freq="1D")
    >>> values = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    ... 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
    >>> flags = xr.DataArray(values, dims=["T"], coords={"T": t})
    >>> longest_run_length(flags, "T")
    <xarray.DataArray ()>
    array(7.)
    Attributes: description:  Longest Run Length
    
    """
    
    # Special case coord.size = 1
    lrl = flagged_data
    if lrl[dim].size != 1:
        # Points to apply diff to
        unflagged_and_ends = (flagged_data == 0) * 1
        unflagged_and_ends[{dim: [0, -1]}] = 1
    
        lrl = lrl.cumsum(dim=dim).where(unflagged_and_ends, other = np.nan).where(
            # first cumul point must be set to 0
            lambda x: x[dim] != lrl[dim][0], other=0
        ).bfill(dim).diff(dim).max(dim=dim)
    lrl.attrs = dict(description="Longest Run Length")
    return lrl


def following_dry_spell_length(daily_rain, wet_thresh, time_coord="T"):
    """Compute the count of consecutive dry days (or dry spell length) after each day

    Parameters
    ----------
    daily_rain : DataArray
        Array of flagged daily rainfall
    wet_thresh : float
        a dry day is a day when `daily_rain` is lesser or equal to `wet_thresh`
    time_coord : str, optional             
        Daily time dimension of `daily_rain` (default `time_coord`="T").
 
    Returns
    -------
    DataArray
        Array of length of dry spell immediately following each day along `time_coord`
        
    See Also
    --------
    
    Notes
    -----
    Ideally we would want to cumulate count of dry days backwards
    and reset count to 0 each time a wet day occurs.
    But that is hard to do vectorially.
    But we can cumulatively count all dry days backwayds
    then apply an offset. In more details:
    Cumulate dry days backwards to get all dry days after a day
    Find when to apply new offset (dry days followed by a wet day)
    Assign cumulated dry days there, Nan elsewhere
    Propagate backwards and the 0s at the tail
    And that is going to be the offset
    Apply offset that is correct for all days followed by dry days
    Eventually reset days followed by wet days to 0

    Examples
    --------
    >>> t = pd.date_range(start="2000-05-01", end="2000-05-14", freq="1D")
    >>> values = [0.054383, 0., 0., 0.027983, 0., 0., 7.763758, 3.27952, 13.375934, 4.271866, 12.16503, 9.706059, 7.048605,  0.]
    >>> precip = xr.DataArray(values, dims=["T"], coords={"T": t})
    >>> following_dry_spell_length(precip, 1)
    <xarray.DataArray (T: 14)>
    array([5., 4., 3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
    Coordinates:
      * T        (T) datetime64[ns] 2000-05-01 2000-05-02 ... 2000-05-13 2000-05-14
    """

    # Find dry days
    dry_day = ~(daily_rain > wet_thresh) * 1
    # Cumul dry days backwards and shift back to get the count to exclude day of
    count_dry_days_after_today = dry_day.reindex({time_coord: dry_day[time_coord][::-1]}).cumsum(
        dim=time_coord
    ).reindex({time_coord: dry_day[time_coord]}).shift({time_coord: -1})
    # Find where dry day followed by wet day
    dry_to_wet_day = dry_day.diff(time_coord, label="lower").where(lambda x : x == -1, other=0)
    # Record cumul dry days on that day and put nan elsewhere
    dry_days_offset = (count_dry_days_after_today * dry_to_wet_day).where(lambda x : x != 0, other=np.nan)
    # Back fill nans and assign 0 to tailing ones
    dry_days_offset = dry_days_offset.bfill(dim=time_coord).fillna(0)
    # Subtract offset and shifted wet days are 0.
    dry_spell_length = (count_dry_days_after_today + dry_days_offset) * dry_day.shift({time_coord: -1})
    return dry_spell_length


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
    ) & ((wet_day*1).rolling(**{time_coord: wet_spell_length}).sum() >= min_wet_days)

    # Find dry spells following wet spells
    dry_day = ~wet_day
    dry_spell = (
        (dry_day*1).rolling(**{time_coord: dry_spell_length}).sum() == dry_spell_length
    )
    # Note that rolling assigns to the last position of the wet_spell
    dry_spell_ahead = (
        (dry_spell*1).rolling(**{time_coord: dry_spell_search})
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


def cess_date(
    soil_moisture, 
    dry_thresh, 
    min_dry_days, 
    time_coord="T"
):

    dry_day = soil_moisture < dry_thresh
    dry_spell = dry_day * 1
    dry_spell_roll = dry_spell.rolling(**{time_coord: min_dry_days}).sum() == min_dry_days
    cess_mask = dry_spell_roll * 1
    cess_mask = cess_mask.where((cess_mask == 1))
    cess_delta = cess_mask.idxmax(dim=time_coord)
    cess_delta = (
        cess_delta
        - np.timedelta64(
            min_dry_days - 1,"D"
            ) - soil_moisture[time_coord][0])
    return cess_delta

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
    """Function reproducing `Ingrid onsetDate function <http://iridl.ldeo.columbia.edu/dochelp/Documentation/details/index.html?func=onsetDate>`_
    combining a function that groups data by season
    and a function that search for an onset date

    Parameters
    ----------
    daily_rain : DataArray
                 Array of daily rainfall totals.
    search_start_day : int
                 The day part (1-31) of the date to start scanning for onset date.
    search_start_month : int
                 The month part (1-12) of the date to start scanning for onset date.
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
            dry_spell_search=dry_spell_search,
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

def seasonal_cess_date(
    soil_moisture,
    search_start_day,
    search_start_month,
    search_days,
    dry_thresh,
    min_dry_days,
    time_coord="T"
):
    # Deal with leap year cases
    if search_start_day == 29 and search_start_month == 2:
        search_start_day = 1
        search_start_month = 3

    # Find an acceptable end_day/_month
    first_end_date = soil_moisture[time_coord].where(
        lambda x: (x.dt.day == search_start_day) & (x.dt.month == search_start_month),
        drop=True,
    )[0] + np.timedelta64(
        search_days,
        "D",
    )

    end_day = first_end_date.dt.day.values

    end_month = first_end_date.dt.month.values

    # Apply daily grouping by season
    grouped_daily_data = daily_tobegroupedby_season(
        soil_moisture, search_start_day, search_start_month, end_day, end_month
    )
    # Apply onset_date
    seasonal_data = (
        grouped_daily_data[soil_moisture.name]
        .groupby(grouped_daily_data["seasons_starts"])
        .map(
            cess_date,
            dry_thresh=dry_thresh,
            min_dry_days=min_dry_days,
        )
        # This was not needed when applying sum
        .drop_vars(time_coord)
        .rename({"seasons_starts": time_coord})
    ).rename("cess_delta")
    # Get the seasons ends
    seasons_ends = grouped_daily_data["seasons_ends"].rename({"group": time_coord})
    seasonal_cess_date = xr.merge([seasonal_data, seasons_ends])

    # Tip to get dates from timedelta search_start_day
    # seasonal_onset_date = seasonal_onset_date[time_coord]
    # + seasonal_onset_date.onset_delta
    return seasonal_cess_date

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


def probExceed(dfMD, search_start):
    columName = dfMD.columns[0]
    Diff = dfMD[columName] - search_start
    Diff_df = Diff.to_frame()
    counts = Diff_df[columName].value_counts()
    countsDF = counts.to_frame().sort_index()
    cumsum = countsDF.cumsum()
    getTime = Diff_df[columName].dt.total_seconds() / (24 * 60 * 60)
    unique = list(set(getTime))
    unique = [x for x in unique if np.isnan(x) == False]
    cumsum["Days"] = unique
    cumsum["probExceed"] = 1 - cumsum[columName] / cumsum[columName][-1]
    return cumsum
