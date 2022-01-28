#temporary file for cessation date functions; 
#these will be moved to the main calc.py file when finished

import calc
import numpy as np
import pandas as pd
import xarray as xr

#params for cess as found in maproom
search_start_day = 1
search_start_month = 9
search_days = 90
dry_thresh = 5
min_dry_days = 3
time_coord = "T"
et = 5
taw = 60
sminit = 0 

rr_mrg = calc.read_zarr_data("/data/aaron/ethiopia-rain-rechunked")

#subset of data for testing
rr_mrgSub = rr_mrg.sel(T=slice('1981-01-01','1983-12-31'))
daily_rain = rr_mrgSub.isel(X=200,Y=160)


#onset date calc used in seasonal cess date func
#first date after START_DAY(1), START_MONTH(9) in SEARCH_DAYS(90) 
#days when the soil water balance falls below DRY_THRESH(5)mm 
#for a period of DRY_SPELL_LENGTH(3) days
def cess_date(daily_rain, dry_thresh, min_dry_days, et, taw, sminit, time_coord="T"):
    water_balance = calc.water_balance(daily_rain, et,taw,sminit,"T")
    dry_day = water_balance < dry_thresh
    first_dry_day = dry_day * 1
    first_dry_day_roll = first_dry_day.rolling(**{time_coord: min_dry_days}).sum() >= min_dry_days
    cess_mask = first_dry_day_roll * 1
    cess_mask = cess_mask.where((cess_mask == 1))
    cess_delta = cess_mask.idxmax(dim=time_coord)
    cess_delta = cess_mask.idxmax(dim=time_coord)
    cess_delta = (
        cess_delta
    # offset relative position of first wet day
    # note it doesn't matter to apply max or min
    # per construction all values are nan but 1
        - (
            min_dry_days
            - 1
            - first_dry_day.where(first_dry_day[time_coord] == cess_delta).max(
                dim=time_coord
            )
        ).astype("timedelta64[D]")
        # delta from 1st day of time series
        - daily_rain[time_coord][0]
    ).rename({"soil_moisture":"cess_delta"})
    return cess_delta

#seasonal cessation date using cessation_date(), calc.water_balance(), calc.daily_tobegroupedby_season()
def seasonal_cess_date(
    search_start_day, 
    search_start_month,
    search_days,
    daily_rain, 
    dry_thresh, 
    min_dry_days, 
    et, 
    taw, 
    sminit, 
    time_coord="T"
):
    # Deal with leap year cases
    if search_start_day == 29 and search_start_month == 2:
        search_start_day = 1
        search_start_month = 3

    # Find an acceptable end_day/_month
    first_end_date = daily_rain[time_coord].where(
        lambda x: (x.dt.day == search_start_day) & (x.dt.month == search_start_month),
        drop=True,
    )[0] + np.timedelta64(
        search_days,
        "D",
    )

    end_day = first_end_date.dt.day.values

    end_month = first_end_date.dt.month.values

    # Apply daily grouping by season
    grouped_daily_data = calc.daily_tobegroupedby_season(
        daily_rain, search_start_day, search_start_month, end_day, end_month
    )
    # Apply onset_date
    seasonal_data = (
        grouped_daily_data["precip"]
        .groupby(grouped_daily_data["seasons_starts"])
        .map(
            cess_date,
            dry_thresh=dry_thresh,
            min_dry_days=min_dry_days,
            et=et,
            taw=taw,
            sminit=sminit,
        )
        # This was not needed when applying sum
        .drop_vars(time_coord)
        .rename({"seasons_starts": time_coord})
    )
    # Get the seasons ends
    seasons_ends = grouped_daily_data["seasons_ends"].rename({"group": time_coord})
    seasonal_cess_date = xr.merge([seasonal_data, seasons_ends])

    # Tip to get dates from timedelta search_start_day
    # seasonal_onset_date = seasonal_onset_date[time_coord]
    # + seasonal_onset_date.onset_delta
    return seasonal_cess_date

cess_dates = seasonal_cess_date(
    search_start_day, 
    search_start_month,
    search_days,
    daily_rain, 
    dry_thresh, 
    min_dry_days, 
    et, 
    taw, 
    sminit, 
    time_coord="T"
)
print(cess_dates)
