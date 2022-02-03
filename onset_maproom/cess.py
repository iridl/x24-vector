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
rr_mrgSub = rr_mrg.sel(T=slice('1983-01-01','1986-12-31'))
daily_rain = rr_mrgSub.sel(X="37.75",Y="9", method="nearest", tolerance=0.04)


#onset date calc used in seasonal cess date func
#first date after START_DAY(1), START_MONTH(9) in SEARCH_DAYS(90) 
#days when the soil water balance falls below DRY_THRESH(5)mm 
#for a period of DRY_SPELL_LENGTH(3) days
def cess_date(water_balance, dry_thresh, min_dry_days, et, taw, sminit, time_coord="T"):
    dry_day = water_balance < dry_thresh
    dry_spell = dry_day * 1
    dry_spell_roll = dry_spell.rolling(**{time_coord: min_dry_days}).sum() == min_dry_days
    cess_mask = dry_spell_roll * 1
    cess_mask = cess_mask.where((cess_mask == 1))
    cess_delta = cess_mask.idxmax(dim=time_coord)
    cess_delta = (
        cess_delta
        - (
            min_dry_days
            #used this to subtract 1 from min_dry_days bc was having difficulty converting
            #int/float value to timedelta.. perhaps not permanent but a workaround for now
            #since the value will always be 1 anyways
            - dry_spell.where(dry_spell[time_coord] == cess_delta).max( 
                dim=time_coord
            )
        ).astype("timedelta64[D]")
        # delta from 1st day of time series
    - water_balance[time_coord][0])
    #cess_delta = (cess_delta-(min_dry_days) - water_balance[time_coord][0]
    #)#.rename({"precip":"cess_delta"})
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
    
    daily_rain = daily_rain.to_array()
    water_balance = calc.water_balance(daily_rain, et,taw,sminit,"T")
    # Apply daily grouping by season
    grouped_daily_data = calc.daily_tobegroupedby_season(
        water_balance, search_start_day, search_start_month, end_day, end_month
    )
    # Apply onset_date
    
    seasonal_data = (
        grouped_daily_data["soil_moisture"]
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
    seasonal_cess_date = xr.merge([seasonal_data.to_dataset(name='cess_delta'), seasons_ends])

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
