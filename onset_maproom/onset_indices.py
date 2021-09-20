import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

np.random.seed(0)


def onset_indices(
    rain, rainy_day, running_days, running_total, min_rainy_days, dry_days, dry_spell
):
    n = rain.shape[0]
    rain = np.pad(
        rain,
        (0, running_days + dry_spell + dry_days - 2),
        mode="constant",
        constant_values=0,
    )
    rainy = rain > rainy_day
    first_rainy = np.apply_along_axis(
        lambda v: np.where(v)[0][0] if len(np.where(v)[0]) != 0 else running_days,
        1,
        sliding_window_view(rainy, window_shape=running_days),
    )
    wet_spells = (
        np.convolve(rain, np.ones(running_days, dtype=int), "valid") > running_total
    ) & (np.convolve(rainy, np.ones(running_days, dtype=int), "valid") > min_rainy_days)
    dry_spells = np.convolve(~rainy, np.ones(dry_days, dtype=int), "valid") == dry_days
    dry_spells_ahead = (
        np.convolve(dry_spells, np.ones(running_days + dry_spell, dtype=int), "valid")
        != 0
    )
    onset_ids = np.where((wet_spells[:n]) & (~dry_spells_ahead))[0]
    onset_ids = np.unique(first_rainy[onset_ids] + onset_ids)
    return onset_ids


def onset_date(
    daily_rain,
    wet_th,
    wet_spell_length,
    wet_spell_th,
    min_wet_days,
    dry_spell_length,
    dry_spell_search,
    time_coord="T",
):
    """Finds the first wet spell of wet_spell_length days
    where cumulative rain exceeds wet_spell_th,
    with at least min_wet_days count of wet days (greater than wet_th),
    not followed by a dry spell of dry_spell_length days of dry days (not wet),
    for the following dry_spell_search days
    returns the time delta of first wet day in that wet spell
    the time delta from first day of daily rain
    """
    # Find wet days
    wet_day = daily_rain > wet_th

    # Find 1st wet day in wet spells length
    first_wet_day = wet_day * 1
    first_wet_day = (
        first_wet_day.rolling(**{time_coord: wet_spell_length})
        .construct("wsl")
        .argmax("wsl")
    )

    # Find wet spells
    wet_spell = (
        daily_rain.rolling(**{time_coord: wet_spell_length}).sum() >= wet_spell_th
    ) & (wet_day.rolling(**{time_coord: wet_spell_length}).sum() >= min_wet_days)

    # Find dry spells following wet spells
    dry_day = ~wet_day
    false_start = (
        dry_day.rolling(**{time_coord: dry_spell_length}).sum() == dry_spell_length
    )
    # Note that rolling assigns to the last position of the wet_spell
    false_start_ahead = (
        false_start.rolling(**{time_coord: dry_spell_search})
        .sum()
        .shift(**{time_coord: dry_spell_search * -1})
        != 0
    )

    # Create a mask of 1s and nans where onset conditions are met
    # Turns False/True into nan/1
    onset_mask = (wet_spell & ~false_start_ahead) * 1
    onset_mask = onset_mask.where((wet_spell & ~false_start_ahead))

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


def run_test():
    import pyaconf
    import os
    from pathlib import Path
    import calc

    CONFIG = pyaconf.load(os.environ["CONFIG"])
    DR_PATH = CONFIG["daily_rainfall_path"]
    RR_MRG_ZARR = Path(DR_PATH)
    rr_mrg = calc.read_zarr_data(RR_MRG_ZARR)
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


run_test()

rain = np.random.rand(1000)
xs = onset_indices(
    rain,
    rainy_day=0.5,
    running_days=10,
    running_total=6.0,
    min_rainy_days=3,
    dry_days=4,
    dry_spell=20,
)
# print(rain)
# print(xs)
