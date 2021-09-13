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


def onset_index(
    data,
    data_flag,
    wlength,
    wthresh,
    flagged_thresh,
    fwlength,
    false_search,
    coord_index,
):
    """Finds the first window of size wlength where data exceeds wthresh,
    with at least flagged_thresh count of flagged data (by data_flag),
    not followed by a false_window of size fwlength of unflagged data,
    for the following false_search indices
    returns the index of first flagged data in that window
    (how much wood would a woodchuck chuck if a woodchuck could chuck wood?)
    """
    flagged = data > data_flag

    window = (data.rolling(**{coord_index: wlength}).sum() > wthresh) & (
        flagged.rolling(**{coord_index: wlength}).sum() > flagged_thresh
    )

    unflagged = ~flagged
    false_window = unflagged.rolling(**{coord_index: fwlength}).sum() == fwlength

    # Note that rolling assigns to the last position of the window
    false_window_ahead = (
        false_window.rolling(**{coord_index: false_search})
        .sum()
        .shift(**{coord_index: false_search * -1})
        != 0
    )

    flagged_within_onset = (flagged) & (
        (window & ~false_window_ahead)
        .rolling(**{coord_index: wlength})
        .sum()
        .shift(**{coord_index: 1 - wlength})
        != 0
    )

    # Turns False/True into nan/1
    onset_mask = flagged_within_onset * 1
    onset_mask = onset_mask.where(flagged_within_onset)

    # Note it doesn't matter to use idxmax or idxmin,
    # it finds the first max thus the first onset date since we have only 1s and nans
    # all nans returns nan
    onset_id = onset_mask.idxmax(dim=coord_index)
    return onset_id


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
    print(onset_index(rr_mrg.precip, 1, 3, 20, 1, 7, 21, "T"))


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
print(rain)
print(xs)
