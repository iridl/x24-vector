import numpy as np
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
    ) & (
        np.convolve(rainy, np.ones(running_days, dtype=int), "valid")
        > min_rainy_days
    )
    dry_spells = (
        np.convolve(~rainy, np.ones(dry_days, dtype=int), "valid") == dry_days
    )
    dry_spells_ahead = (
        np.convolve(dry_spells, np.ones(running_days + dry_spell, dtype=int), "valid")
        != 0
    )
    onset_ids = np.where((wet_spells[:n]) & (~dry_spells_ahead))[0]
    onset_ids = np.unique(first_rainy[onset_ids] + onset_ids)
    return onset_ids


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
