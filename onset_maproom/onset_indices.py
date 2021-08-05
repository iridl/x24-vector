import numpy as np


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
    rainy_days = rain > rainy_day
    wet_spells = (
        np.convolve(rain, np.ones(running_days, dtype=int), "valid") > running_total
    ) & (
        np.convolve(rainy_days, np.ones(running_days, dtype=int), "valid")
        > min_rainy_days
    )
    dry_spells = (
        np.convolve(~rainy_days, np.ones(dry_days, dtype=int), "valid") == dry_days
    )
    dry_spells_ahead = (
        np.convolve(dry_spells, np.ones(running_days + dry_spell, dtype=int), "valid")
        != 0
    )
    onset_ids = np.where((wet_spells[:n]) & (~dry_spells_ahead))[0]
    return onset_ids


xs = onset_indices(
    np.random.rand(1000),
    rainy_day=0.5,
    running_days=10,
    running_total=6.0,
    min_rainy_days=3,
    dry_days=4,
    dry_spell=20,
)
print(xs, xs.shape)
