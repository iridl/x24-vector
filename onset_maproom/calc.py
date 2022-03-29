import numpy as np
import pandas as pd
import xarray as xr
#from pint import application_registry as ureg
#import cf_xarray.units

# Data Reading functions


def read_zarr_data(zarr_path):
    zarr_data = xr.open_zarr(zarr_path)
    return zarr_data


# Water Balance functions


def api_sum(x, axis=None):
    """Weighted-sum for Antecedent Precipitation Index
    for an array of length n, applies weights of
    1/2 for last element
    1/(n-i-1) for all others
    """
    axis = axis[0]
    api_weights = np.arange(x.shape[axis] - 1, -1, -1)
    api_weights[-1] = 2
    api_weights = 1 / api_weights
    api = np.sum(x * api_weights, axis=axis)
    return api


def api_runoff_select(
    daily_rain,
    api,
    api_cat=[12.5, 6.3, 19, 31.7, 44.4, 57.1, 69.8],
    api_poly=np.array(
        [
            [0.858, 0.0895, 0.0028],
            [-1.14, 0.042, 0.0026],
            [-2.34, 0.12, 0.0026],
            [-2.36, 0.19, 0.0026],
            [-2.78, 0.25, 0.0026],
            [-3.17, 0.32, 0.0024],
            [-4.21, 0.438, 0.0018],
        ]
    ),
):
    """Computes Runoff using Antecedent Precipitation Index
    currently not in use other than for testing other method
    """
    func = lambda x, y: np.select(
        [
            x <= api_cat[0],
            y <= api_cat[1],
            y <= api_cat[2],
            y <= api_cat[3],
            y <= api_cat[4],
            y <= api_cat[5],
            y <= api_cat[6],
        ],
        [
            0,
            api_poly[0, 0] - api_poly[0, 1] * x + api_poly[0, 2] * np.square(x),
            api_poly[1, 0] + api_poly[1, 1] * x + api_poly[1, 2] * np.square(x),
            api_poly[2, 0] + api_poly[2, 1] * x + api_poly[2, 2] * np.square(x),
            api_poly[3, 0] + api_poly[3, 1] * x + api_poly[3, 2] * np.square(x),
            api_poly[4, 0] + api_poly[4, 1] * x + api_poly[4, 2] * np.square(x),
            api_poly[5, 0] + api_poly[5, 1] * x + api_poly[5, 2] * np.square(x),
        ],
        default=api_poly[6, 0] + api_poly[6, 1] * x + api_poly[6, 2] * np.square(x),
    )
    return xr.apply_ufunc(func, daily_rain, api)


def weekly_api_runoff(
    daily_rain,
    no_runoff=12.5,
    api_thresh=xr.DataArray([6.3, 19, 31.7, 44.4, 57.1, 69.8], dims=["api_cat"]),
    api_poly=xr.DataArray(
        [
            [0.858, 0.0895, 0.0028],
            [-1.14, 0.042, 0.0026],
            [-2.34, 0.12, 0.0026],
            [-2.36, 0.19, 0.0026],
            [-2.78, 0.25, 0.0026],
            [-3.17, 0.32, 0.0024],
            [-4.21, 0.438, 0.0018],
        ],
        dims=["api_cat", "powers"],
    ),
    time_coord="T",
):
    """Computes Runoff using Antecedent Precipitation Index
    api_runoff_select is a more human-reading friendly version
    runoff is a polynomial of daily_rain of order 2
    Polynomial is chosen based on API categories
    except runoff is 0 if it rains less or equal than no_runoff
    and negative runoff is 0
    """
    # Compute API
    api = daily_rain.rolling(
        **{time_coord:7}
    ).reduce(api_sum).dropna(time_coord)
    #    runoff = api_runoff_select(daily_rain, api).clip(min=0)
    # xr.dot of rain polynomial with categorical mask
    runoff = (
        xr.dot(
            # xr.dot of powers of rain with polynomial coeffs
            xr.dot(
                api_poly,
                xr.concat(
                    [
                        np.power(daily_rain, 0),
                        daily_rain,
                        np.square(daily_rain)
                    ],
                    dim="powers",
                # runoff is 0 if not enough rain
                ).where(daily_rain > no_runoff, 0),
            ),
            # Minimum API categories, last category is "not NaN"
            (xr.concat([api <= api_thresh, ~np.isnan(api)], dim="api_cat"))
            # All categories following a True are False
            .cumsum(dim="api_cat").where(lambda x: x <= 1, other=0),
            dims="api_cat",
        # runoff can not be negative
        ).clip(min=0).rename("runoff")
    )
    runoff.attrs = dict(description="Runoff", units="mm")
    return runoff


def scs_cn_runoff(daily_rain, cn):
    """Computes Runoff based on the
    Soil Conservation Service (SCS) Curve Number (CN) method,
    basic reference is here:
    https://engineering.purdue.edu/mapserve/LTHIA7/documentation/scs.htm
    so looks like cn could be function of space at some point
    I suspect cn can not be greater than 100
    """
    s_int = 25.4 * (1000 / cn - 10)
    runoff = (np.square((daily_rain - 0.2 * s_int).clip(min=0)) / (
        daily_rain + 0.8 * s_int
    )).rename("runoff")
    runoff.attrs = dict(description="Runoff", units="mm")
    return runoff


def solar_radiation(doy, lat):
    """Computes solar radiation for day of year and latitude in radians"""
    distance_relative = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    solar_declination = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    sunset_hour_angle = np.arccos(-1 * np.tan(lat) * np.tan(solar_declination))
    ra = (
        24 * 60 * 0.082 * distance_relative
        * (
            sunset_hour_angle * np.sin(lat) * np.sin(solar_declination)
            + np.sin(sunset_hour_angle) * np.cos(lat) * np.cos(solar_declination)
        )
        / np.pi
    ).rename("ra")
    ra.attrs = dict(description="Extraterrestrial Radiation", units="MJ/m**2/day")
    return ra


def hargreaves_et_ref(temp_avg, temp_amp, ra):
    """Computes Reference Evapotranspiration as a function of
    temperature (average and amplitude in Celsius) and solar radation
    """
    # the Hargreaves coefficient.
    ah = 0.0023
    # the value of 0.408 is
    # the inverse of the latent heat flux of vaporization at 20C,
    # changing the extraterrestrial radiation units from MJ m−2 day−1
    # into mm day−1 of evaporation equivalent
    bh = 0.408
    et_ref = (
        ah * (temp_avg + 17.8) * np.sqrt(temp_amp) * bh * ra
    ).rename("et_ref")
    et_ref.attrs = dict(description="Reference Evapotranspiration", units="mm")
    return et_ref


def planting_date(soil_moisture, sm_threshold, time_coord="T"):
    """Planting Date is the day after 1st date when
    soil_moisture reaches sm_threshold
    """
    wet_day = soil_moisture >= sm_threshold
    planting_mask = wet_day * 1
    planting_mask = planting_mask.where((planting_mask == 1))
    planting_delta = planting_mask.idxmax(dim=time_coord)
    planting_delta = (
        planting_delta + np.timedelta64(1, "D") - soil_moisture[time_coord][0]
    ).rename("planting_delta")
    planting_delta.attrs = dict(description="Planting Date")
    return planting_delta


def kc_interpolation(planting_date, kc_params, time_coord="T"):
    """Interpolates Crop Cultivar values against time_coord
    from Planting Date and according to Kc deltas
    kc_params are the starting, ending and inflexion points
    of the kc curve against the time deltas in days as coord
    This is how Kc data is most often provided
    Kc is 1 outside the growing season
    """
    # getting deltas from 0 rather than consecituve ones
    kc = kc_params.assign_coords(
        kc_periods=kc_params["kc_periods"].cumsum(dim="kc_periods")
    )
    # case all planting_date are NaT
    if np.isnat(planting_date.min(skipna=True)):
        kc = xr.ones_like(
            planting_date.isel({time_coord: 0}, drop=True), dtype="float64"
        ).rename("kc")
    else:
        # get the dates where Kc values must occur
        kc_time = (
            (planting_date[time_coord] + planting_date + kc["kc_periods"])
            .drop_vars(time_coord)
            .squeeze(time_coord)
        )
        # create the 1D time grid that will be used for output
        kc_time_1d = pd.date_range(
            start=kc_time.min(skipna=True).values,
            end=kc_time.max(skipna=True).values,
            freq="1D",
        )
        kc_time_1d = xr.DataArray(kc_time_1d, coords=[kc_time_1d], dims=[time_coord])
        # assingn Kc values on the 1D time grid,
        # get rid of kc_periods and interpolate
        kc = (
            kc.where(kc_time_1d == kc_time)
            # by design there is a value or all NaN
            .sum("kc_periods", skipna=True, min_count=1)
            .interpolate_na(dim=time_coord)
        ).fillna(1).rename("kc")
    kc.attrs = dict(description="Crop Cultivars")
    return kc


def crop_evapotranspiration(et_ref, kc, time_coord="T"):
    """Computes Crop Evapotranspiration
    from Reference Evapotransipiration and Crop Cultivars
    Kc is typically defined for T when crop grows
    But we may want to run the water balance before and after
    so Kc=1 everywhere else on et_ref's T
    """
    kc = kc * xr.ones_like(et_ref)
    kc, et_ref_aligned = xr.align(kc, et_ref, join="outer")
    kc = kc.fillna(1)
    et_crop = (et_ref * kc).rename("et_crop")
    et_crop.attrs = dict(description="Crop Evapotranspiration", units="mm")
    return et_crop


def calibrate_available_water(taw, rho):
    """Scales Total Available Water to Readily Available Water
    Warning: rho can be a function of et_crop!!
    and thus depend on time, space, crop...
    """
    # Case where both inputs are constants
    if np.size(taw) == 1 and np.size(rho) == 1:
        raw = xr.DataArray(np.ones(1) * taw * rho).squeeze("dim_0").rename("raw")
    else:
        raw = (rho * taw).rename("raw")
    raw.attrs = dict(description="Readily Available Water", units="mm")
    return raw


def single_stress_coeff(soil_moisture, taw, raw, time_coord="T"):
    """Used to adjust ETcrop under soil water stress condition
    (using single coefficient approach of FAO56)
    Refer figure 42 from FAO56 page 167
    This is where it gets tricky because that one depends on SM(t-1)
    """
    ks = (
        soil_moisture.assign_coords(
            {time_coord: soil_moisture[time_coord] + np.timedelta64(1, "D")}
        )
        / (taw - raw)
    ).clip(max=1).rename("ks")
    ks.attrs = dict(description="Ks")
    return ks


def reduce_crop_evapotranspiration(et_crop, ks):
    """Scales to actual Crop Evapotranspiration
    """
    et_crop_red = (ks * et_crop).rename("et_crop_red")
    et_crop_red.attrs = dict(description="Reduced Crop Evapotranspiration", units="mm")
    return et_crop_red


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


def soil_plant_water_balance(
    daily_rain,
    et,
    taw,
    sminit,
    runoff=None,
    kc_params=None,
    p_d=None,
    sm_threshold=20,
    rho=None,
    time_coord="T",
):
    """Simulates soil, plant, water balance from
    rainfall,
    evapotranspiration,
    total available water and
    intial soil moisture value, knowing that:
    wb(t) = wb(t-1) + rain(t) - et(t)
    and:
    wb(t) = sm(t) + drain(t) with sm roofed at taw
    Then, some options can be triggered:
    if runoff is provided, then
    rain becomes daily_rain - runoff (and is called peffective)
    The simulation becomes crop dependent if kc_params are given
    in which case evapotranspiration is scaled by Kc
    If not, Kc is 1
    If crop dependent, crops need be planted at a planting date
    If planting date (p_d) is not given,
    it is estimated by planting_date function (dependent on sm
    and on sm_threshold that is set by default at 20 mm)
    Finally, an additional stress scale factor, Ks,
    can be triggered by definining rho
    then Ks is estimated by single_stress_coeff
    Then the outputs of the soil plant water balance are:
    soil moisture,
    runoff,
    effective precipitation,
    evapotranspiration (input) broadcasted against peffective,
    planting date (if it exists),
    reduced (or actual) evapotranspiration,
    drain
    """
    # Start water balance ds with runoff
    if runoff is None:
        runoff = xr.zeros_like(daily_rain).rename("runoff")
        runoff.attrs = dict(description="Runoff", units="mm")
    water_balance = xr.Dataset().merge(runoff)
    # Compute Effective Precipitation
    peffective = (daily_rain - water_balance.runoff).rename("peffective")
    peffective.attrs = dict(description="Effective Precipitation", units="mm")
    water_balance = water_balance.merge(peffective)
    # Additional vars and coords
    et = (et * xr.ones_like(water_balance.peffective)).rename("et")
    water_balance = water_balance.merge(et)
    if kc_params is not None:
        for adim in kc_params.dims:
            for thedims in water_balance.dims:
                if adim != thedims and adim != "kc_periods":
                    water_balance[adim] = kc_params[adim]
    if p_d is not None:
        for adim in p_d.dims:
            for thedims in water_balance.dims:
                if adim != thedims and adim != time_coord:
                    water_balance[adim] = p_d[adim]
    # Get time_coord info
    time_coord_size = water_balance.peffective[time_coord].size
    # Intializing sm, et_crop and et_crop_red
    # Creating variables and broadcasting everybody
    soil_moisture = xr.full_like(
        water_balance.peffective, np.nan
    ).rename("soil_moisture")
    soil_moisture.attrs = dict(description="Soil Moisture", units="mm")
    water_balance = water_balance.merge(soil_moisture)
    et_crop = xr.full_like(
        water_balance.peffective, np.nan
    ).rename("et_crop")
    et_crop.attrs = dict(description="Crop Evapotranspiration", units="mm")
    water_balance = water_balance.merge(et_crop)
    (water_balance,) = xr.broadcast(water_balance)
    water_balance["soil_moisture"] = water_balance.soil_moisture.copy()
    water_balance["et_crop"] = water_balance.et_crop.copy()
    # Give time dimension to sminit
    t0 = water_balance["soil_moisture"][time_coord][0] - np.timedelta64(1, 'D')
    sminit0 = xr.DataArray(coords=dict({time_coord: [t0.values]}), data=[sminit])
    # Initialize Ks
    ks = 1
    if rho is not None:
        raw = calibrate_available_water(taw, rho)
        ks = single_stress_coeff(
            sminit0,
            taw,
            raw.isel({time_coord: 0}, missing_dims="ignore"),
            time_coord=time_coord,
        ).squeeze(time_coord)
    # Create or Initialize Kc
    if kc_params is None:
        kc0 = xr.ones_like(water_balance.et)
    else:
        if p_d is not None:
            kc0 = kc_interpolation(p_d, kc_params, time_coord=time_coord)
        else:
            p_d_find = planting_date(
                sminit0, sm_threshold, time_coord=time_coord
            ).expand_dims(dim=time_coord)
            kc0 = kc_interpolation(p_d_find, kc_params, time_coord=time_coord)
    # Initializaing sm
    if time_coord in kc0:
        kc0 = kc0.sel({time_coord: water_balance.soil_moisture[time_coord][0]})
    water_balance.et_crop[{time_coord: 0}] = crop_evapotranspiration(
        water_balance.et.isel({time_coord: 0}).expand_dims(dim=time_coord),
        kc0,
        time_coord=time_coord,
    ).squeeze(time_coord)
    water_balance.soil_moisture[{time_coord: 0}] = (
        sminit
        + water_balance.peffective.isel({time_coord: 0})
        - reduce_crop_evapotranspiration(
            water_balance.et_crop
            .isel({time_coord: 0}).expand_dims(dim=time_coord),
            ks,
        ).squeeze(time_coord)
    ).clip(0, taw)
    # Looping on time_coord
    for i in range(1, time_coord_size):
        if rho is not None:
            ks = single_stress_coeff(
                water_balance.soil_moisture
                .isel({time_coord: i - 1}).expand_dims(dim=time_coord),
                taw,
                raw,
                time_coord=time_coord,
            ).squeeze(time_coord)
        if kc_params is not None and p_d is None:
            p_d_find = planting_date(
                xr.concat(
                    [
                        sminit0,
                        water_balance.soil_moisture
                        .isel({time_coord: slice(0, i - 1)}),
                    ],
                    time_coord,
                ),
                sm_threshold,
                time_coord=time_coord,
            ).expand_dims(dim=time_coord)
            kc = kc_interpolation(p_d_find, kc_params, time_coord=time_coord)
        else:
            kc = kc0
        if time_coord in kc:
            kc = kc.sel({time_coord: water_balance.soil_moisture[time_coord][i]})
        water_balance.et_crop[{time_coord: i}] = crop_evapotranspiration(
            water_balance.et
            .isel({time_coord: i}).expand_dims(dim=time_coord),
            kc,
            time_coord=time_coord,
        ).squeeze(time_coord)
        water_balance.soil_moisture[{time_coord: i}] = (
            water_balance.soil_moisture.isel({time_coord: i - 1}, drop=True)
            + water_balance.peffective.isel({time_coord: i}, drop=True)
            - reduce_crop_evapotranspiration(
                water_balance.et_crop
                .isel({time_coord: i}).expand_dims(dim=time_coord),
                ks,
            ).squeeze(time_coord)
        ).clip(0, taw)
    # Save planting date if computed
    if kc_params is not None and p_d is None:
        water_balance = water_balance.merge(
            p_d_find.rename({time_coord: time_coord + "_p_d"}).rename("p_d")
        )
    # Recomputing reduced ET crop
    if rho is not None:
        ks = single_stress_coeff(
            water_balance.soil_moisture, taw, raw, time_coord=time_coord
        )
    et_crop_red = reduce_crop_evapotranspiration(
        water_balance.et_crop,
        ks
    )
    water_balance = water_balance.merge(et_crop_red)
    if rho is not None:
        ks = single_stress_coeff(
            sminit0, taw, raw, time_coord=time_coord
        ).squeeze(time_coord)
    water_balance.et_crop_red[{time_coord: 0}] = reduce_crop_evapotranspiration(
        water_balance.et_crop.isel({time_coord: 0}),
        ks
    )
    # Recomputing Drain
    drain = (
        (
            water_balance.peffective
            - water_balance.et_crop_red
            - water_balance.soil_moisture.diff(time_coord)
        )
        .clip(min=0)
        .rename("drain")
    )
    drain.attrs = dict(description="Drain", units="mm")
    water_balance = water_balance.merge(drain)
    water_balance.drain[{time_coord: 0}] = (
        water_balance.peffective.isel({time_coord: 0})
        - water_balance.et_crop_red.isel({time_coord: 0})
        - (water_balance.soil_moisture.isel({time_coord: 0}) - sminit)
    ).clip(min=0)

    return water_balance


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
    counts = onsetDiff_df["onset"].value_counts()
    countsDF = counts.to_frame().sort_index()
    cumsum = countsDF.cumsum()
    onset = onsetDiff_df.onset.dt.total_seconds() / (24 * 60 * 60)
    onset_unique = list(set(onset))
    onset_unique = [x for x in onset_unique if np.isnan(x) == False]
    cumsum["Days"] = onset_unique
    cumsum["probExceed"] = 1 - cumsum.onset / cumsum.onset[-1]
    return cumsum
