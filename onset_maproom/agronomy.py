import numpy as np
import pandas as pd
import xarray as xr
#from pint import application_registry as ureg
#import cf_xarray.units


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
    kc_params are the starting, ending and inflection points
    of the kc curve against the time deltas in days as coord
    This is how Kc data is most often provided
    Kc is 1 outside the growing season
    """
    # getting deltas from 0 rather than consecutive ones
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
    wb(t) = sm(t) + drainage(t) with sm roofed at taw
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
    drainage
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
            for thedim in water_balance.dims:
                if adim != thedim and adim != "kc_periods":
                    water_balance[adim] = kc_params[adim]
    if p_d is not None:
        for adim in p_d.dims:
            for thedim in water_balance.dims:
                if adim != thedim and adim != time_coord:
                    water_balance[adim] = p_d[adim]
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
    if time_coord in kc0.dims:
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
    # Get time_coord info
    time_coord_size = water_balance.peffective[time_coord].size
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
        if time_coord in kc.dims:
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
    # Recomputing Drainage
    drainage = (
        (
            water_balance.peffective
            - water_balance.et_crop_red
            - water_balance.soil_moisture.diff(time_coord)
        )
        .clip(min=0)
        .rename("drainage")
    )
    drainage.attrs = dict(description="Drainage", units="mm")
    water_balance = water_balance.merge(drainage)
    water_balance.drainage[{time_coord: 0}] = (
        water_balance.peffective.isel({time_coord: 0})
        - water_balance.et_crop_red.isel({time_coord: 0})
        - (water_balance.soil_moisture.isel({time_coord: 0}) - sminit)
    ).clip(min=0)

    return water_balance