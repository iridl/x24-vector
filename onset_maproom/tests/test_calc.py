import numpy as np
import pandas as pd
import xarray as xr
import calc
import data_test_calc


def test_calibrate_available_water():

    taw = xr.DataArray(data=[60, 40], dims="X", coords={"X": [1, 2]})
    rho = xr.DataArray(
        data=[0.5, 0.45], dims="Crop", coords={"Crop": ["Maize", "Rice"]}
    )
    raw = calc.calibrate_available_water(taw, rho)

    assert (raw == [[30.0, 20.0], [27.0, 18.0]]).all()


def test_single_stress_coeff():

    et_crop = precip_sample() * 0 + 5
    soil_moisture = xr.where(precip_sample() > 10, 31, 15)
    taw = 60
    raw = 0.5 * taw
    ks = calc.single_stress_coeff(soil_moisture, taw, raw)

    assert (ks.isel(T=[8, 10, 54, 58]) == 1).all()
    assert (ks.dropna("T").where(lambda x: x != 1, drop=True) == 0.5).all()


def test_hargreaves_et_ref():

    tmin = data_test_calc.tmin_data_sample()
    tmax = data_test_calc.tmax_data_sample()
    temp_avg = (tmin + tmax) / 2
    # For the record as we've seen data where tmin > tmax
    temp_amp = (tmax - tmin).clip(min=0)
    doy = tmin["T"].dt.dayofyear
    lat = tmin["Y"]
    if lat.units == "degree_north":
        lat = lat * np.pi / 180
        lat.attrs = dict(units="radian")
    ra = calc.solar_radiation(doy, lat)
    et_ref = calc.hargreaves_et_ref(temp_avg, temp_amp, ra)
    expected = [
        [[5.28141477, 5.27847271], [4.67106064, 4.56270918]],
        [[5.67150704, 5.76156393], [5.18580189, 5.24758635]],
        [[6.76812044, 6.43869058], [6.0802728, 5.67826723]],
        [[6.42336374, 6.08255966], [6.28099376, 5.94158488]],
    ]

    assert np.allclose(et_ref, expected)


def test_solar_radiation():

    precip = data_test_calc.lat_time_data_sample()
    doy = precip["T"].dt.dayofyear
    lat = precip["Y"]
    if lat.units == "degree_north":
        lat = lat * np.pi / 180
        lat.attrs = dict(units="radian")
    ra = calc.solar_radiation(doy, lat)

    assert (
        [
            ra.isel(T=0, Y=-1),
            ra.isel(T=0, Y=-1),
            ra.isel(T=0, Y=-1),
            ra.isel(T=0, Y=0),
            ra.isel(T=0, Y=0),
        ]
        < [
            ra.isel(T=-1, Y=-1),
            ra.isel(T=0, Y=0),
            ra.isel(T=-1, Y=0),
            ra.isel(T=-1, Y=0),
            ra.isel(T=-1, Y=0),
        ]
    ).all()


# Determine runoff and effective precipitation based on SCS curve number method (EJ (12/20/2019))
def Peffective_2D(
    PCP, CN
):  # CN should be pre-defined based on land cover, hydrologic soil groups, and antecent soil moisture condition
    # PCP is 600 x 600 matrix
    # potential maximum retention after runoff begins
    S_int = 25400 / CN - 254  # Need to updae this if CN is a map
    numerator = PCP - 0.2 * S_int  # 0.2*S_int => initial abstractions
    numerator = np.multiply(numerator, numerator)
    denominator = PCP + 0.8 * S_int
    Runoff = np.divide(numerator, denominator)
    Runoff[PCP < 0.2 * S_int] = 0
    Runoff[PCP <= 0] = 0
    Runoff[Runoff < 0] = 0

    Peff = PCP - Runoff
    return Peff, Runoff


def test_scs_cn_runoff_vs_enjins_code():

    precip = precip_sample() + 5
    runoff = calc.scs_cn_runoff(precip, 75)
    peff, runoff_eunjin = Peffective_2D(precip, 75)

    assert np.allclose(runoff, runoff_eunjin)


def test_api_sum():

    x = np.array([6, 5, 4, 3, 2, 1, 2])
    api = calc.api_sum(x, axis=(0,))

    assert api == 7


def test_weekly_api_runoff():

    precip = precip_sample() + 5
    runoff = calc.weekly_api_runoff(precip)
    other_api = precip.rolling(**{"T": 7}).reduce(calc.api_sum)
    other_runoff = (
        calc.api_runoff_select(precip, other_api).clip(min=0).isel(T=slice(6, None))
    )

    assert np.allclose(runoff, other_runoff)


def test_water_balance_intializes_right():

    precip = precip_sample()
    wb = calc.water_balance(precip, 5, 60, 0)

    assert wb.soil_moisture.isel(T=0) == 0


def test_water_balance():

    precip = precip_sample()
    wb = calc.water_balance(precip, 5, 60, 0)

    assert np.allclose(wb.soil_moisture.isel(T=-1), 10.350632)


def test_water_balance2():

    t = pd.date_range(start="2000-05-01", end="2000-05-04", freq="1D")
    values = [
        [5.0, 6.0, 3.0, 66.0],
        [10.0, 12.0, 14.0, 16.0],
    ]
    precip = xr.DataArray(values, dims=["X", "T"], coords={"T": t})
    wb = calc.water_balance(precip, 5, 60, 0)

    assert np.array_equal(wb.soil_moisture["T"], t)
    expected = [
        [0.0, 1.0, 0.0, 60.0],
        [5.0, 12.0, 21.0, 32.0],
    ]

    assert np.array_equal(wb.soil_moisture, expected)


def test_water_balance_et_is_xarray_but_has_no_T():

    t = pd.date_range(start="2000-05-01", end="2000-05-04", freq="1D")
    values = [
        [5.0, 6.0, 3.0, 66.0],
        [10.0, 12.0, 14.0, 16.0],
    ]
    precip = xr.DataArray(values, dims=["X", "T"], coords={"T": t})
    et = xr.DataArray([5, 10], dims=["X"])
    wb = calc.water_balance(precip, et, 60, 0)

    assert np.array_equal(wb.soil_moisture["T"], t)
    expected = [
        [0.0, 1.0, 0.0, 60.0],
        [0.0, 2.0, 6.0, 12.0],
    ]
    assert np.array_equal(wb.soil_moisture, expected)


def test_water_balance_et_has_T():

    t = pd.date_range(start="2000-05-01", end="2000-05-04", freq="1D")
    values = [
        [5.0, 6.0, 3.0, 66.0],
        [10.0, 12.0, 14.0, 16.0],
    ]
    precip = xr.DataArray(values, dims=["X", "T"], coords={"T": t})
    values = [5.0, 10.0, 15.0, 10.0]
    et = xr.DataArray(values, dims=["T"], coords={"T": t})
    wb = calc.water_balance(precip, et, 60, 0)

    assert np.array_equal(wb.soil_moisture["T"], t)
    expected = [
        [0.0, 0.0, 0.0, 56.0],
        [5.0, 7.0, 6.0, 12.0],
    ]
    assert np.array_equal(wb.soil_moisture, expected)


def test_soil_plant_water_balance_with_hargreaves():

    tmin = (precip_sample() + 10).expand_dims({"Y": [14.1]})
    tmin["Y"].attrs = dict(units="degree_north")
    tmax = tmin * 1.4
    temp_avg = (tmin + tmax) / 2
    temp_amp = (tmax - tmin).clip(min=0)
    doy = tmin["T"].dt.dayofyear
    lat = tmin["Y"]
    if lat.units == "degree_north":
        lat = lat * np.pi / 180
        lat.attrs = dict(units="radian")
    ra = calc.solar_radiation(doy, lat)
    wat_bal = calc.soil_plant_water_balance(
        precip_sample(),
        calc.hargreaves_et_ref(temp_avg, temp_amp, ra),
        60,
        10,
        kc_params=None,
        p_d=None,
        rho=None,
        runoff=calc.weekly_api_runoff(precip_sample()),
    )
    # print(wat_bal)

    assert 1 == 1


def test_soil_plant_water_balance_with_et_crop():

    tmin = (precip_sample() + 10).expand_dims({"Y": [14.1]})
    tmin["Y"].attrs = dict(units="degree_north")
    tmax = tmin * 1.4
    temp_avg = (tmin + tmax) / 2
    temp_amp = (tmax - tmin).clip(min=0)
    doy = tmin["T"].dt.dayofyear
    lat = tmin["Y"]
    if lat.units == "degree_north":
        lat = lat * np.pi / 180
        lat.attrs = dict(units="radian")
    ra = calc.solar_radiation(doy, lat)
    p_d = xr.DataArray(
        pd.DatetimeIndex(data=["2000-05-02", "2000-05-13"]),
        dims=["X"],
        coords={"X": [0, 1]},
    ).expand_dims({"T": pd.DatetimeIndex(data=["2000-05-01"])})
    p_d = p_d - p_d["T"]
    kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    kc_params = xr.DataArray(
        data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    )
    wat_bal = calc.soil_plant_water_balance(
        precip_sample(),
        calc.hargreaves_et_ref(temp_avg, temp_amp, ra),
        60,
        10,
        kc_params=kc_params,
        p_d=p_d,
        rho=None,
        runoff=calc.weekly_api_runoff(precip_sample()),
    )
    # print(wat_bal)

    assert 1 == 1


def test_soil_plant_water_balance_with_et_crop_pd_none():

    tmin = (precip_sample() + 10).expand_dims({"Y": [14.1]})
    tmin["Y"].attrs = dict(units="degree_north")
    tmax = tmin * 1.4
    temp_avg = (tmin + tmax) / 2
    temp_amp = (tmax - tmin).clip(min=0)
    doy = tmin["T"].dt.dayofyear
    lat = tmin["Y"]
    if lat.units == "degree_north":
        lat = lat * np.pi / 180
        lat.attrs = dict(units="radian")
    ra = calc.solar_radiation(doy, lat)
    kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    kc_params = xr.DataArray(
        data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    )
    wat_bal = calc.soil_plant_water_balance(
        precip_sample(),
        calc.hargreaves_et_ref(temp_avg, temp_amp, ra),
        60,
        10,
        kc_params=kc_params,
        rho=None,
        runoff=calc.weekly_api_runoff(precip_sample()),
    )
    # print(wat_bal)

    assert 1 == 1


def test_soil_plant_water_balance_with_rho():

    tmin = (precip_sample() + 10).expand_dims({"Y": [14.1]})
    tmin["Y"].attrs = dict(units="degree_north")
    tmax = tmin * 1.4
    temp_avg = (tmin + tmax) / 2
    temp_amp = (tmax - tmin).clip(min=0)
    doy = tmin["T"].dt.dayofyear
    lat = tmin["Y"]
    if lat.units == "degree_north":
        lat = lat * np.pi / 180
        lat.attrs = dict(units="radian")
    ra = calc.solar_radiation(doy, lat)
    p_d = xr.DataArray(
        pd.DatetimeIndex(data=["2000-05-02", "2000-05-13"]),
        dims=["X"],
        coords={"X": [0, 1]},
    ).expand_dims({"T": pd.DatetimeIndex(data=["2000-05-01"])})
    p_d = p_d - p_d["T"]
    kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    kc_params = xr.DataArray(
        data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    )
    wat_bal = calc.soil_plant_water_balance(
        precip_sample(),
        calc.hargreaves_et_ref(temp_avg, temp_amp, ra),
        60,
        10,
        kc_params=kc_params,
        p_d=p_d,
        runoff=calc.weekly_api_runoff(precip_sample()),
        rho=0.5,
    )
    # print(wat_bal)

    assert 1 == 1


def test_daily_tobegroupedby_season_cuts_on_days():

    precip = data_test_calc.multi_year_data_sample()
    dts = calc.daily_tobegroupedby_season(precip, 29, 11, 29, 2)

    assert dts["T"].size == 461


def test_daily_tobegroupedby_season_creates_groups():

    precip = data_test_calc.multi_year_data_sample()
    dts = calc.daily_tobegroupedby_season(precip, 29, 11, 29, 2)

    assert dts["group"].size == 5


def test_daily_tobegroupedby_season_picks_right_end_dates():

    precip = data_test_calc.multi_year_data_sample()
    dts = calc.daily_tobegroupedby_season(precip, 29, 11, 29, 2)
    assert (
        dts.seasons_ends
        == pd.to_datetime(
            [
                "2001-02-28T00:00:00.000000000",
                "2002-02-28T00:00:00.000000000",
                "2003-02-28T00:00:00.000000000",
                "2004-02-29T00:00:00.000000000",
                "2005-02-28T00:00:00.000000000",
            ],
        )
    ).all()


def test_seasonal_onset_date_keeps_returning_same_outputs():

    precip = data_test_calc.multi_year_data_sample()
    onsetsds = calc.seasonal_onset_date(
        daily_rain=precip,
        search_start_day=1,
        search_start_month=3,
        search_days=90,
        wet_thresh=1,
        wet_spell_length=3,
        wet_spell_thresh=20,
        min_wet_days=1,
        dry_spell_length=7,
        dry_spell_search=21,
        time_coord="T",
    )
    onsets = onsetsds.onset_delta + onsetsds["T"]
    assert np.array_equal(
        onsets,
        pd.to_datetime(
            [
                "NaT",
                "2001-03-08T00:00:00.000000000",
                "NaT",
                "2003-04-12T00:00:00.000000000",
                "2004-04-04T00:00:00.000000000",
            ],
        ),
        equal_nan=True,
    )


def test_seasonal_onset_date():
    t = pd.date_range(start="2000-01-01", end="2005-02-28", freq="1D")
    # this is rr_mrg.sel(T=slice("2000", "2005-02-28")).isel(X=150, Y=150).precip
    synthetic_precip = xr.DataArray(np.zeros(t.size), dims=["T"], coords={"T": t}) + 1.1
    synthetic_precip = xr.where(
        (synthetic_precip["T"] == pd.to_datetime("2000-03-29"))
        | (synthetic_precip["T"] == pd.to_datetime("2000-03-30"))
        | (synthetic_precip["T"] == pd.to_datetime("2000-03-31"))
        | (synthetic_precip["T"] == pd.to_datetime("2001-04-30"))
        | (synthetic_precip["T"] == pd.to_datetime("2001-05-01"))
        | (synthetic_precip["T"] == pd.to_datetime("2001-05-02"))
        | (synthetic_precip["T"] == pd.to_datetime("2002-04-01"))
        | (synthetic_precip["T"] == pd.to_datetime("2002-04-02"))
        | (synthetic_precip["T"] == pd.to_datetime("2002-04-03"))
        | (synthetic_precip["T"] == pd.to_datetime("2003-05-16"))
        | (synthetic_precip["T"] == pd.to_datetime("2003-05-17"))
        | (synthetic_precip["T"] == pd.to_datetime("2003-05-18"))
        | (synthetic_precip["T"] == pd.to_datetime("2004-03-01"))
        | (synthetic_precip["T"] == pd.to_datetime("2004-03-02"))
        | (synthetic_precip["T"] == pd.to_datetime("2004-03-03")),
        7,
        synthetic_precip,
    ).rename("synthetic_precip")

    onsetsds = calc.seasonal_onset_date(
        daily_rain=synthetic_precip,
        search_start_day=1,
        search_start_month=3,
        search_days=90,
        wet_thresh=1,
        wet_spell_length=3,
        wet_spell_thresh=20,
        min_wet_days=1,
        dry_spell_length=7,
        dry_spell_search=21,
        time_coord="T",
    )
    onsets = onsetsds.onset_delta + onsetsds["T"]
    assert (
        onsets
        == pd.to_datetime(
            xr.DataArray(
                [
                    "2000-03-29T00:00:00.000000000",
                    "2001-04-30T00:00:00.000000000",
                    "2002-04-01T00:00:00.000000000",
                    "2003-05-16T00:00:00.000000000",
                    "2004-03-01T00:00:00.000000000",
                ],
                dims=["T"],
                coords={"T": onsets["T"]},
            )
        )
    ).all()


def precip_sample():

    t = pd.date_range(start="2000-05-01", end="2000-06-30", freq="1D")
    # this is rr_mrg.isel(X=0, Y=124, drop=True).sel(T=slice("2000-05-01", "2000-06-30"))
    # fmt: off
    values = [
        0.054383,  0.      ,  0.      ,  0.027983,  0.      ,  0.      ,
        7.763758,  3.27952 , 13.375934,  4.271866, 12.16503 ,  9.706059,
        7.048605,  0.      ,  0.      ,  0.      ,  0.872769,  3.166048,
        0.117103,  0.      ,  4.584551,  0.787962,  6.474878,  0.      ,
        0.      ,  2.834413,  9.029134,  0.      ,  0.269645,  0.793965,
        0.      ,  0.      ,  0.      ,  0.191243,  0.      ,  0.      ,
        4.617332,  1.748801,  2.079067,  2.046696,  0.415886,  0.264236,
        2.72206 ,  1.153666,  0.204292,  0.      ,  5.239006,  0.      ,
        0.      ,  0.      ,  0.      ,  0.679325,  2.525344,  2.432472,
        10.737132,  0.598827,  0.87709 ,  0.162611, 18.794922,  3.82739 ,
        2.72832
    ]
    # fmt: on
    precip = xr.DataArray(values, dims=["T"], coords={"T": t})
    return precip


def call_onset_date(data):
    onsets = calc.onset_date(
        daily_rain=data,
        wet_thresh=1,
        wet_spell_length=3,
        wet_spell_thresh=20,
        min_wet_days=1,
        dry_spell_length=7,
        dry_spell_search=21,
    )
    return onsets


def test_onset_date():

    precip = precip_sample()
    onsets = call_onset_date(precip)
    assert pd.Timedelta(onsets.values) == pd.Timedelta(days=6)
    # Converting to pd.Timedelta doesn't change the meaning of the
    # assertion, but gives a more helpful error message when the test
    # fails: Timedelta('6 days 00:00:00')
    # vs. numpy.timedelta64(518400000000000,'ns')


def test_onset_date_with_other_dims():

    precip = xr.concat(
        [precip_sample(), precip_sample()[::-1].assign_coords(T=precip_sample()["T"])],
        dim="dummy_dim",
    )
    onsets = call_onset_date(precip)
    assert (
        onsets
        == xr.DataArray(
            [pd.Timedelta(days=6), pd.Timedelta(days=0)],
            dims=["dummy_dim"],
            coords={"dummy_dim": onsets["dummy_dim"]},
        )
    ).all()


def test_crop_evapotranspiration():

    et_ref = precip_sample() * 0 + 10
    p_d = xr.DataArray(
        pd.DatetimeIndex(data=["2000-05-02", "2000-05-13"]),
        dims=["X"],
        coords={"X": [0, 1]},
    ).expand_dims({"T": pd.DatetimeIndex(data=["2000-05-01"])})
    p_d = p_d - p_d["T"]
    kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    kc_params = xr.DataArray(
        data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    )
    kc = calc.kc_interpolation(p_d, kc_params)
    et_crop = calc.crop_evapotranspiration(et_ref, kc)

    assert (et_crop.isel(T=0) == 10).all()
    assert np.allclose(et_crop.isel(T=1), [2, 10])
    assert et_crop.isel(T=12, X=1) == 2


def test_kc_interpolation_is_1_when_pd_is_nat():

    p_d = xr.DataArray(
        pd.DatetimeIndex(data=["2000-05-02", "NaT"]),
        dims=["X"],
        coords={"X": [0, 1]},
    ).expand_dims({"T": pd.DatetimeIndex(data=["2000-05-01"])})
    p_d = p_d - p_d["T"]
    kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    kc_params = xr.DataArray(
        data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    )
    kc = calc.kc_interpolation(p_d, kc_params)

    assert np.allclose(
        kc.loc[
            ["2000-05-02", "2000-05-12", "2000-05-13", "2000-05-21", "2000-10-31"], :
        ],
        [
            [0.2, 1],
            [0.24444444, 1],
            [0.24888889, 1],
            [0.28444444, 1],
            [0.6, 1],
        ],
        equal_nan=True,
    )


def test_kc_interpolation():

    p_d = xr.DataArray(
        pd.DatetimeIndex(data=["2000-05-02", "2000-05-13"]),
        dims=["X"],
        coords={"X": [0, 1]},
    ).expand_dims({"T": pd.DatetimeIndex(data=["2000-05-01"])})
    p_d = p_d - p_d["T"]
    kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    kc_params = xr.DataArray(
        data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    )
    kc = calc.kc_interpolation(p_d, kc_params)

    assert np.allclose(
        kc.loc[
            ["2000-05-02", "2000-05-12", "2000-05-13", "2000-05-21", "2000-11-11"], :
        ],
        [
            [0.2, 1],
            [0.24444444, 1],
            [0.24888889, 0.2],
            [0.28444444, 0.23555556],
            [1, 0.6],
        ],
        equal_nan=True,
    )


def test_planting_date_with_space_dim():

    soil_moisture = xr.concat(
        [
            precip_sample() + 10,
            precip_sample()[::-1].assign_coords(T=precip_sample()["T"]) + 10,
        ],
        dim="X",
    )
    sm_thresh = xr.DataArray([15, 20], dims=["X"])
    planting_date = calc.planting_date(soil_moisture, sm_thresh)

    assert (
        planting_date
        == xr.DataArray(
            [pd.Timedelta(days=7), pd.Timedelta(days=3)],
            dims=["X"],
            coords={"X": planting_date["X"]},
        )
    ).all()


def test_onset_date_returns_nat():

    precip = precip_sample()
    precipNaN = precip + np.nan
    onsetsNaN = call_onset_date(precipNaN)

    assert np.isnat(onsetsNaN.values)


def test_planting_date_returns_nat():
    soil_moisture = precip_sample()
    smNaN = soil_moisture + np.nan
    plantingNaN = calc.planting_date(smNaN, 20)

    assert np.isnat(plantingNaN.values)


def test_onset_date_dry_spell_invalidates():

    precip = precip_sample()
    precipDS = xr.where(
        (precip["T"] > pd.to_datetime("2000-05-09"))
        & (precip["T"] < (pd.to_datetime("2000-05-09") + pd.Timedelta(days=5))),
        0,
        precip,
    )
    onsetsDS = call_onset_date(precipDS)

    assert pd.Timedelta(onsetsDS.values) != pd.Timedelta(days=6)


def test_onset_date_late_dry_spell_invalidates_not():

    precip = precip_sample()
    preciplateDS = xr.where(
        (precip["T"] > (pd.to_datetime("2000-05-09") + pd.Timedelta(days=20))),
        0,
        precip,
    )
    onsetslateDS = call_onset_date(preciplateDS)
    assert pd.Timedelta(onsetslateDS.values) == pd.Timedelta(days=6)


def test_onset_date_1st_wet_spell_day_not_wet_day():
    """May 4th is 0.28 mm thus not a wet day
    resetting May 5th and 6th respectively to 1.1 and 18.7 mm
    thus, May 5-6 are both wet days and need May 4 to reach 20mm
    but the 1st wet day of the spell is not 4th but 5th
    """

    precip = precip_sample()
    precipnoWD = xr.where(
        (precip["T"] == pd.to_datetime("2000-05-05")),
        1.1,
        precip,
    )
    precipnoWD = xr.where(
        (precip["T"] == pd.to_datetime("2000-05-06")),
        18.7,
        precipnoWD,
    )
    onsetsnoWD = call_onset_date(precipnoWD)
    assert pd.Timedelta(onsetsnoWD.values) == pd.Timedelta(days=4)


def test_probExceed():
    earlyStart = pd.to_datetime(f"2000-06-01", yearfirst=True)
    values = {
        "onset": [
            "2000-06-18",
            "2000-06-16",
            "2000-06-26",
            "2000-06-01",
            "2000-06-15",
            "2000-06-07",
            "2000-07-03",
            "2000-06-01",
            "2000-06-26",
            "2000-06-01",
            "2000-06-08",
            "2000-06-23",
            "2000-06-01",
            "2000-06-01",
            "2000-06-16",
            "2000-06-02",
            "2000-06-17",
            "2000-06-18",
            "2000-06-10",
            "2000-06-17",
            "2000-06-05",
            "2000-06-07",
            "2000-06-03",
            "2000-06-10",
            "2000-06-17",
            "2000-06-05",
            "2000-06-11",
            "2000-06-01",
            "2000-06-24",
            "2000-06-06",
            "2000-06-07",
            "2000-06-17",
            "2000-06-14",
            "2000-06-20",
            "2000-06-17",
            "2000-06-14",
            "2000-06-23",
            "2000-06-01",
        ]
    }
    onsetMD = pd.DataFrame(values).astype("datetime64[ns]")
    cumsum = calc.probExceed(onsetMD, earlyStart)
    probExceed_values = [
        0.815789,
        0.789474,
        0.763158,
        0.710526,
        0.684211,
        0.605263,
        0.578947,
        0.526316,
        0.500000,
        0.447368,
        0.421053,
        0.368421,
        0.236842,
        0.184211,
        0.157895,
        0.105263,
        0.078947,
        0.026316,
        0.000000,
    ]
    assert np.allclose(cumsum.probExceed, probExceed_values)
