import numpy as np
import pandas as pd
import xarray as xr
import agronomy
import data_test_calc


def test_calibrate_available_water():

    taw = xr.DataArray(data=[60, 40], dims="X", coords={"X": [1, 2]})
    rho = xr.DataArray(
        data=[0.5, 0.45], dims="Crop", coords={"Crop": ["Maize", "Rice"]}
    )
    raw = agronomy.calibrate_available_water(taw, rho)

    assert (raw == [[30.0, 20.0], [27.0, 18.0]]).all()


def test_single_stress_coeff():

    et_crop = precip_sample() * 0 + 5
    soil_moisture = xr.where(precip_sample() > 10, 31, 15)
    taw = 60
    raw = 0.5 * taw
    ks = agronomy.single_stress_coeff(soil_moisture, taw, raw)

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
    ra = agronomy.solar_radiation(doy, lat)
    et_ref = agronomy.hargreaves_et_ref(temp_avg, temp_amp, ra)
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
    ra = agronomy.solar_radiation(doy, lat)

    assert (
        ra.isel(T=0, Y=-1) < ra.isel(T=-1, Y=-1) and
        ra.isel(T=0, Y=-1) < ra.isel(T=0, Y=0) and
        ra.isel(T=0, Y=-1) < ra.isel(T=-1, Y=0) and
        ra.isel(T=0, Y=0) < ra.isel(T=-1, Y=0) and
        ra.isel(T=0, Y=0) < ra.isel(T=-1, Y=0)
    )

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
    runoff = agronomy.scs_cn_runoff(precip, 75)
    peff, runoff_eunjin = Peffective_2D(precip, 75)

    assert np.allclose(runoff, runoff_eunjin)


def test_api_sum():

    x = np.array([6, 5, 4, 3, 2, 1, 2])
    api = agronomy.api_sum(x, axis=(0,))

    assert api == 7


def test_weekly_api_runoff():

    precip = precip_sample() + 5
    runoff = agronomy.weekly_api_runoff(precip)
    other_api = precip.rolling(**{"T": 7}).reduce(agronomy.api_sum)
    other_runoff = (
        agronomy.api_runoff_select(precip, other_api).clip(min=0).isel(T=slice(6, None))
    )

    assert np.allclose(runoff, other_runoff)


def test_water_balance_intializes_right():

    precip = precip_sample()
    wb = agronomy.water_balance(precip, 5, 60, 0)

    assert wb.soil_moisture.isel(T=0) == 0


def test_water_balance():

    precip = precip_sample()
    wb = agronomy.water_balance(precip, 5, 60, 0)

    assert np.allclose(wb.soil_moisture.isel(T=-1), 10.350632)


def test_water_balance2():

    t = pd.date_range(start="2000-05-01", end="2000-05-04", freq="1D")
    values = [
        [5.0, 6.0, 3.0, 66.0],
        [10.0, 12.0, 14.0, 16.0],
    ]
    precip = xr.DataArray(values, dims=["X", "T"], coords={"T": t})
    wb = agronomy.water_balance(precip, 5, 60, 0)

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
    wb = agronomy.water_balance(precip, et, 60, 0)

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
    wb = agronomy.water_balance(precip, et, 60, 0)

    assert np.array_equal(wb.soil_moisture["T"], t)
    expected = [
        [0.0, 0.0, 0.0, 56.0],
        [5.0, 7.0, 6.0, 12.0],
    ]
    assert np.array_equal(wb.soil_moisture, expected)


def test_soil_plant_water_balance_with_et_ref_cst():

    wat_bal = agronomy.soil_plant_water_balance(
        precip_sample(),
        5,
        60,
        10,
        kc_params=None,
        p_d=None,
        rho=None,
        runoff=agronomy.weekly_api_runoff(precip_sample()),
    )

    assert (wat_bal.et == 5).all()
    assert (wat_bal.et_crop == 5).all()
    assert (wat_bal.et_crop_red == 5).all()
    expected = [12.763758  , 11.043278  , 19.419212  , 18.691078  , 25.856108  ,
       30.562167  , 32.610772  , 27.610772  , 22.610772  , 17.610772  ,
       13.483541  , 11.649589  ,  6.766692  ,  1.766692  ,  1.351243  ,
        0.        ,  1.474878  ,  0.        ,  0.        ,  0.        ,
        4.029134  ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.239006  ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  5.737132  ,  1.335959  ,
        0.        ,  0.        , 13.22708763, 12.05447763,  9.78279763]
    assert np.allclose(wat_bal.soil_moisture, expected)


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
    ra = agronomy.solar_radiation(doy, lat)
    wat_bal = agronomy.soil_plant_water_balance(
        precip_sample(),
        agronomy.hargreaves_et_ref(temp_avg, temp_amp, ra),
        60,
        10,
        kc_params=None,
        p_d=None,
        rho=None,
        runoff=agronomy.weekly_api_runoff(precip_sample()),
    )

    assert (wat_bal.et_crop == wat_bal.et_crop_red).all()
    expected = [[10.53881199],
       [10.59168222],
       [ 9.38878189],
       [ 7.42175294],
       [ 5.29016389],
       [ 7.34315793],
       [ 5.21220041],
       [ 3.0815251 ],
       [ 0.95110917],
       [ 0.        ],
       [ 0.        ],
       [ 0.        ],
       [ 0.        ],
       [ 6.34475728],
       [ 4.69855057],
       [ 3.27655007],
       [ 1.27869775],
       [13.15848481],
       [14.09643733],
       [14.15885817]
    ]
    assert np.allclose(
        wat_bal.soil_moisture.isel(T=slice(-20, None)),
        expected
    )


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
    ra = agronomy.solar_radiation(doy, lat)
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
    wat_bal = agronomy.soil_plant_water_balance(
        precip_sample(),
        agronomy.hargreaves_et_ref(temp_avg, temp_amp, ra),
        60,
        10,
        kc_params=kc_params,
        p_d=p_d,
        rho=None,
        runoff=agronomy.weekly_api_runoff(precip_sample()),
    )

    assert (wat_bal.et_crop == wat_bal.et_crop_red).all()
    expected = [[[55.81081862, 56.74815773]],
       [[57.01759313, 58.28147063]],
       [[58.09657863, 59.71744128]],
       [[60.        , 60.        ]],
       [[59.35689346, 59.72076937]],
       [[58.92301271, 59.68844135]],
       [[57.8169261 , 58.98686702]],
       [[60.        , 60.        ]],
       [[60.        , 60.        ]],
       [[60.        , 60.        ]]]
    assert np.allclose(
        wat_bal.soil_moisture.isel(T=slice(-10, None)),
        expected
    )
    

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
    ra = agronomy.solar_radiation(doy, lat)
    kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    kc_params = xr.DataArray(
        data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    )
    wat_bal = agronomy.soil_plant_water_balance(
        precip_sample(),
        agronomy.hargreaves_et_ref(temp_avg, temp_amp, ra),
        60,
        10,
        kc_params=kc_params,
        rho=None,
        runoff=agronomy.weekly_api_runoff(precip_sample()),
    )

    assert wat_bal.p_d == pd.Timedelta(days=4)
    assert (
        wat_bal.et.isel(T=slice(0, 3)) == wat_bal.et_crop.isel(T=slice(0, 3))
    ).all()
    assert (wat_bal.et.isel(T=4) != wat_bal.et_crop.isel(T=4)).any()
    assert (wat_bal.et_crop == wat_bal.et_crop_red).all()
    expected = [[59.18486355],
       [60.        ],
       [60.        ],
       [59.43226677],
       [58.66489471],
       [60.        ],
       [59.21391345],
       [58.41846133],
       [57.61363754],
       [56.79943557],
       [56.60437686],
       [58.1026769 ],
       [59.50388587],
       [60.        ],
       [59.66260018],
       [59.54178707],
       [58.72989121],
       [60.        ],
       [60.        ],
       [60.        ]
    ]
    assert np.allclose(
        wat_bal.soil_moisture.isel(T=slice(-20, None)),
        expected
    )
    

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
    ra = agronomy.solar_radiation(doy, lat)
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
    wat_bal = agronomy.soil_plant_water_balance(
        precip_sample(),
        agronomy.hargreaves_et_ref(temp_avg, temp_amp, ra),
        60,
        10,
        kc_params=kc_params,
        p_d=p_d,
        runoff=agronomy.weekly_api_runoff(precip_sample()),
        rho=0.5,
    )
    
    assert np.allclose(
        wat_bal.et_crop.where(wat_bal.soil_moisture == 60, drop=True),
        wat_bal.et_crop_red.where(wat_bal.soil_moisture == 60, drop=True),
        equal_nan=True,
    )
    expected = [[[55.81081862, 56.74815773]],
       [[57.01759313, 58.28147063]],
       [[58.09657863, 59.71744128]],
       [[60.        , 60.        ]],
       [[59.35689346, 59.72076937]],
       [[58.92301271, 59.68844135]],
       [[57.8169261 , 58.98686702]],
       [[60.        , 60.        ]],
       [[60.        , 60.        ]],
       [[60.        , 60.        ]]]
    assert np.allclose(
        wat_bal.soil_moisture.isel(T=slice(-10, None)),
        expected
    )


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
    kc = agronomy.kc_interpolation(p_d, kc_params)
    et_crop = agronomy.crop_evapotranspiration(et_ref, kc)

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
    kc = agronomy.kc_interpolation(p_d, kc_params)

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
    kc = agronomy.kc_interpolation(p_d, kc_params)

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
    planting_date = agronomy.planting_date(soil_moisture, sm_thresh)

    assert (
        planting_date
        == xr.DataArray(
            [pd.Timedelta(days=7), pd.Timedelta(days=3)],
            dims=["X"],
            coords={"X": planting_date["X"]},
        )
    ).all()


def test_planting_date_returns_nat():
    soil_moisture = precip_sample()
    smNaN = soil_moisture + np.nan
    plantingNaN = agronomy.planting_date(smNaN, 20)

    assert np.isnat(plantingNaN.values)
