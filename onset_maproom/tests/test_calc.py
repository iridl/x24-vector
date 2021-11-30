import numpy as np
import pandas as pd
import xarray as xr
import calc
import data_test_calc


def test_estimate_sm_intializes_right():

    precip = precip_sample()
    sm = calc.estimate_sm(precip, 5, 60, 0)

    assert sm.isel(T=0) == 0


def test_estimate_sm():

    precip = precip_sample()
    sm = calc.estimate_sm(precip, 5, 60, 0)

    # assert sm.isel(T=-1) == 10.350632000000001
    assert np.allclose(sm.isel(T=-1), 10.350632)


def test_estimate_sm2():

    t = pd.date_range(start="2000-05-01", end="2000-05-04", freq="1D")
    values = [
        [5.0, 6.0, 3.0, 66.0],
        [10.0, 12.0, 14.0, 16.0],
    ]
    precip = xr.DataArray(values, dims=["X", "T"], coords={"T": t})
    sm = calc.estimate_sm(precip, 5, 60, 0)

    assert np.array_equal(sm["T"], t)
    expected = [
        [0., 1., 0., 60.],
        [5., 12., 21., 32.],
    ]
    assert np.array_equal(sm, expected)


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
    print(onsets)
    assert (
        onsets
        == xr.DataArray(
            [pd.Timedelta(days=6), pd.Timedelta(days=0)],
            dims=["dummy_dim"],
            coords={"dummy_dim": onsets["dummy_dim"]},
        )
    ).all()


def test_onset_date_returns_nat():

    precip = precip_sample()
    precipNaN = precip + np.nan
    onsetsNaN = call_onset_date(precipNaN)
    assert np.isnat(onsetsNaN.values)


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
    earlyStart = pd.to_datetime(f'2000-06-01', yearfirst=True)
    values = {'onset': ['2000-06-18', '2000-06-16', '2000-06-26', '2000-06-01', '2000-06-15', '2000-06-07', '2000-07-03', '2000-06-01', '2000-06-26', '2000-06-01', '2000-06-08', '2000-06-23', '2000-06-01', '2000-06-01', '2000-06-16', '2000-06-02', '2000-06-17', '2000-06-18', '2000-06-10', '2000-06-17', '2000-06-05', '2000-06-07', '2000-06-03', '2000-06-10', '2000-06-17', '2000-06-05', '2000-06-11', '2000-06-01', '2000-06-24', '2000-06-06', '2000-06-07', '2000-06-17', '2000-06-14', '2000-06-20', '2000-06-17', '2000-06-14', '2000-06-23', '2000-06-01']}
    onsetMD = pd.DataFrame(values).astype('datetime64[ns]')
    cumsum = calc.probExceed(onsetMD, earlyStart)
    probExceed_values = [0.815789,0.789474,0.763158,0.710526,0.684211,0.605263,0.578947,0.526316,0.500000,0.447368,0.421053,0.368421,0.236842,0.184211,0.157895,0.105263,0.078947,0.026316,0.000000]
    assert np.allclose(cumsum.probExceed, probExceed_values)
