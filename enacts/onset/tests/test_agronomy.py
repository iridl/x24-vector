import xarray as xr
import agronomy
import pandas as pd
import numpy as np


def test_spwbu_basic():

    sm_previous_day = xr.DataArray(30)
    peffective = xr.DataArray(10)
    et = xr.DataArray(5)
    taw = xr.DataArray(60)
    sm, drainage = agronomy.soil_plant_water_step(
        sm_previous_day,
        peffective,
        et,
        taw,
    )
    
    assert drainage == 0
    assert sm == 35
    
    
def test_spwb_with_dims_and_drainage():

    sm_previous_day = xr.DataArray([30, 56])
    peffective = xr.DataArray([10, 10])
    et = xr.DataArray([5, 5])
    taw = xr.DataArray([60, 60])
    sm, drainage = agronomy.soil_plant_water_step(
        sm_previous_day,
        peffective,
        et,
        taw,
    )

    assert (drainage == [0, 1]).all()
    assert (sm == [35, 60]).all()


def test_spwba_basic():
    
    sm, drainage, et_crop = agronomy.soil_plant_water_balance(
        precip_sample(),
        5,
        60,
        10,
    )
    expected = [
        5.054383,  0.054383,  0.      ,  0.      ,  0.      ,  0.      ,
        2.763758,  1.043278,  9.419212,  8.691078, 15.856108, 20.562167,
        22.610772, 17.610772, 12.610772,  7.610772,  3.483541,  1.649589,
        0.      ,  0.      ,  0.      ,  0.      ,  1.474878,  0.      ,
        0.      ,  0.      ,  4.029134,  0.      ,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.239006,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,
        5.737132,  1.335959,  0.      ,  0.      , 13.794922, 12.622312,
        10.350632
    ]
    
    assert np.allclose(sm, expected)
    assert np.allclose(drainage, 0)


def test_spwba_kc():
    kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    kc_params = xr.DataArray(
        data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    )
    p_d = xr.DataArray(
        pd.DatetimeIndex(data=["2000-05-04"]),
        dims=["X"],
        coords={"X": [0]},
    )
    sm, drainage, et_crop = agronomy.soil_plant_water_balance(
        precip_sample(),
        5,
        60,
        10,
        kc_params=kc_params,
        planting_date=p_d,
    )
    kc_inflex = kc_params.assign_coords(
        kc_periods=kc_params["kc_periods"].cumsum(dim="kc_periods")
    )
    planted_since = precip_sample()["T"] - p_d
    kc = kc_inflex.interp(
        kc_periods=planted_since, kwargs={"fill_value": 1}
    )
    et_crop_vect = kc * 5
        
    assert np.allclose(
        et_crop,
        et_crop_vect,
    )
    
    
def test_spwba_kc_2pds():
    kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    kc_params = xr.DataArray(
        data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    )
    p_d = xr.DataArray(
        pd.DatetimeIndex(data=["2000-05-02", "2000-05-13"]),
        dims=["X"],
        coords={"X": [0, 1]},
    )
    sm, drainage, et_crop = agronomy.soil_plant_water_balance(
        precip_sample(),
        5,
        60,
        10,
        kc_params=kc_params,
        planting_date=p_d,
    )
    kc_inflex = kc_params.assign_coords(
        kc_periods=kc_params["kc_periods"].cumsum(dim="kc_periods")
    )
    planted_since = precip_sample()["T"] - p_d
    kc = kc_inflex.interp(
        kc_periods=planted_since, kwargs={"fill_value": 1}
    )
    et_crop_vect = kc * 5
        
    assert np.allclose(
        et_crop,
        et_crop_vect,
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
