import xarray as xr
import agronomy


def test_spwb_basic():

    sm_previous_day = xr.DataArray(30)
    rain = xr.DataArray(10)
    et = xr.DataArray(5)
    taw = xr.DataArray(60)
    sm, peffective, drainage = agronomy.soil_plant_water_balance(
        sm_previous_day,
        rain,
        et,
        taw
    )
    
    assert peffective == 10
    assert drainage == 0
    assert sm == 35
    
    
def test_spwb_with_dims_and_drainage():

    sm_previous_day = xr.DataArray([30, 56])
    rain = xr.DataArray([10, 10])
    et = xr.DataArray([5, 5])
    taw = xr.DataArray([60, 60])
    sm, peffective, drainage = agronomy.soil_plant_water_balance(
        sm_previous_day,
        rain,
        et,
        taw
    )

    assert (peffective == [10, 10]).all()
    assert (drainage == [0, 1]).all()
    assert (sm == [35, 60]).all()


