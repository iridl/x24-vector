import xarray as xr
import agronomy


def test_spwb_basic():

    sm_previous_day = xr.DataArray(30)
    peffective = xr.DataArray(10)
    et = xr.DataArray(5)
    taw = xr.DataArray(60)
    sm, drainage = agronomy.soil_plant_water_bucket(
        sm_previous_day,
        peffective,
        et,
        taw
    )
    
    assert drainage == 0
    assert sm == 35
    
    
def test_spwb_with_dims_and_drainage():

    sm_previous_day = xr.DataArray([30, 56])
    peffective = xr.DataArray([10, 10])
    et = xr.DataArray([5, 5])
    taw = xr.DataArray([60, 60])
    sm, drainage = agronomy.soil_plant_water_bucket(
        sm_previous_day,
        peffective,
        et,
        taw
    )

    assert (drainage == [0, 1]).all()
    assert (sm == [35, 60]).all()


