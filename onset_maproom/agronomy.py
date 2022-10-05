import xarray as xr

def soil_plant_water_balance(
sm_previous_day,
rain,
et,
taw,
runoff=0
):
    """Compute soil-plant-water balance from one day to the next.
    The balance is defined as:
    `sm`(t) + `drainage`(t) = `sm`(t-1) + `peffective`(t) - `et`(t)
    where:
    `sm` is the soil moisture and can not exceed total available water `taw`.
    `drainage` is the residual soil moisutre occasionally excessing `taw` that drains through the soil.
    `peffective` is the effective precipitation that enters the soil and is the `rain` minus a `runoff`.
    `et` is the evapotranspiration consumed by the plant.
    
    Parameters
    ------
    sm_previous_day : DataArray
        soil moisture of the previous day.
    rain : DataArray
        rainfall during the day.
    et : DataArray
        evapotransipiration of the plant during the day.
    taw: DataArray
        total available water that represents the maximum water capacity of the soil
    runoff : DataArray
        amount of rainfall lost to runoff during the day (default `runoff`=0).
    Returns
    -------
    sm, peffective, drainage : Tuple of DataArray
        next day soil moisture, effective precipitation and drainage
    See Also
    --------
    Notes
     -----
    """
    # Water Balance
    peffective = rain - runoff
    wb = sm_previous_day + peffective - et
    drainage = (wb - taw).clip(min=0)
    sm = wb - drainage
    return sm, peffective, drainage