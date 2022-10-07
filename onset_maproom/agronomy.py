import xarray as xr

def soil_plant_water_balance(
    sm_yesterday,
    rain,
    et,
    taw,
    runoff=0
):
    """Compute soil-plant-water balance from yesterday to today.
    The balance is defined as:
    
    `sm` (t) + `drainage` (t) = `sm` (t-1) + `peffective` (t) - `et` (t)
    
    where:
    
    `sm` is the soil moisture and can not exceed total available water `taw`.
    
    `drainage` is the residual soil moisture occasionally exceeding `taw` that drains through the soil.
    
    `peffective` is the effective precipitation that enters the soil and is the `rain` minus a `runoff`.
    
    `et` is the evapotranspiration yielded by the plant.
    
    Parameters
    ----------
    sm_yesterday : DataArray
        soil moisture of yesterday.
    rain : DataArray
        rainfall today.
    et : DataArray
        evapotransipiration of the plant today.
    taw : DataArray
        total available water that represents the maximum water capacity of the soil
    runoff : DataArray, optional
        amount of rainfall lost to runoff today (default `runoff` =0).
        
    Returns
    -------
    sm, peffective, drainage : Tuple of DataArray
        today soil moisture, effective precipitation and drainage
    
    """
    
    # Water Balance
    peffective = rain - runoff
    wb = sm_yesterday + peffective - et
    drainage = (wb - taw).clip(min=0)
    sm = wb - drainage
    return sm, peffective, drainage