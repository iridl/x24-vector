import xarray as xr

def soil_plant_water_bucket(
    sm_yesterday,
    peffective,
    et,
    taw
):
    """Compute soil-plant-water balance from yesterday to today.
    The balance is thought as a bucket with water coming in and out:
    
    `sm` (t) + `drainage` (t) = `sm` (t-1) + `peffective` (t) - `et` (t)
    
    where:
    
    `sm` is the soil moisture and can not exceed total available water `taw`.
    
    `drainage` is the residual soil moisture occasionally exceeding `taw`
    that drains through the soil.
    
    `peffective` is the effective precipitation that enters the soil.
    
    `et` is the evapotranspiration yielded by the plant.
    
    Parameters
    ----------
    sm_yesterday : DataArray
        soil moisture of yesterday.
    peffective : DataArray
        effective precipitation today.
    et : DataArray
        evapotransipiration of the plant today.
    taw : DataArray
        total available water that represents the maximum water capacity of the soil.
        
    Returns
    -------
    sm, drainage : Tuple of DataArray
        today soil moisture and drainage
    
    """
    
    # Water Balance
    wb = sm_yesterday + peffective - et
    drainage = (wb - taw).clip(min=0)
    sm = wb - drainage
    return sm, drainage