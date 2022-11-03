import xarray as xr
import numpy as np


def soil_plant_water_step(
    sm_yesterday,
    peffective,
    et,
    taw,
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
        evapotranspiration of the plant today.
    taw : DataArray
        total available water that represents the maximum water capacity of the soil.
        
    Returns
    -------
    sm, drainage : Tuple of DataArray
        today soil moisture and drainage
        
    See Also
    --------
    soil_plant_water_balance
    
    """
    
    # Water Balance
    wb = (sm_yesterday + peffective - et).clip(min=0)
    drainage = (wb - taw).clip(min=0)
    sm = wb - drainage
    return sm, drainage


def soil_plant_water_balance(
    peffective,
    et,
    taw,
    sminit,
    time_dim="T",
):
    """Compute soil-plant-water balance day after day over a growing season.
    See `soil_plant_water_step` for the step by step algorithm definition.
    
    Parameters
    ----------
    peffective : DataArray
        daily effective precipitation.
    et : DataArray
        daily evapotranspiration of the plant.
    taw : DataArray
        total available water that represents the maximum water capacity of the soil.
    sminit : DataArray
        timeless soil moisture to initialize the loop with.
    time_dim : str, optional
        daily time dimension to run the balance against (default `time_dim` ="T").
        
    Returns
    -------
    sm, drainage : Tuple of DataArray
        daily soil moisture and drainage over the growing season.
        
    See Also
    --------
    soil_plant_water_step
    
    """
    
    # First Step
    if np.size(et) == 1:
        et = xr.DataArray(et)
    sm0, drainage0 = soil_plant_water_step(
        sminit,
        peffective.isel({time_dim: 0}, drop=True),
        et.isel({time_dim: 0}, missing_dims='ignore', drop=True),
        taw,
    )
    # Give time dimension to sm and drainage    
    sm = sm0.expand_dims({time_dim: peffective[time_dim]}).copy()
    drainage = drainage0.expand_dims({time_dim: peffective[time_dim]}).copy()
    # Filling/emptying bucket day after day
    for doy in range(1, peffective[time_dim].size):
        sm[{time_dim: doy}], drainage[{time_dim: doy}] = soil_plant_water_step(
            sm.isel({time_dim: doy - 1}),
            peffective.isel({time_dim: doy}),
            et.isel({time_dim: doy}, missing_dims='ignore'),
            taw,
        )
    return sm, drainage