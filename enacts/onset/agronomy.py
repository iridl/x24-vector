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
    kc_params=None,
    planting_date=None,
    sm_threshold=None,
    time_dim="T",
):
    """Compute soil-plant-water balance day after day over a growing season.
    See `soil_plant_water_step` for the step by step algorithm definition.

    The daily evapotranspiration `et` can be scaled by a Crop Cultivar Kc
    modelizing a crop needs in water according to the stage of its growth.
    Kc is set to 1 outside of the growing period. i.e. before planting date
    and after the last Kc curve inflection point. The planting date can either be
    prescribed as a parameter or evaluated by the simulation as the day following
    the day that soil moisture reached a cetain value for the first time.
    
    Parameters
    ----------
    peffective : DataArray
        Daily effective precipitation.
    et : DataArray
        Daily evapotranspiration of the plant.
    taw : DataArray
        Total available water that represents the maximum water capacity of the soil.
    sminit : DataArray
        Timeless soil moisture to initialize the loop with.
    kc_params : DataArray
        Crop Cultivar Kc parameters as a function of the inflection points of the Kc curve,
        expressed in consecutive daily time deltas originating from `planting_date`
        as coordinate `kc_periods` (default `kc_params` =None in which case Kc is set to 1).
    planting_date : DataArray[datetime64[ns]]
        Dates when planting (default `planting_date` =None in which case
        `planting_date` is assigned by the simulation according to a soil moisture
        criterion parametrizable through `sm_threshold` )
    sm_threshold : DataArray
        Planting the day after soil moisture is greater or equal to `sm_threshold`
        in units of soil moisture (default `sm_threshold` =None in which case
        `planting_date` must be defined)
    time_dim : str, optional
        Daily time dimension to run the balance against (default `time_dim` ="T").
        
    Returns
    -------
    sm, drainage, et_crop, planting_date : Tuple of DataArray
        Daily soil moisture, drainage and crop evapotranspiration over the growing season.
        Planting dates as given in parameters or as evaluated by the simulation.
        
    See Also
    --------
    soil_plant_water_step
    
    Examples
    --------
    Example of kc_params:
    
    >>> kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    >>> kc_params = xr.DataArray(
    >>>    data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    >>> )
    <xarray.DataArray (kc_periods: 5)>
    array([0.2, 0.4, 1.2, 1.2, 0.6])
    Coordinates:
        * kc_periods  (kc_periods) timedelta64[ns] 0 days 45 days ... 45 days 45 days
    
    Example of planting_date:
    
    >>> p_d = xr.DataArray(
    >>>    pd.DatetimeIndex(data=["2000-05-02", "2000-05-13"]),
    >>>    dims=["station"],
    >>>    coords={"station": [0, 1]},
    >>> )
    <xarray.DataArray (station: 2)>
    array(['2000-05-02T00:00:00.000000000', '2000-05-13T00:00:00.000000000'],
        dtype='datetime64[ns]')
    Coordinates:
        * station        (station) int64 0 1
    """
    
    # Initializations
    et = xr.DataArray(et)
    if kc_params is None:
        if planting_date is not None or sm_threshold is not None:
            raise Exception(
                "if Kc is not defined, neither planting_date nor sm_threshold should be"
            )
        kc = 1
        et_crop = et
    else: #et_crop depends on et, kc and planting_date dims, and time_dim
        kc_inflex = kc_params.assign_coords(
            kc_periods=kc_params["kc_periods"].cumsum(dim="kc_periods")
        )
        if planting_date is not None: # distance between 1st and planting days
            if sm_threshold is not None:
                raise Exception("either planting_date or sm_threshold should be defined")
            planted_since = peffective[time_dim][0].drop_vars(time_dim) - planting_date
        else: # 1st day is planting day if sminit met condition
            if sm_threshold is not None:
                planted_since = xr.where(
                    sminit >= sm_threshold, 0, np.nan
                ).astype("timedelta64[D]")
            else:
                raise Exception("if planting_date is not defined, then define a sm_threshold")
        et_crop = (xr.full_like(et, fill_value=np.nan)
            + xr.zeros_like(kc_params.isel({"kc_periods": 0}, drop=True))
            + xr.zeros_like(peffective[time_dim], dtype=et.dtype)
            + xr.zeros_like(planted_since, dtype=et.dtype)
        )
    # sminit depends on peffective, et_crop and taw dims, but not time_dim
    sminit = (sminit
        + xr.zeros_like(peffective.isel({time_dim: 0}, drop=True))
        + xr.zeros_like(et_crop.isel({time_dim: 0}, missing_dims='ignore', drop=True))
        + 0 * taw
    )
    # sm depends on sminit dims and time_dim
    sm = xr.full_like(sminit, fill_value=np.nan) + xr.zeros_like(peffective)
    # sminit is the day before first time_dim point
    sminit = sminit.assign_coords({time_dim: peffective[time_dim][0] - np.timedelta64(1, "D")})
    # drainage depends on sm dims
    drainage = xr.full_like(sm, fill_value=np.nan)
    # sm starts with initial condition sminit
    sm = xr.concat([sminit, sm], time_dim)
    # Filling/emptying bucket day after day
    for doy in range(0, peffective[time_dim].size):
        if kc_params is not None: # interpolate kc value per distance from planting
            kc = kc_inflex.interp(
                kc_periods=planted_since, kwargs={"fill_value": 1}
            ).where(lambda x: x.notnull(), other=1).drop_vars("kc_periods")
            if time_dim in et_crop.dims: # et _crop depends on time_dim but et might not
                et_crop[{time_dim: doy}] = kc * et.isel({time_dim: doy}, missing_dims='ignore')
        # water balance step
        sm[{time_dim: doy+1}], drainage[{time_dim: doy}] = soil_plant_water_step(
            sm.isel({time_dim: doy}, drop=True),
            peffective.isel({time_dim: doy}, drop=True),
            et_crop.isel({time_dim: doy}, missing_dims='ignore', drop=True),
            taw,
        )
        # Increment planted_since
        if kc_params is not None:
            if planting_date is None: # did doy met planting conditions?
                planted_since = planted_since.where(
                    lambda x: x.notnull(), # no planting date found yet
                    other=xr.where( # next day is planting if sm condition met
                        sm.isel({time_dim: doy+1}) >= sm_threshold, -1, np.nan
                    ).astype("timedelta64[D]"),
                )
            planted_since = planted_since + np.timedelta64(1, "D")
    # Let's have sm same shape as other variables
    sm = sm.isel({time_dim: slice(1,None)})
    # Let's save planting_date
    if kc_params is not None and planting_date is None:
        planting_date = (peffective[time_dim][-1].drop_vars(time_dim)
            - (planted_since - np.timedelta64(1, "D"))
        )
    return sm, drainage, et_crop, planting_date