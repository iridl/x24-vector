import os
import shutil
import sys
import numpy as np
import xarray as xr
import datetime as dt
from pathlib import Path
import pingrid
from functools import partial


CONFIG = pingrid.load_config(os.environ["CONFIG"])

def set_up_dims(xda, time_res="daily"):
    
    datestr = Path(xda.encoding["source"]).name.split("_")[2]
    year = int(datestr[0:4])
    month = int(datestr[4:6])
    if time_res == "daily":
        day = int(datestr[6:8])
    elif time_res == "dekadal":
        day = (int(datestr[6:7]) - 1) * 10 + 1
    else:
        raise Exception(
            "time resolution must be 'daily' or 'dekadal' "
        )
    xda = xda.expand_dims(T = [dt.datetime(year, month, day)])
    xda = xda.rename({'Lon': 'X','Lat': 'Y'})
    
    return xda

def regridding(data, resolution):
    if not np.isclose(data['X'][1] - data['X'][0], resolution):
    # TODO this method of regridding is inaccurate because it pretends
    # that (X, Y) define a Euclidian space. In reality, grid cells
    # farther from the equator cover less area and thus should be
    # weighted less heavily. Also, consider using conservative
    # interpolation instead of bilinear, since when going to a much
    # coarser resoution, bilinear discards a lot of information. See [1],
    # and look into xESMF.
    #
    # [1] https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/regridding-overview
        print("Your data will be regridded. Refer to function documentation for more information on this.")
        data = data.interp(
            X=np.arange(data.X.min(), data.X.max() + resolution, resolution),
            Y=np.arange(data.Y.min(),data.Y.max() + resolution, resolution),
        )    
    return data

def convert(variable, time_res="daily"):
    print(f"converting files for: {time_res} {variable}")

    zarr_resolution = CONFIG['datasets'][time_res]["zarr_resolution"]
    input_path, output_path, var_name = CONFIG['datasets'][time_res]['vars'][variable]    
    nc_path = f"{CONFIG['datasets'][time_res]['nc_path']}{input_path}"
    netcdf = list(sorted(Path(nc_path).glob("*.nc")))
    
    data = xr.open_mfdataset(
        netcdf,
        preprocess = partial(set_up_dims, time_res=time_res),
        parallel=False
    )[var_name]
    if zarr_resolution != None:
        print("attempting regrid")    
        data = regridding(data, zarr_resolution)

    data = data.chunk(chunks=CONFIG['datasets'][time_res]['chunks'])
    
    if output_path == None:
        zarr = f"{CONFIG['datasets'][time_res]['zarr_path']}{input_path}"
    else:
        zarr = f"{CONFIG['datasets'][time_res]['zarr_path']}{output_path}"
    
    shutil.rmtree(zarr, ignore_errors=True)
    os.mkdir(zarr)
    
    xr.Dataset().merge(data).to_zarr(
        store = zarr
    )
    
    if not os.access(zarr, os.W_OK | os.X_OK):
        sys.exit("can't write to output directory")
    
    print(f"conversion for {variable} complete.")
    return zarr

temporal_resolutions = [
    "daily",
    "dekadal",
]
for t in temporal_resolutions:
    for v in CONFIG['datasets'][t]['vars']:
        convert(v, time_res=t)

