import os
import shutil
import sys
import numpy as np
import xarray as xr
import datetime as dt
from pathlib import Path
import pingrid
import pandas as pd


CONFIG = pingrid.load_config(os.environ["CONFIG"])

ZARR_RESOLUTION = CONFIG["zarr_resolution"]

def set_up_dims(xda):
    
    datestr = Path(xda.encoding["source"]).name.split("_")[2]
    xda = xda.expand_dims(T = [dt.datetime(
      int(datestr[0:4]),
      int(datestr[4:6]),
      int(datestr[6:8])
    )])
    xda = xda.rename({'Lon': 'X','Lat': 'Y'})
    
    return xda

def convert(variable):
    print(f"converting files for: {variable}")
    var_name = CONFIG['vars'][variable][2]    
    
    nc_path = f"{CONFIG['nc_path']}{CONFIG['vars'][variable][0]}"
    
    netcdf = list(sorted(Path(nc_path).glob("*.nc")))
    
    data = xr.open_mfdataset(
        netcdf,
        preprocess = set_up_dims,
        parallel=False
    )[var_name]
    
    if not np.isclose(data['X'][1] - data['X'][0], ZARR_RESOLUTION):
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
            X=np.arange(data.X.min(), data.X.max() + ZARR_RESOLUTION, ZARR_RESOLUTION),
            Y=np.arange(data.Y.min(),data.Y.max() + ZARR_RESOLUTION, ZARR_RESOLUTION),
        )    
    
    data = data.chunk(chunks=CONFIG['chunks'])
    
    zarr = f"{CONFIG['zarr_path']}{CONFIG['vars'][variable][1]}"
    
    shutil.rmtree(zarr, ignore_errors=True)
    os.mkdir(zarr)
    
    xr.Dataset().merge(data).to_zarr(
        store = zarr
    )
    
    if not os.access(zarr, os.W_OK | os.X_OK):
        sys.exit("can't write to output directory")
    
    print(f"conversion for {variable} complete.")
    return zarr

for i in CONFIG['vars']:
    convert(i)
