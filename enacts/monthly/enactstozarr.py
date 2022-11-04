import os
import shutil
import sys
import numpy as np
import xarray as xr
import datetime as dt
from pathlib import Path
import pingrid

CONFIG = pingrid.load_config(os.environ["MONTHLY_CONFIG"])

if not os.access(CONFIG["data_dir"], os.W_OK | os.X_OK):
    sys.exit("can't write to output directory")

def set_up_dims(xda):
    datestr = Path(xda.encoding["source"]).name.split("_")[2]
    year = int(datestr[0:4])
    month = int(datestr[4:6])
    dekad = int(datestr[6:7])
    day = (dekad - 1) * 10 + 1
    xda = xda.expand_dims(T = [dt.datetime(year, month, day)])
    xda = xda.rename({'Lon': 'X','Lat': 'Y'})
    return xda

def convert(variable):
    netcdf = list(sorted(Path(CONFIG['data_src'][variable]).glob("*.nc")))

    data = xr.open_mfdataset(
        netcdf,
        preprocess = set_up_dims,
        parallel=False
    )#[variable]
    print(data)

    data = data.chunk(chunks=CONFIG['chunks'])

    zarr = f"{CONFIG['data_dir']}/{variable}.zarr"
    shutil.rmtree(zarr, ignore_errors=True)
    os.mkdir(zarr)

    xr.Dataset().merge(data).to_zarr(
        store = zarr
    )

    return zarr


convert("rfe")
convert("tmin")
convert("tmax")
