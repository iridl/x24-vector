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

RR_MRG_NC_PATH = CONFIG["onset"]["rr_mrg_nc_path"]
RR_MRG_ZARR_PATH = CONFIG["onset"]["rr_mrg_zarr_path"]
RESOLUTION = CONFIG["onset"]["rr_mrg_resolution"]
CHUNKS = CONFIG["onset"]["rr_mrg_chunks"]

#Read daily files of daily rainfall data
#Concatenate them against added time dim made up from filenames

RR_MRG_PATH = Path(RR_MRG_NC_PATH)
RR_MRG_FILE = list(sorted(RR_MRG_PATH.glob("*.nc")))

def set_up_dims(xda):
    datestr = Path(xda.encoding["source"]).name.split("_")[2]
    xda = xda.expand_dims(T = [dt.datetime(
      int(datestr[0:4]),
      int(datestr[4:6]),
      int(datestr[6:8])
    )])
    xda = xda.rename({'Lon': 'X','Lat': 'Y'})
    return xda

rr_mrg = xr.open_mfdataset(
    RR_MRG_FILE,
    preprocess = set_up_dims,
    parallel=False
).precip

if not np.isclose(rr_mrg['X'][1] - rr_mrg['X'][0], RESOLUTION):
    # TODO this method of regridding is inaccurate because it pretends
    # that (X, Y) define a Euclidian space. In reality, grid cells
    # farther from the equator cover less area and thus should be
    # weighted less heavily. Also, consider using conservative
    # interpolation instead of bilinear, since when going to a much
    # coarser resoution, bilinear discards a lot of information. See [1],
    # and look into xESMF.
    #
    # [1] https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/regridding-overview
    rr_mrg = rr_mrg.interp(
        X=np.arange(rr_mrg.X.min(), rr_mrg.X.max() + RESOLUTION, RESOLUTION),
        Y=np.arange(rr_mrg.Y.min(), rr_mrg.Y.max() + RESOLUTION, RESOLUTION),
    )

rr_mrg = rr_mrg.chunk(chunks=CHUNKS)

xr.Dataset().merge(rr_mrg).to_zarr(
  store = RR_MRG_ZARR_PATH
)

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
