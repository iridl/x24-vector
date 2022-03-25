import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from pathlib import Path
import pyaconf

CONFIG = pyaconf.load(os.environ["CONFIG"])

RR_MRG_NC_PATH = CONFIG["rr_mrg_nc_path"]
RR_MRG_ZARR_PATH = CONFIG["rr_mrg_zarr_path"]

#Read daily files of daily rainfall data
#Concatenate them against added time dim made up from filenames

RR_MRG_PATH = Path(RR_MRG_NC_PATH)
RR_MRG_FILE = list(sorted(RR_MRG_PATH.glob("*.nc")))

RESOLUTION = .25


def set_up_dims(xda):
    xda = xda.expand_dims(T = [dt.datetime(
      int(xda.encoding["source"].partition("rr_mrg_")[2].partition("_")[0][0:4]),
      int(xda.encoding["source"].partition("rr_mrg_")[2].partition("_")[0][4:6]),
      int(xda.encoding["source"].partition("rr_mrg_")[2].partition("_")[0][6:8])
    )])
    xda = xda.rename({'Lon': 'X','Lat': 'Y'})
    return xda

rr_mrg = xr.open_mfdataset(
    RR_MRG_FILE,
    preprocess = set_up_dims,
    parallel=False
).precip.interp(
    X=np.arange(33, 48 + RESOLUTION, RESOLUTION),
    Y=np.arange(3, 15 + RESOLUTION, RESOLUTION),
).chunk(chunks={'T': 600, 'Y': 24, 'X': 30})

xr.Dataset().merge(rr_mrg).to_zarr(
  store = RR_MRG_ZARR_PATH
)

