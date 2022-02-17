import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from pathlib import Path
import pyaconf

CONFIG = pyaconf.load(os.environ["CONFIG"])

TMAX_MRG_NC_PATH = CONFIG["tmax_mrg_nc_path"]
TMAX_MRG_ZARR_PATH = CONFIG["tmax_mrg_zarr_path"]

#Read daily files of daily rainfall data
#Concatenate them against added time dim made up from filenames
#Reading only 6 months of data for the sake of saving time for testing

TMAX_MRG_PATH = Path(TMAX_MRG_NC_PATH)
TMAX_MRG_FILE = list(sorted(TMAX_MRG_PATH.glob("*.nc")))


def set_up_dims(xda):
    xda = xda.expand_dims(T = [dt.datetime(
      int(xda.encoding["source"].rpartition("/")[2].partition("tmax_mrg_")[2].partition("_")[0][0:4]),
      int(xda.encoding["source"].rpartition("/")[2].partition("tmax_mrg_")[2].partition("_")[0][4:6]),
      int(xda.encoding["source"].rpartition("/")[2].partition("tmax_mrg_")[2].partition("_")[0][6:8])
    )])
    xda = xda.rename({'Lon': 'X','Lat': 'Y'})
    return xda

tmax_mrg = xr.open_mfdataset(
  TMAX_MRG_FILE,
  preprocess = set_up_dims,
  parallel=False
)

tmax_mrg.to_zarr(
  store = TMAX_MRG_ZARR_PATH
)
