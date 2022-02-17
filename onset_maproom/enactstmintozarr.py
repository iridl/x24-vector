import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from pathlib import Path
import pyaconf

CONFIG = pyaconf.load(os.environ["CONFIG"])

TMIN_MRG_NC_PATH = CONFIG["tmin_mrg_nc_path"]
TMIN_MRG_ZARR_PATH = CONFIG["tmin_mrg_zarr_path"]

#Read daily files of daily rainfall data
#Concatenate them against added time dim made up from filenames
#Reading only 6 months of data for the sake of saving time for testing

TMIN_MRG_PATH = Path(TMIN_MRG_NC_PATH)
TMIN_MRG_FILE = list(sorted(TMIN_MRG_PATH.glob("*.nc")))


def set_up_dims(xda):
    xda = xda.expand_dims(T = [dt.datetime(
      int(xda.encoding["source"].rpartition("/")[2].partition("tmin_mrg_")[2].partition("_")[0][0:4]),
      int(xda.encoding["source"].rpartition("/")[2].partition("tmin_mrg_")[2].partition("_")[0][4:6]),
      int(xda.encoding["source"].rpartition("/")[2].partition("tmin_mrg_")[2].partition("_")[0][6:8])
    )])
    xda = xda.rename({'Lon': 'X','Lat': 'Y'})
    return xda

tmin_mrg = xr.open_mfdataset(
  TMIN_MRG_FILE,
  preprocess = set_up_dims,
  parallel=False
)

tmin_mrg.to_zarr(
  store = TMIN_MRG_ZARR_PATH
)
