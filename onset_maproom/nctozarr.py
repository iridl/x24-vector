import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from pathlib import Path
import pyaconf

CONFIG = pyaconf.load(os.environ["CONFIG"])

NC_PATH = CONFIG["wat_cap_abs_nc_path"]
ZARR_PATH = CONFIG["zarr_path"]

DATA_PATH = Path(NC_PATH)
DATA_FILE = list(sorted(DATA_PATH.glob("*.nc")))

print(DATA_FILE[0])

data = xr.open_dataset(
  DATA_FILE[0],
)

data.to_zarr(
  store = ZARR_PATH
)
