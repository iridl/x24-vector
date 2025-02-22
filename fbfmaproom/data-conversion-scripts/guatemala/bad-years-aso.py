import cftime
import xarray as xr
years = range(1993, 2023)
bad_years = [y for y in years if y % 2]
