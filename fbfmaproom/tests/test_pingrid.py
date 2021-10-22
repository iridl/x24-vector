import cftime
import contextlib
import io
import os
import pytest
import tempfile
import xarray as xr

import pingrid


def test_open_dataset_fix_cal():
    # xr.open_dataset can't open a BytesIO without scipy installed, so
    # write to an actual file.
    with tempfilename() as fname:
        ingrid_ds().to_netcdf(fname)
        ds = pingrid.open_dataset(fname)
    assert ds["T"].values[0] == cftime.Datetime360Day(1960, 1, 1)


def test_open_dataset_no_decode():
    with tempfilename() as fname:
        ingrid_ds().to_netcdf(fname)
        ds = pingrid.open_dataset(fname, decode_times=False)
    assert ds["T"].values[0] == 0


def ingrid_ds():
    """Returns a small xr.Dataset with a metadata error that mimics Ingrid's output"""
    ds = xr.Dataset(
        coords={"T": range(3)}
    )
    ds["T"].attrs = {
        "calendar": "360",  # non-CF-compliant metadata produced by Ingrid
        "units": "months since 1960-01-01",
    }
    return ds


# https://stackoverflow.com/questions/3924117/how-to-use-tempfile-namedtemporaryfile-in-python
@contextlib.contextmanager
def tempfilename(suffix=None):
  """Context that introduces a temporary file.

  Creates a temporary file, yields its name, and upon context exit, deletes it.
  (In contrast, tempfile.NamedTemporaryFile() provides a 'file' object and
  deletes the file as soon as that file object is closed, so the temporary file
  cannot be safely re-opened by another library or process.)

  Args:
    suffix: desired filename extension (e.g. '.mp4').

  Yields:
    The name of the temporary file.
  """
  import tempfile
  try:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_name = f.name
    f.close()
    yield tmp_name
  finally:
    os.unlink(tmp_name)
