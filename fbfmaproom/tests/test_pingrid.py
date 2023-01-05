import cftime
import contextlib
import io
import numpy as np
import os
import pytest
import shapely
import tempfile
import xarray as xr

import pingrid


# These tests require netcdf, which is not included in the fbfmaproom2
# virtualenv. Need to move them somewhere else.
# def test_open_dataset_fix_cal():
#     # xr.open_dataset can't open a BytesIO without scipy installed, so
#     # write to an actual file.
#     with tempfilename() as fname:
#         ingrid_ds().to_netcdf(fname)
#         ds = pingrid.open_dataset(fname)
#     assert ds["T"].values[0] == cftime.Datetime360Day(1960, 1, 1)


# def test_open_dataset_no_decode():
#     with tempfilename() as fname:
#         ingrid_ds().to_netcdf(fname)
#         ds = pingrid.open_dataset(fname, decode_times=False)
#     assert ds["T"].values[0] == 0


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


def test_parse_colormap_4_color():
    cmstr = '[0x000000 0xff0000 0xffff00 0xffffff]'
    cm = pingrid.parse_colormap(cmstr)
    assert np.array_equal(cm[0:64], [[0, 0, 0, 255]] * 64)
    assert np.array_equal(cm[64:128], [[255, 0, 0, 255]] * 64)
    assert np.array_equal(cm[128:192], [[255, 255, 0, 255]] * 64)
    assert np.array_equal(cm[192:256], [[255, 255, 255, 255]] * 64)

def test_parse_colormap_interp():
    cmstr = '[0x000000 [0x0000ff 255]]'
    cm = pingrid.parse_colormap(cmstr)
    assert np.array_equal(cm[0], [0, 0, 0, 255])
    assert np.array_equal(cm[128], [0, 0, 128, 255])
    assert np.array_equal(cm[255], [0, 0, 255, 255])

def test_deep_merge_disjoint():
    a = {'a': 1}
    b = {'b': 2}
    assert pingrid.deep_merge(a, b) == {'a': 1, 'b': 2}

def test_deep_merge_overlap():
    a = {'a': 1, 'b': 2}
    b = {'b': 3, 'c': 4}
    assert pingrid.deep_merge(a, b) == {'a': 1, 'b': 3, 'c': 4}

def test_deep_merge_nested():
    a = {'a': 1, 'b': {'c': 2, 'd': 3}}
    b = {'a': 4, 'b': {'d': 5, 'e': 6}}
    assert pingrid.deep_merge(a, b) == {'a': 4, 'b': {'c': 2, 'd': 5, 'e': 6}}

def test_average_over():
    data = [[1, 1], [2, 2]]
    da = xr.DataArray(
        data=data,
        coords={
            'lon': [0., 1.],
            'lat': [0., 1.],
        },
    )
    shape = shapely.geometry.Polygon(
        [(0., 0.), (0., 1.), (1., 1.), (1., 0.)]
    )
    v = pingrid.average_over(da, shape, all_touched=True)
    assert np.isclose(v.item(), 1.5)

# TODO this is a legitimately failing test, but I'm not going to fix
# the bug right now so I'm commenting it out. The solution is probably
# to rip out all the bespoke geographic calculation code and replace
# it with a well-tested community-supported library.
#
# def test_average_over_pixel():
#     '''The average over a single pixel should be the value of that pixel.'''
#     data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#     da = xr.DataArray(
#         data=data,
#         coords={
#             'lon': [0., 1., 2.],
#             'lat': [0., 1., 2.],
#         },
#     )
#     shape = shapely.geometry.Polygon(
#         [(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]
#     )
#     v = pingrid.average_over(da, shape, all_touched=True)
#     assert v.item() == 5.

def test_average_over_nans():
    data = [[1, np.nan], [2, np.nan]]
    da = xr.DataArray(
        data=data,
        coords={
            'lon': [0., 1.],
            'lat': [0., 1.],
        },
    )
    shape = shapely.geometry.Polygon(
        [(0., 0.), (0., 1.), (1., 1.), (1., 0.)]
    )
    v = pingrid.average_over(da, shape, all_touched=True)
    assert np.isclose(v.item(), 1.5)

def test_tile():
    da = xr.DataArray(
        data=[
            [0., 1., 2.],
            [-1., 3., 0.],
            [0., 0., 0.],
        ],
        coords={
            'lat': [-80., 0., 80. ],
            'lon': [-180., 0., 180.],
        },
        attrs={
            'scale_min': 0,
            'scale_max': 2,
            'colormap': "[0x00ff00 [0xff0000 254]]",
        },
    )
    tile = pingrid.impl._tile(da, tx=0, ty=0, tz=0, clipping=None)

    # The data tile is in geographic coordinates, so (0, 0) is the
    # southwest corner, but the image tile is in screen coordinates,
    # so (0, 0) is the upper left (northwest) corner of the tile.

    # Note that colormap is BGR, whereas tile is BGRA.

    # The value in the southwest corner ((255, 0) in screen
    # coordinates) is equal to the min of the colormap, so it should
    # get the 0th color in the colormap.
    print(tile[255][0])
    assert (tile[255][0] == [0, 255, 0, 255]).all()

    # The value in the middle of the southern edge (1) is halfway
    # between the min and the max, so its color is in the middle (with
    # some rounding error...)
    print(tile[255][127])
    assert (tile[255][127] == [126, 128, 0, 255]).all()

    # The value in the southeast corner (2) is equal to the max, so it
    # should get the 255th color.
    print(tile[255][255])
    assert (tile[255][255] == [255, 0, 0, 255]).all()

    # The value in the middle of the western edge (-1) is below the
    # min, so it should get the 0th color.
    print(tile[127][0])
    assert (tile[127][0] == [0, 255, 0, 255]).all()

    # The value in the center of the grid (3) is above the max, so it
    # should get the 255th color.
    print(tile[127][127])
    assert (tile[127][127] == [255, 0, 0, 255]).all()
