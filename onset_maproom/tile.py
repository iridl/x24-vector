import io
import numpy as np
import xarray as xr
import calc
import mercantile
from PIL import Image

# example tile coordinates
# https://a.tile.opentopomap.org/7/78/60.png


def bounds(z, x, y):
  """abstract over mercantile in case we move to morecantile (&c) in the
  future, and reorder the dimensions to be consistent with the order in
  the e.g. opentopomap standard"""
  bounds = mercantile.bounds(x, y, z)
  return bounds.north, bounds.south, bounds.west, bounds.east


def tile_data(da, z, x, y):
  """select the data specific to a single tile"""
  north, south, west, east = bounds(z, x, y)
  subset = da.sel(X=slice(west, east), Y=slice(south, north))
  return subset

def make_image(da, minimum, maximum, fn=None):
  """we're doing the interpolation in PIL which may be less than ideal"""
  prepared = da.clip(min=minimum, max=maximum).fillna(0)
  pixels = prepared / maximum * 256
  im = Image.fromarray(np.uint8(pixels), mode="L")
  im2 = im.resize((256, 256), resample=Image.NEAREST)
  if fn is not None:
    out = fn
    im2.save(out, format="PNG")
  else:
    out = io.BytesIO()
    im2.save(out, format="PNG")
    out.seek(0)
  return out
