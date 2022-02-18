import io
import numpy as np
import xarray as xr
import calc
import mercantile
from PIL import Image

# example tile coordinates
# https://a.tile.opentopomap.org/7/78/60.png

# this math isn't quite right but works well enough for now
def generate_rainbow_palette():
  pal = []
  for g in range(64):
    pal.append(0)   #R
    pal.append(g*4) #G
    pal.append(255) #B
  for b in range(64)[::-1]:
    pal.append(0)   #R
    pal.append(255) #G
    pal.append(b*4) #B
  for r in range(64):
    pal.append(r*4) #R
    pal.append(255) #G
    pal.append(0)   #B
  for g in range(64)[::-1]:
    pal.append(255) #R
    pal.append(g*4) #G
    pal.append(0)   #B
  return pal

palette = generate_rainbow_palette()


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
  im = Image.fromarray(np.uint8(pixels), mode="P")
  im.putpalette(palette)
  im = im.resize((256, 256), resample=Image.NEAREST)
  if fn is not None:
    out = fn
  else:
    out = io.BytesIO()
  im.save(out, format="PNG")
  if fn is None:
    out.seek(0) # won't work unless we seek to 0
  return out
