from typing import Any, Dict, Tuple, List, Literal, Optional, Union, Callable, Hashable
from typing import NamedTuple
import math
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from collections.abc import Iterable
from scipy import interpolate
import cv2
import psycopg2.extensions
from psycopg2 import sql
from queuepool.psycopg2cm import ConnectionManagerExtended
from queuepool.pool import Pool
import rasterio.features
import rasterio.transform
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.polygon import LinearRing


def init_dbpool(name, config):
    dbpoolConf = config[name]
    dbpool = Pool(
        name=dbpoolConf["name"],
        capacity=dbpoolConf["capacity"],
        maxIdleTime=dbpoolConf["max_idle_time"],
        maxOpenTime=dbpoolConf["max_open_time"],
        maxUsageCount=dbpoolConf["max_usage_count"],
        closeOnException=dbpoolConf["close_on_exception"],
    )
    for i in range(dbpool.capacity):
        dbpool.put(
            ConnectionManagerExtended(
                name=dbpool.name + "-" + str(i),
                autocommit=False,
                isolation_level=psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE,
                dbname=dbpool.name,
                host=dbpoolConf["host"],
                port=dbpoolConf["port"],
                user=dbpoolConf["user"],
                password=dbpoolConf["password"],
            )
        )
    if dbpoolConf["recycle_interval"] is not None:
        dbpool.startRecycler(dbpoolConf["recycle_interval"])
    return dbpool


FuncInterp2d = Callable[[np.ndarray, np.ndarray], np.ndarray]


class DataArrayEntry(NamedTuple):
    name: str
    data_array: xr.DataArray
    interp2d: Optional[FuncInterp2d]
    min_val: Optional[float]
    max_val: Optional[float]
    colormap: Optional[np.ndarray]


class Extent(NamedTuple):
    dim: str
    left: float
    right: float
    point_width: float


class BGR(NamedTuple):
    blue: int
    green: int
    red: int


class BGRA(NamedTuple):
    blue: int
    green: int
    red: int
    alpha: int


class DrawAttrs(NamedTuple):
    line_color: Union[int, BGR, BGRA]
    background_color: Union[int, BGR, BGRA]
    line_thickness: int
    line_type: int  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA


def from_mercator_rad(lats: float) -> float:
    return np.arctan(0.5 * (np.exp(lats) - np.exp(-lats)))


def from_mercator_deg(lats: float) -> float:
    return np.rad2deg(from_mercator_rad(np.deg2rad(lats)))


def to_mercator_rad(lats: float) -> float:
    return np.log(np.tan(np.pi / 4 + lats / 2))


def to_mercator_deg(lats: float) -> float:
    return np.rad2deg(to_mercator_rad(np.deg2rad(lats)))


def create_interp2d(
    da: xr.DataArray, dims: Tuple[str, str]
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    x = da[dims[1]].values
    y = da[dims[0]].values
    z = da.values
    f = interpolate.interp2d(x, y, z, kind="linear", copy=False, bounds_error=False)
    return f


def from_months_since(x, year_since=1960):
    int_x = int(x)
    return datetime.date(
        int_x // 12 + year_since, int_x % 12 + 1, int(30 * (x - int_x) + 1)
    )


from_months_since_v = np.vectorize(from_months_since)


def to_months_since(d, year_since=1960):
    return (d.year - year_since) * 12 + (d.month - 1) + (d.day - 1) / 30.0


def extent(da: xr.DataArray, dim: str, default: Optional[Extent] = None) -> Extent:
    if default is None:
        default = Extent(dim, np.nan, np.nan, np.nan)
    coord = da[dim]
    n = len(coord.values)
    if n == 0:
        res = Extent(dim, np.nan, np.nan, default.point_width)
    elif n == 1:
        res = Extent(
            dim,
            coord.values[0] - default.point_width / 2,
            coord.values[0] + default.point_width / 2,
            default.point_width,
        )
    else:
        point_width = coord.values[1] - coord.values[0]
        res = Extent(
            dim,
            coord.values[0] - point_width / 2,
            coord.values[-1] + point_width / 2,
            point_width,
        )
    return res


def extents(
    da: xr.DataArray, dims: List[str] = None, defaults: Optional[List[Extent]] = None
) -> List[Extent]:
    if dims is None:
        dims = da.dims
    if defaults is None:
        defaults = [Extent(dim, np.nan, np.nan, np.nan) for dim in dims]
    return [extent(da, k, defaults[i]) for i, k in enumerate(dims)]


def g_lon(tx: int, tz: int) -> float:
    return tx * 360 / 2 ** tz - 180


def g_lat(tx: int, tz: int) -> float:
    return tx * 180 / 2 ** tz - 90


def g_lat_3857(ty: int, tz: int) -> float:
    a = math.pi - 2 * math.pi * ty / 2 ** tz
    return np.rad2deg(from_mercator_rad(a))


def tile_extents(g: Callable[[int, int], float], tx: int, tz: int, n: int = 1):
    assert n >= 1 and tz >= 0 and 0 <= tx < 2 ** tz
    a = g(tx, tz)
    for i in range(1, n + 1):
        b = g(tx + i / n, tz)
        yield a, b
        a = b


def sql_key(fields, table=None):
    if table is None:
        res = sql.SQL(", ").join(sql.Identifier(k) for k in fields)
    else:
        res = sql.SQL(", ").join(
            sql.SQL("{table}.{field}").format(
                table=sql.Identifier(table), field=sql.Identifier(k)
            )
            for k in fields
        )
    return res


def produce_bkg_tile(
    background_color: BGRA,
    tile_width: int = 256,
    tile_height: int = 256,
) -> np.ndarray:
    im = np.zeros((tile_height, tile_width, 4), np.uint8) + background_color
    return im


def produce_data_tile(
    interp2d: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tx: int,
    ty: int,
    tz: int,
    tile_width: int = 256,
    tile_height: int = 256,
) -> np.ndarray:
    x = np.fromiter(
        (a + (b - a) / 2.0 for a, b in tile_extents(g_lon, tx, tz, tile_width)),
        np.double,
    )
    y = np.fromiter(
        (a + (b - a) / 2.0 for a, b in tile_extents(g_lat_3857, ty, tz, tile_height)),
        np.double,
    )
    # print("*** produce_data_tile:", tz, tx, ty, x[0], x[-1], y[0], y[-1])
    z = interp2d(x, y)
    return z


def to_multipolygon(p: Union[Polygon, MultiPolygon]) -> MultiPolygon:
    if not isinstance(p, MultiPolygon):
        p = MultiPolygon([p])
    return p


def rasterize_linearring(
    im: np.ndarray,
    ring: LinearRing,
    fxs: Callable[[np.ndarray], np.ndarray] = lambda xs: xs,
    fys: Callable[[np.ndarray], np.ndarray] = lambda ys: ys,
    line_type: int = cv2.LINE_AA,  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA,
    color: Union[int, BGRA] = 255,
    shift: int = 0,
) -> np.ndarray:
    if not ring.is_empty:
        xs, ys = ring.coords.xy
        xs = fxs(np.array(xs)).astype(np.int32)
        ys = fys(np.array(ys)).astype(np.int32)
        pts = np.column_stack((xs, ys))
        pts = pts.reshape((1,) + pts.shape)
        cv2.fillPoly(im, pts, color, line_type, shift)
    return im


def rasterize_multipolygon(
    im: np.ndarray,
    mp: MultiPolygon,
    fxs: Callable[[np.ndarray], np.ndarray] = lambda xs: xs,
    fys: Callable[[np.ndarray], np.ndarray] = lambda ys: ys,
    line_type: int = cv2.LINE_AA,  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA,
    fg_color: Union[int, BGRA] = 255,
    bg_color: Union[int, BGRA] = 0,
    shift: int = 0,
) -> np.ndarray:
    for p in mp:
        if not p.is_empty:
            rasterize_linearring(im, p.exterior, fxs, fys, line_type, fg_color, shift)
            for q in p.interiors:
                rasterize_linearring(im, q, fxs, fys, line_type, bg_color, shift)


def flatten(im_fg: np.ndarray, im_bg: np.ndarray) -> np.ndarray:
    im_fg = im_fg.astype(np.float64) / 255.0
    im_bg = im_bg.astype(np.float64) / 255.0
    c_fg = im_fg[:, :, :3]
    a_fg = im_fg[:, :, 3:]
    c_bg = im_bg[:, :, :3]
    a_bg = im_bg[:, :, 3:]
    a_comp = a_fg + (1.0 - a_fg) * a_bg
    c_comp = (a_fg * c_fg + (1.0 - a_fg) * a_bg * c_bg) / a_comp
    im_comp = np.concatenate((c_comp, a_comp), axis=2) * 255.0
    return im_comp.astype(np.uint8)


def apply_mask(
    im: np.ndarray, mask: np.ndarray, mask_color: BGRA = BGRA(0, 0, 0, 0)
) -> np.ndarray:
    h = im.shape[0]
    w = im.shape[1]
    mask = mask.reshape(mask.shape + (1,)).astype(np.float64) / 255
    mask_color = np.array(mask_color, np.float64).reshape((1, 1, 4))
    im_fg = mask_color * mask
    im_bg = im * (1.0 - mask)
    im_comp = flatten(im_fg, im_bg)
    return im_comp


def produce_shape_tile(
    im: np.ndarray,
    shapes: List[Tuple[MultiPolygon, DrawAttrs]],
    tx: int,
    ty: int,
    tz: int,
    oper: Literal["intersection", "difference"] = "intersection",
) -> np.ndarray:
    tile_height = im.shape[0]
    tile_width = im.shape[1]

    x0, x1 = list(tile_extents(g_lon, tx, tz, 1))[0]
    y0, y1 = list(tile_extents(g_lat_3857, ty, tz, 1))[0]

    x_ratio = tile_width / (x1 - x0)
    y0_mercator = to_mercator_deg(y0)
    y_ratio_mercator = tile_height / (to_mercator_deg(y1) - y0_mercator)

    tile_bounds = (x0, y0, x1, y1)
    tile = MultiPoint([(x0, y0), (x1, y1)]).envelope

    for s, a in shapes:
        mask = np.zeros(im.shape[:2], np.uint8)
        if oper == "difference":
            if tile.intersects(s):
                mp = to_multipolygon(tile.difference(s))
            else:
                mp = to_multipolygon(tile)
        elif oper == "intersection":
            if tile.intersects(s):
                mp = to_multipolygon(tile.intersection(s))
            else:
                continue
        fxs = lambda xs: (xs - x0) * x_ratio
        fys = lambda ys: (to_mercator_deg(ys) - y0_mercator) * y_ratio_mercator
        rasterize_multipolygon(mask, mp, fxs, fys, a.line_type, 255, 0)
        im = apply_mask(im, mask, a.background_color)

    return im


def produce_test_tile(
    im: np.ndarray,
    text: str = "",
    color: BGRA = BGRA(0, 255, 0, 255),
    line_thickness: int = 1,
    line_type: int = cv2.LINE_AA,  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA
) -> np.ndarray:
    h = im.shape[0]
    w = im.shape[1]

    cv2.ellipse(
        im,
        (w // 2, h // 2),
        (w // 3, h // 3),
        0,
        0,
        360,
        color,
        line_thickness,
        lineType=line_type,
    )

    cv2.putText(
        im,
        text,
        (w // 2, h // 2),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=color,
        thickness=1,
        lineType=line_type,
    )

    cv2.rectangle(im, (0, 0), (w, h), color, line_thickness, lineType=line_type)

    cv2.line(im, (0, 0), (w - 1, h - 1), color, line_thickness, lineType=line_type)
    cv2.line(im, (0, h - 1), (w - 1, 0), color, line_thickness, lineType=line_type)

    return im


def correct_coord(da: xr.DataArray, coord_name: str) -> xr.DataArray:
    coord = da[coord_name]
    values = coord.values
    min_val = values[0]
    max_val = values[-1]
    n = len(values)
    point_width = (max_val - min_val) / n
    vs = np.fromiter(
        ((min_val + point_width / 2.0) + i * point_width for i in range(n)), np.double
    )
    return da.assign_coords({coord_name: vs})


def parse_color(s: str) -> BGRA:
    v = int(s)
    return BGRA(v >> 0 & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF, 255)


def parse_color_item(vs: List[BGRA], s: str) -> List[BGRA]:
    if s == "null":
        rs = [BGRA(0, 0, 0, 0)]
    elif s[0] == "[":
        rs = [parse_color(s[1:])]
    elif s[-1] == "]":
        n = int(s[:-1])
        assert 1 < n <= 256 and len(vs) != 0
        rs = [vs[-1]] * (n - 1)
    else:
        rs = [parse_color(s)]
    return vs + rs


def parse_colormap(s: str) -> np.ndarray:
    vs = []
    for x in s[1:-1].split(" "):
        vs = parse_color_item(vs, x)
    # print(
    #     "*** CM cm:",
    #     len(vs),
    #     [f"{v.red:02x}{v.green:02x}{v.blue:02x}{v.alpha:02x}" for v in vs],
    # )
    rs = np.array([vs[int(i / 256.0 * len(vs))] for i in range(0, 256)], np.uint8)
    return rs


def to_dash_colorscale(cm: np.ndarray) -> List[str]:
    cs = []
    for x in cm:
        v = BGRA(*x)
        cs.append(f"#{v.red:02x}{v.green:02x}{v.blue:02x}{v.alpha:02x}")
    return cs


def apply_colormap(im: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    im = im.astype(np.uint8)
    im = cv2.merge(
        [
            cv2.LUT(im, colormap[:, 0]),
            cv2.LUT(im, colormap[:, 1]),
            cv2.LUT(im, colormap[:, 2]),
            cv2.LUT(im, colormap[:, 3]),
        ]
    )
    return im


#
# Functions to deal with spatial averaging
#


def trim_to_bbox(ds, s, lon_name="lon", lat_name="lat"):
    """Given a Dataset and a shape, return the subset of the Dataset that
    intersects the shape's bounding box.
    """
    lon_res = ds[lon_name].values[1] - ds[lon_name].values[0]
    lat_res = ds[lat_name].values[1] - ds[lat_name].values[0]

    lon_min, lat_min, lon_max, lat_max = s.bounds
    # print("*** shape bounds:", lon_min, lat_min, lon_max, lat_max)

    lon_min -= 1 * lon_res
    lon_max += 1 * lon_res
    lat_min -= 1 * lat_res
    lat_max += 1 * lat_res

    return ds.sel(
        {lon_name: slice(lon_min, lon_max), lat_name: slice(lat_min, lat_max)}
    )


def average_over(ds, s, lon_name="lon", lat_name="lat", all_touched=False):
    """Average a Dataset over a shape"""
    lon_res = ds[lon_name].values[1] - ds[lon_name].values[0]
    lat_res = ds[lat_name].values[1] - ds[lat_name].values[0]

    lon_min = ds[lon_name].values[0] - 0.5 * lon_res
    lon_max = ds[lon_name].values[-1] + 0.5 * lon_res
    lat_min = ds[lat_name].values[0] - 0.5 * lat_res
    lat_max = ds[lat_name].values[-1] + 0.5 * lat_res

    lon_size = ds.sizes[lon_name]
    lat_size = ds.sizes[lat_name]

    t = rasterio.transform.Affine(
        (lon_max - lon_min) / lon_size,
        0,
        lon_min,
        0,
        (lat_max - lat_min) / lat_size,
        lat_min,
    )

    r0 = rasterio.features.rasterize(
        s, out_shape=(lat_size, lon_size), transform=t, all_touched=all_touched
    )
    r0 = xr.DataArray(
        r0,
        dims=(lat_name, lon_name),
        coords={lat_name: ds[lat_name], lon_name: ds[lon_name]},
    )
    r = r0 * np.cos(np.deg2rad(ds[lat_name]))

    res = (ds * r).sum([lat_name, lon_name], skipna=True)
    res = res / (~np.isnan(ds) * r).sum([lat_name, lon_name])

    res.name = ds.name
    return res


def average_over_trimmed(ds, s, lon_name="lon", lat_name="lat", all_touched=False):
    ds = trim_to_bbox(ds, s, lon_name=lon_name, lat_name=lat_name)
    res = average_over(
        ds, s, lon_name=lon_name, lat_name=lat_name, all_touched=all_touched
    )
    return res


#
# Functions to deal with periodic dimension (e.g. longitude)
#


def __dim_range(ds, dim, period=360.0):
    c0, c1 = ds[dim].values[0], ds[dim].values[-1]
    d = (period - (c1 - c0)) / 2.0
    c0, c1 = c0 - d, c1 + d
    return c0, c1


def __normalize_vals(v0, vals, period=360.0, right=False):

    vs = vals if isinstance(vals, Iterable) else [vals]

    v1 = v0 + period
    assert v0 <= 0.0 <= v1

    vs = np.mod(vs, period)
    if right:
        vs[vs > v1] -= period
    else:
        vs[vs >= v1] -= period

    vs = vs if isinstance(vals, Iterable) else vs[0]

    return vs


def __normalize_dim(ds, dim, period=360.0):
    """Doesn't copy ds. Make a copy if necessary."""
    c0, c1 = __dim_range(ds, dim, period)
    if c0 > 0.0:
        ds[dim] = ds[dim] - period
    elif c1 < 0.0:
        ds[dim] = ds[dim] + period


def roll_to(ds, dim, val, period=360.0):
    """Rolls the ds to the first dim's label that is greater or equal to
    val, and then makes dim monitonically increasing. Assumes that dim
    is monotonically increasing, covers exactly one period, and overlaps
    val. If val is outside of the dim, this function does nothing.
    """
    a = np.argwhere(ds[dim].values >= val)
    n = a[0, 0] if a.shape[0] != 0 else 0
    if n != 0:
        ds = ds.copy()
        ds = ds.roll(**{dim: -n}, roll_coords=True)
        ds[dim] = xr.where(ds[dim] < val, ds[dim] + period, ds[dim])
        __normalize_dim(ds, dim, period)
    return ds


def sel_periodic(ds, dim, vals, period=360.0):
    """Assumes that dim is monotonically increasing, covers exactly one period, and overlaps 0.0
    Examples: lon: 0..360, -180..180, -90..270, -360..0, etc.
    TODO: change API to match xarray's `sel`
    """
    c0, c1 = __dim_range(ds, dim, period)
    print(f"*** sel_periodic (input): {c0}..{c1}: {vals}")

    if isinstance(vals, slice):
        if vals.step is None or vals.step >= 0:
            s0 = __normalize_vals(c0, vals.start, period)
            s1 = __normalize_vals(c0, vals.stop, period, True)
        else:
            s0 = __normalize_vals(c0, vals.stop, period)
            s1 = __normalize_vals(c0, vals.start, period, True)

        print(f"*** sel_periodic (normalized): {c0}..{c1}: {s0=}, {s1=}")

        if s0 > s1:
            ds = roll_to(ds, dim, s1, period)
            c0, c1 = __dim_range(ds, dim, period)
            s0 = __normalize_vals(c0, s0, period)
            s1 = __normalize_vals(c0, s1, period, True)
            print(f"*** sel_periodic (rolled): {c0}..{c1}: {s0=}, {s1=}")

        if vals.step is None or vals.step >= 0:
            vals = slice(s0, s1, vals.step)
        else:
            vals = slice(s1, s0, vals.step)

        print(f"*** sel_periodic (slice): {c0}..{c1}: {vals}")

    else:
        vals = __normalize_vals(c0, vals, period=period)
        print(f"*** sel_periodic (array): {c0}..{c1}: {vals}")

    ds = ds.sel({dim: vals})

    return ds
