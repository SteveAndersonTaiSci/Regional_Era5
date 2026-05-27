"""Offline tests (no network)."""
from __future__ import annotations

import numpy as np

from regional_era5.common import RegionBBox, lat_index_mask, lon_index_mask, parse_bbox


def test_parse_bbox_default():
    b = parse_bbox(None)
    assert b.lat_min == 21.0 and b.lon_max == 123.0


def test_parse_bbox_custom():
    b = parse_bbox([35.0, 36.0, 139.0, 141.0])
    assert b.lat_max == 36.0


def test_lon_index_negative_west():
    lon = np.linspace(0, 360, 1440, endpoint=False)
    m = lon_index_mask(lon, RegionBBox(40, 50, -10, 10))
    assert m.sum() > 0
