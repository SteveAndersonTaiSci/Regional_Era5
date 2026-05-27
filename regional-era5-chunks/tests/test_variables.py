"""Variable sets and NetCDF profile detection (offline)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from regional_era5.common import (
    MULTIVAR_VARS,
    STANDARD_VARS,
    VARIABLE_SETS,
    WIND_VARS,
    detect_mode,
    parse_var_list,
)


def test_parse_var_list_default_standard():
    vars_, label = parse_var_list()
    assert label == "standard"
    assert vars_ == STANDARD_VARS


def test_parse_var_list_multivar_flag():
    vars_, label = parse_var_list(multivar_flag=True)
    assert label == "multivar"
    assert vars_ == MULTIVAR_VARS


def test_parse_var_list_custom():
    vars_, label = parse_var_list(vars_csv="surface_pressure,2m_temperature")
    assert label == "custom"
    assert vars_ == ["surface_pressure", "2m_temperature"]


def test_variable_sets_keys():
    assert set(VARIABLE_SETS) == {"wind", "surface", "multivar", "standard"}


def _write_mini_nc(path: Path, var_names: list[str]) -> None:
    t = np.array(["2023-07-01T00:00:00", "2023-07-01T01:00:00"], dtype="datetime64[ns]")
    lat = np.linspace(21.0, 26.0, 3)
    lon = np.linspace(119.0, 123.0, 4)
    data = np.zeros((2, 3, 4), dtype=np.float32)
    ds = xr.Dataset(
        {v: (("time", "latitude", "longitude"), data) for v in var_names},
        coords={"time": t, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path)


def test_detect_mode_standard(tmp_path: Path):
    p = tmp_path / "std.nc"
    _write_mini_nc(p, STANDARD_VARS)
    assert detect_mode(p) == "standard"


def test_detect_mode_multivar(tmp_path: Path):
    p = tmp_path / "mv.nc"
    _write_mini_nc(p, MULTIVAR_VARS)
    assert detect_mode(p) == "multivar"


def test_detect_mode_wind(tmp_path: Path):
    p = tmp_path / "wind.nc"
    _write_mini_nc(p, WIND_VARS)
    assert detect_mode(p) == "wind"
