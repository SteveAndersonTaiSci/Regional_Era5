"""Shared constants and helpers for ERA5 download / prepare scripts."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

# Package repo root (regional-era5-chunks/)
ROOT = Path(__file__).resolve().parents[1]

# Default crop (Taiwan); override with --bbox LAT_MIN LAT_MAX LON_MIN LON_MAX
DEFAULT_BBOX = (21.0, 26.0, 119.0, 123.0)

# ERA5 variable names (ARCO Zarr & NetCDF output). Same strings for CDS API requests.
WIND_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind"]

SURFACE_VARS = [
    "surface_pressure",
    "2m_temperature",
    "2m_relative_humidity",
    "2m_dewpoint_temperature",
    "total_column_water_vapour",
]

PRESSURE_LEVEL_VARS = frozenset({"geopotential"})

# Legacy 6-field set (TokaMind cross-field recipe); no RH/dewpoint.
MULTIVAR_VARS = [
    "surface_pressure",
    "2m_temperature",
    "total_column_water_vapour",
    "geopotential",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# Default for open-source users: surface moisture + wind + 500 hPa height.
STANDARD_VARS = [
    *SURFACE_VARS,
    "geopotential",
    *WIND_VARS,
]

VARIABLE_SETS: dict[str, list[str]] = {
    "wind": list(WIND_VARS),
    "surface": list(SURFACE_VARS),
    "multivar": list(MULTIVAR_VARS),
    "standard": list(STANDARD_VARS),
}

DEFAULT_VAR_SET = "standard"

# Units & short descriptions (for describe / docs).
VARIABLE_META: dict[str, dict[str, str]] = {
    "surface_pressure": {"unit": "Pa", "label": "Surface pressure"},
    "2m_temperature": {"unit": "K", "label": "2 m temperature"},
    "2m_relative_humidity": {"unit": "%", "label": "2 m relative humidity"},
    "2m_dewpoint_temperature": {"unit": "K", "label": "2 m dewpoint"},
    "total_column_water_vapour": {"unit": "kg m⁻²", "label": "Total column water vapour"},
    "geopotential": {"unit": "m² s⁻²", "label": "Geopotential (configurable level)"},
    "10m_u_component_of_wind": {"unit": "m s⁻¹", "label": "10 m U wind"},
    "10m_v_component_of_wind": {"unit": "m s⁻¹", "label": "10 m V wind"},
}

CDS_SINGLE_LEVEL_REQUEST: dict[str, str] = {k: k for k in SURFACE_VARS + list(WIND_VARS)}

# CDS NetCDF short names → pipeline names
CDS_NC_ALIASES: dict[str, str] = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "sp": "surface_pressure",
    "tcwv": "total_column_water_vapour",
    "r": "2m_relative_humidity",
    "r2": "2m_relative_humidity",
    "d2m": "2m_dewpoint_temperature",
    "z": "geopotential",
    "gh": "geopotential",
    "geopotential": "geopotential",
}

PRESETS: dict[str, tuple[str, str]] = {
    "week1": ("2023-07-01", "2023-07-07"),
    "july2023": ("2023-07-01", "2023-08-01"),
    "summer2023": ("2023-06-01", "2023-09-01"),
    "q3_2023": ("2023-07-01", "2023-10-01"),
}

ZARR_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_CHUNK_DIR = ROOT / "data" / "chunks"


@dataclass(frozen=True)
class RegionBBox:
    """Regional crop in degrees (WGS84). Longitude may be [-180, 180] or [0, 360]."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat_min < self.lat_max <= 90.0):
            raise ValueError(
                f"invalid latitude span: lat_min={self.lat_min} lat_max={self.lat_max} "
                "(require -90 <= lat_min < lat_max <= 90)"
            )
        if self.lon_min == self.lon_max:
            raise ValueError("lon_min and lon_max must differ")

    @property
    def lon_min_360(self) -> float:
        return _lon_to_360(self.lon_min)

    @property
    def lon_max_360(self) -> float:
        return _lon_to_360(self.lon_max)

    def cds_area(self) -> list[float]:
        """CDS API area: [North, West, South, East]."""
        lon_w, lon_e = self.lon_min, self.lon_max
        if self.lon_min_360 > self.lon_max_360:
            raise ValueError(
                "bbox crosses the antimeridian in 0–360 longitude; "
                "split into two downloads or narrow the box"
            )
        return [float(self.lat_max), float(lon_w), float(self.lat_min), float(lon_e)]

    def as_dict(self) -> dict[str, float]:
        return {
            "lat_min": float(self.lat_min),
            "lat_max": float(self.lat_max),
            "lon_min": float(self.lon_min),
            "lon_max": float(self.lon_max),
        }

    def __str__(self) -> str:
        return (
            f"lat [{self.lat_min}, {self.lat_max}] "
            f"lon [{self.lon_min}, {self.lon_max}]"
        )


def _lon_to_360(lon: float) -> float:
    x = float(lon) % 360.0
    return x if x >= 0.0 else x + 360.0


def parse_bbox(values: list[float] | None) -> RegionBBox:
    if values is None:
        return RegionBBox(*DEFAULT_BBOX)
    if len(values) != 4:
        raise ValueError("--bbox requires exactly 4 numbers: LAT_MIN LAT_MAX LON_MIN LON_MAX")
    return RegionBBox(
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )


def all_known_variables() -> frozenset[str]:
    out: set[str] = set()
    for vs in VARIABLE_SETS.values():
        out.update(vs)
    return frozenset(out)


def parse_var_list(
    *,
    var_set: str | None = None,
    vars_csv: str | None = None,
    multivar_flag: bool = False,
    skip_csv: str = "",
) -> tuple[list[str], str]:
    """Resolve download variable list and return (vars, set_name)."""
    if vars_csv and vars_csv.strip():
        names = [v.strip() for v in vars_csv.split(",") if v.strip()]
        unknown = [v for v in names if v not in all_known_variables()]
        if unknown:
            raise ValueError(
                f"unknown --vars: {unknown}\n  known: {sorted(all_known_variables())}"
            )
        chosen = names
        label = "custom"
    elif multivar_flag:
        chosen = list(VARIABLE_SETS["multivar"])
        label = "multivar"
    else:
        key = (var_set or DEFAULT_VAR_SET).strip().lower()
        if key not in VARIABLE_SETS:
            raise ValueError(f"unknown --var-set {key!r}; choose from {sorted(VARIABLE_SETS)}")
        chosen = list(VARIABLE_SETS[key])
        label = key
    skip = {v.strip() for v in skip_csv.split(",") if v.strip()}
    out = [v for v in chosen if v not in skip]
    if not out:
        raise ValueError("empty variable list after --skip-vars")
    return out, label


def add_var_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--var-set",
        choices=sorted(VARIABLE_SETS.keys()),
        default=None,
        help=f"Variable bundle (default: {DEFAULT_VAR_SET}). "
        "standard = surface RH/dewpoint/TCWV + wind + 500 hPa Z",
    )
    g.add_argument(
        "--vars",
        default=None,
        help="Comma-separated ERA5 variable names (overrides --var-set)",
    )
    parser.add_argument(
        "--multivar",
        action="store_true",
        help="Shortcut for --var-set multivar (legacy 6 fields, no RH)",
    )
    parser.add_argument(
        "--skip-vars",
        default="",
        help="Comma-separated variables to drop from the resolved set",
    )


def add_bbox_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        default=None,
        help=(
            "Regional crop in degrees (default: 21 26 119 123). "
            "Longitude may be negative (e.g. -10 10 -80 -70) or 0–360."
        ),
    )


def lat_index_mask(lat: np.ndarray, bbox: RegionBBox) -> np.ndarray:
    lat = np.asarray(lat, dtype=np.float64)
    return (lat >= bbox.lat_min) & (lat <= bbox.lat_max)


def lon_index_mask(lon: np.ndarray, bbox: RegionBBox) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64)
    l0, l1 = bbox.lon_min_360, bbox.lon_max_360
    grid = np.where(lon < 0.0, (lon + 360.0) % 360.0, lon % 360.0)
    if l0 <= l1:
        return (grid >= l0) & (grid <= l1)
    # Antimeridian wrap in 0–360 space
    return (grid >= l0) | (grid <= l1)


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def chunk_ok(path: Path, min_hours: int = 1) -> bool:
    if not path.is_file() or path.stat().st_size < 2048:
        return False
    try:
        with xr.open_dataset(path) as ds:
            return int(ds.sizes.get("time", 0)) >= min_hours
    except Exception:
        return False


def chunk_glob_for_stem(stem: str) -> str:
    return f"data/chunks/{stem}_chunk*.nc"


# --- NetCDF I/O (used by prepare.check) ---

REQUIRED_NC_VARS_UV = tuple(WIND_VARS)
REQUIRED_NC_VARS_MULTIVAR = tuple(MULTIVAR_VARS)
REQUIRED_NC_VARS_STANDARD = tuple(STANDARD_VARS)

SP_VAR = "surface_pressure"
T2M_VAR = "2m_temperature"
RH2M_VAR = "2m_relative_humidity"
D2M_VAR = "2m_dewpoint_temperature"
TCWV_VAR = "total_column_water_vapour"
Z_VAR = "geopotential"
U_VAR = "10m_u_component_of_wind"
V_VAR = "10m_v_component_of_wind"


def resolve_nc_path(nc: str | Path) -> Path:
    raw = Path(nc)
    if raw.is_file():
        return raw.resolve()
    for base in (Path.cwd(), ROOT / "data", ROOT):
        for p in (base / raw.name, base / raw):
            if p.is_file():
                return p.resolve()
    raise FileNotFoundError(
        f"NetCDF not found: {nc}\n  cwd={Path.cwd()}\n  package_root={ROOT}"
    )


def detect_mode(nc_path: Path) -> str:
    with xr.open_dataset(nc_path) as ds:
        names = set(ds.data_vars)
    if all(v in names for v in REQUIRED_NC_VARS_STANDARD):
        return "standard"
    if all(v in names for v in REQUIRED_NC_VARS_MULTIVAR):
        return "multivar"
    if names and names <= set(WIND_VARS):
        return "wind"
    if names:
        return "partial"
    raise ValueError(f"{nc_path} has no data variables")
