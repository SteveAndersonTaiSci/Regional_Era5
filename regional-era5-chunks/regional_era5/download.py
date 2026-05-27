#!/usr/bin/env python3
"""
Download regional ERA5 NetCDF (daily chunks under data/chunks/).

Use --bbox LAT_MIN LAT_MAX LON_MIN LON_MAX for any crop (default: Taiwan).

Data prep (merge, check, manifest) → scripts/era5_prepare.py

Backends:
  zarr  — ARCO ERA5 on GCS (ZarrDirectReader)
  cds   — Copernicus CDS API (~/.cdsapirc)

Usage:
  python scripts/era5_download.py --preset july2023 --multivar --backend zarr --max-days 1
  python scripts/era5_download.py --preset july2023 --multivar --backend cds
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from regional_era5.common import (
    CDS_NC_ALIASES,
    CDS_SINGLE_LEVEL_REQUEST,
    PRESSURE_LEVEL_VARS,
    PRESETS,
    ROOT,
    RegionBBox,
    ZARR_STORE,
    add_bbox_arg,
    add_var_args,
    chunk_ok,
    lat_index_mask,
    lon_index_mask,
    parse_bbox,
    parse_var_list,
)

ZARR = ZARR_STORE

def _decode_zarr_time(tnode, mapper=None) -> pd.DatetimeIndex:
    """Decode ARCO/CF time coordinate (raw ints are NOT wall-clock without units)."""
    raw = np.asarray(tnode[:])
    attrs = dict(getattr(tnode, "attrs", {}) or {})
    units = attrs.get("units")
    calendar = attrs.get("calendar", "standard")

    if not units and mapper is not None:
        try:
            tcoord = xr.open_zarr(mapper, consolidated=True, decode_times=True)["time"]
            return pd.DatetimeIndex(tcoord.values).tz_localize(None)
        except Exception:
            pass

    if not units:
        vmax = float(np.max(raw)) if raw.size else 0.0
        # ARCO ERA5 hourly counts since 1900-01-01 (~1.08e6 for year 2023).
        if 5e5 < vmax < 2e7:
            units = "hours since 1900-01-01 00:00:00.0"
            print(f"[zarr] inferred time units={units!r} (raw max={vmax:.0f})", flush=True)

    if units:
        decoded = xr.coding.times.decode_cf_datetime(
            raw, units, calendar, use_cftime=False
        )
        return pd.DatetimeIndex(decoded).tz_localize(None)

    if np.issubdtype(raw.dtype, np.integer) and int(np.max(raw)) > 10**14:
        return pd.to_datetime(raw, unit="ns", utc=True).tz_convert(None)
    return pd.DatetimeIndex(pd.to_datetime(raw)).tz_localize(None)


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def _parse_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


class ZarrDirectReader:
    """Read one timestep / one variable via Zarr oindex (dimension-order aware)."""

    def __init__(self, z_level_hpa: int = 500, *, bbox: RegionBBox | None = None):
        self.z_level_hpa = int(z_level_hpa)
        self.bbox = bbox if bbox is not None else parse_bbox(None)
        self._root = None
        self._fs = None
        self._times: pd.DatetimeIndex | None = None
        self._lat_idx: np.ndarray | None = None
        self._lon_idx: np.ndarray | None = None
        self._z_idx: int | None = None
        self._lats: np.ndarray | None = None
        self._lons: np.ndarray | None = None
        self._lock = threading.Lock()

    def _open(self) -> None:
        if self._root is not None:
            return
        import gcsfs
        import zarr

        print("[zarr] opening store (sync gcsfs)...", flush=True)
        # Sync client avoids aiohttp loop teardown crashes on exit / heavy loops.
        self._fs = gcsfs.GCSFileSystem(token="anon")
        mapper = self._fs.get_mapper(ZARR)
        self._root = zarr.open(mapper, mode="r")
        lat = np.asarray(self._root["latitude"][:])
        lon = np.asarray(self._root["longitude"][:])
        self._lat_idx = np.where(lat_index_mask(lat, self.bbox))[0]
        self._lon_idx = np.where(lon_index_mask(lon, self.bbox))[0]
        if len(self._lat_idx) == 0 or len(self._lon_idx) == 0:
            raise RuntimeError(
                f"bbox {self.bbox} matches no grid points "
                f"(lat hits={len(self._lat_idx)} lon hits={len(self._lon_idx)}; "
                f"store lat [{float(lat.min()):.2f}, {float(lat.max()):.2f}] "
                f"lon [{float(lon.min()):.2f}, {float(lon.max()):.2f}])"
            )
        self._lats = lat[self._lat_idx]
        self._lons = lon[self._lon_idx]
        tnode = self._root["time"]
        self._times = _decode_zarr_time(tnode, mapper)
        if "level" in self._root:
            lev = np.asarray(self._root["level"][:], dtype=np.float64)
            self._z_idx = int(np.argmin(np.abs(lev - float(self.z_level_hpa))))
            print(f"[zarr] geopotential level idx={self._z_idx} ({lev[self._z_idx]} hPa)", flush=True)
        sp_dims = self._root["surface_pressure"].attrs.get("_ARRAY_DIMENSIONS")
        tunits = dict(tnode.attrs).get("units", "?")
        print(
            f"[zarr] bbox {self.bbox} | "
            f"grid {len(self._lat_idx)}x{len(self._lon_idx)} | "
            f"time steps={len(self._times)} | range={self._times[0]} .. {self._times[-1]} | "
            f"time_units={tunits!r} | sp_dims={sp_dims} | rss={_rss_mb():.0f}MB",
            flush=True,
        )

    def close(self) -> None:
        self._root = None
        self._fs = None
        gc.collect()

    def _time_index(self, when: pd.Timestamp) -> int:
        assert self._times is not None
        when = pd.Timestamp(when).tz_localize(None)
        if when < self._times[0] or when > self._times[-1]:
            raise ValueError(
                f"requested {when} outside zarr time range "
                f"{self._times[0]} .. {self._times[-1]}"
            )
        hit = np.where(self._times == when)[0]
        if len(hit):
            return int(hit[0])
        pos = int(self._times.searchsorted(when))
        if pos >= len(self._times):
            pos = len(self._times) - 1
        elif pos > 0:
            before = self._times[pos - 1]
            after = self._times[pos]
            if abs(before - when) <= abs(after - when):
                pos -= 1
        return pos

    def _oindex_2d(self, var: str, ti: int) -> np.ndarray:
        assert self._root is not None and self._lat_idx is not None and self._lon_idx is not None
        node = self._root[var]
        dims = tuple(node.attrs.get("_ARRAY_DIMENSIONS", ("time", "latitude", "longitude")))
        lat_sl = slice(int(self._lat_idx[0]), int(self._lat_idx[-1]) + 1)
        lon_sl = slice(int(self._lon_idx[0]), int(self._lon_idx[-1]) + 1)
        idx: list = []
        for d in dims:
            if d == "time":
                idx.append(ti)
            elif d in ("latitude", "lat"):
                idx.append(lat_sl)
            elif d in ("longitude", "lon"):
                idx.append(lon_sl)
            elif d in ("level", "isobaricInhPa", "pressure_level"):
                if self._z_idx is None:
                    raise RuntimeError(f"{var} needs level dim but store has none")
                idx.append(self._z_idx)
            else:
                raise ValueError(f"unsupported dim {d!r} in {dims} for {var}")
        data = np.asarray(node.oindex[tuple(idx)], dtype=np.float32)
        # Ensure output [latitude, longitude].
        if "latitude" in dims and "longitude" in dims:
            if dims.index("longitude") < dims.index("latitude"):
                data = data.T
        if data.ndim != 2:
            raise ValueError(f"{var} expected 2D, got {data.shape} dims={dims}")
        return data

    def read_2d(self, var: str, when: pd.Timestamp) -> np.ndarray:
        with self._lock:
            self._open()
            ti = self._time_index(when)
            arr = self._oindex_2d(var, ti)
            finite = np.isfinite(arr)
            if not finite.any():
                t_at = self._times[ti] if self._times is not None else "?"
                raise ValueError(
                    f"{var} @ {when} all NaN — time_idx={ti} decoded_time={t_at}"
                )
            if not finite.all():
                arr = np.where(finite, arr, np.nanmean(arr))
            return arr

    @property
    def lats(self) -> np.ndarray:
        self._open()
        assert self._lats is not None
        return self._lats

    @property
    def lons(self) -> np.ndarray:
        self._open()
        assert self._lons is not None
        return self._lons


def _fetch_var_slice(reader: ZarrDirectReader, var: str, when: pd.Timestamp) -> tuple[str, np.ndarray]:
    return var, reader.read_2d(var, when)


def _download_day_zarr(
    reader: ZarrDirectReader,
    day_start: datetime,
    out: Path,
    var_list: list[str],
    z_level_hpa: int,
    workers: int = 1,
) -> int:
    day_end = day_start + timedelta(days=1)
    times = pd.date_range(day_start, day_end, freq="h", inclusive="left")
    stacks: dict[str, list[np.ndarray]] = {v: [] for v in var_list}
    n_workers = max(1, min(int(workers), len(var_list)))

    for hi, when in enumerate(times):
        print(
            f"[download]   hour {hi + 1}/{len(times)} {when} | "
            f"workers={n_workers} | rss={_rss_mb():.0f}MB",
            flush=True,
        )
        if n_workers <= 1:
            for var in var_list:
                stacks[var].append(reader.read_2d(var, when))
        else:
            hour_slices: dict[str, np.ndarray] = {}
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = [pool.submit(_fetch_var_slice, reader, v, when) for v in var_list]
                for fu in as_completed(futs):
                    var, arr = fu.result()
                    hour_slices[var] = arr
            for var in var_list:
                stacks[var].append(hour_slices[var])
        gc.collect()

    data_vars = {
        v: (("time", "latitude", "longitude"), np.stack(stacks[v], axis=0)) for v in var_list
    }
    out_ds = xr.Dataset(
        data_vars,
        coords={"time": times.values, "latitude": reader.lats, "longitude": reader.lons},
    )
    if "geopotential" in out_ds:
        out_ds["geopotential"].attrs["level_hpa"] = z_level_hpa

    tmp = out.with_suffix(".tmp.nc")
    if tmp.is_file():
        tmp.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(tmp)
    tmp.replace(out)
    return int(out_ds.sizes["time"])


def _cds_dataset_from_files(
    sl_path: Path, pl_path: Path | None, var_list: list[str], z_level_hpa: int
) -> xr.Dataset:
    """Map CDS NetCDF short names to pipeline names; keep only requested vars."""
    out_vars: dict[str, xr.DataArray] = {}
    with xr.open_dataset(sl_path) as sfc:
        for vname, da in sfc.data_vars.items():
            canon = CDS_NC_ALIASES.get(str(vname), str(vname))
            if canon in var_list and canon not in PRESSURE_LEVEL_VARS:
                out_vars[canon] = da
    if "geopotential" in var_list:
        if pl_path is None or not pl_path.is_file():
            raise FileNotFoundError("pressure-level file required for geopotential")
        with xr.open_dataset(pl_path) as pl:
            zname = "z" if "z" in pl else "geopotential" if "geopotential" in pl else None
            if zname is None:
                zname = str(next(iter(pl.data_vars)))
            out_vars["geopotential"] = pl[zname].squeeze()
            out_vars["geopotential"].attrs["level_hpa"] = z_level_hpa
    missing = [v for v in var_list if v not in out_vars]
    if missing:
        raise ValueError(f"CDS files missing requested variables: {missing}")
    ds = xr.Dataset(out_vars)
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds.sortby("time")


def _download_month_cds(
    year: int,
    month: int,
    out_dir: Path,
    z_level_hpa: int,
    *,
    bbox: RegionBBox,
    var_list: list[str],
) -> list[Path]:
    """Download one calendar month via CDS; returns list of daily-like files (single month nc)."""
    try:
        import cdsapi
    except ImportError as e:
        raise SystemExit("pip install cdsapi  &&  configure ~/.cdsapirc") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    sl_path = out_dir / f"cds_sfc_{year}{month:02d}.nc"
    pl_path = out_dir / f"cds_pl{z_level_hpa}_{year}{month:02d}.nc"

    area = bbox.cds_area()
    print(f"[cds] area (N,W,S,E)={area}", flush=True)
    days = [f"{d:02d}" for d in range(1, 32)]
    hours = [f"{h:02d}:00" for h in range(24)]
    c = cdsapi.Client()

    sl_request = [
        CDS_SINGLE_LEVEL_REQUEST[v]
        for v in var_list
        if v in CDS_SINGLE_LEVEL_REQUEST
    ]
    if not sl_request:
        raise ValueError(f"no single-level variables in request: {var_list}")

    if not sl_path.is_file():
        print(f"[cds] single-levels ({len(sl_request)} vars) -> {sl_path.name}", flush=True)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": sl_request,
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": hours,
                "area": area,
                "format": "netcdf",
            },
            str(sl_path),
        )

    need_pl = "geopotential" in var_list
    if need_pl and not pl_path.is_file():
        print(f"[cds] pressure-level geopotential -> {pl_path.name}", flush=True)
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": "geopotential",
                "pressure_level": str(z_level_hpa),
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": hours,
                "area": area,
                "format": "netcdf",
            },
            str(pl_path),
        )

    ds = _cds_dataset_from_files(
        sl_path, pl_path if need_pl else None, var_list, z_level_hpa
    )

    chunk_paths: list[Path] = []
    days = sorted({pd.Timestamp(t).normalize() for t in pd.to_datetime(ds["time"].values)})
    for di, day in enumerate(days):
        sub = ds.sel(time=slice(day, day + pd.Timedelta(days=1)))
        if int(sub.sizes.get("time", 0)) == 0:
            continue
        p = out_dir / f"cds_{year}{month:02d}_day{di:03d}.nc"
        sub.to_netcdf(p)
        chunk_paths.append(p)
        print(f"[cds] wrote {p.name} T={sub.sizes['time']}", flush=True)
    return chunk_paths


def _day_job(payload: tuple[str, str, list[str], int, int, tuple[float, float, float, float]]) -> tuple[str, int]:
    """ProcessPool worker: one calendar day -> one chunk file."""
    day_iso, chunk_out_str, var_list, z_level_hpa, workers, bbox_t = payload
    reader = ZarrDirectReader(z_level_hpa, bbox=RegionBBox(*bbox_t))
    try:
        n = _download_day_zarr(
            reader,
            datetime.fromisoformat(day_iso),
            Path(chunk_out_str),
            var_list,
            z_level_hpa,
            workers=workers,
        )
        return chunk_out_str, n
    finally:
        reader.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="july2023")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default="data/taiwan_era5_202307.nc")
    p.add_argument("--z-level-hpa", type=int, default=500)
    p.add_argument("--backend", choices=["zarr", "cds"], default="zarr")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="parallel GCS reads per hour (one thread per variable, max len(var_list))",
    )
    p.add_argument(
        "--day-workers",
        type=int,
        default=1,
        help="parallel calendar days (separate processes; each opens its own Zarr reader)",
    )
    add_bbox_arg(p)
    add_var_args(p)
    args = p.parse_args()

    bbox = parse_bbox(args.bbox)
    var_list, var_set = parse_var_list(
        var_set=args.var_set,
        vars_csv=args.vars,
        multivar_flag=args.multivar,
        skip_csv=args.skip_vars,
    )
    preset_start, preset_end = PRESETS[args.preset]
    start = args.start or preset_start
    end = args.end or preset_end

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    chunk_dir = out.parent / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[download] bbox={bbox} | {start} .. {end} | backend={args.backend} | "
        f"var_set={var_set} | n_vars={len(var_list)}",
        flush=True,
    )

    if args.backend == "cds":
        y, m = int(start[:4]), int(start[5:7])
        paths = _download_month_cds(
            y, m, chunk_dir, args.z_level_hpa, bbox=bbox, var_list=var_list
        )
        for i, src in enumerate(sorted(paths)):
            dst = chunk_dir / f"{out.stem}_chunk{i:03d}{out.suffix}"
            if dst.is_file() and args.resume and not args.force:
                src.unlink(missing_ok=True)
                continue
            src.replace(dst)
        manifest = {
            "backend": "cds",
            "bbox": bbox.as_dict(),
            "start": start,
            "end": end,
            "var_set": var_set,
            "variables": var_list,
            "z_level_hpa": int(args.z_level_hpa),
            "chunks": [
                str((chunk_dir / f"{out.stem}_chunk{i:03d}{out.suffix}").relative_to(ROOT))
                for i in range(len(paths))
            ],
            "nc_glob": f"data/chunks/{out.stem}_chunk*.nc",
        }
        (chunk_dir / f"{out.stem}_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print("[download] CDS done. Run: era5-prepare list", flush=True)
        return

    t0 = _parse_ymd(start)
    t1 = _parse_ymd(end)
    total = 0
    idx = 0
    cur = t0
    max_days = int(args.max_days)
    var_workers = max(1, int(args.workers))
    day_workers = max(1, int(args.day_workers))
    pending: list[tuple[datetime, Path, int]] = []

    while cur < t1:
        if max_days > 0 and idx >= max_days:
            break
        nxt = min(cur + timedelta(days=1), t1)
        chunk_out = chunk_dir / f"{out.stem}_chunk{idx:03d}{out.suffix}"
        expect_h = int((nxt - cur).total_seconds() // 3600)

        if args.resume and not args.force and chunk_ok(chunk_out, min_hours=max(1, expect_h - 1)):
            with xr.open_dataset(chunk_out) as ds:
                t_part = int(ds.sizes.get("time", 0))
            print(f"[download] resume skip {chunk_out.name} T={t_part}", flush=True)
            total += t_part
        else:
            pending.append((cur, chunk_out, expect_h))
        cur = nxt
        idx += 1

    if pending:
        print(
            f"[download] fetch {len(pending)} day(s) | var_workers={var_workers} | "
            f"day_workers={day_workers}",
            flush=True,
        )
        if day_workers <= 1:
            reader = ZarrDirectReader(args.z_level_hpa, bbox=bbox)
            try:
                for cur, chunk_out, expect_h in pending:
                    if chunk_out.is_file():
                        chunk_out.unlink()
                    t_part = _download_day_zarr(
                        reader, cur, chunk_out, var_list, args.z_level_hpa, workers=var_workers
                    )
                    total += t_part
                    print(
                        f"[download] day {cur.date()} OK T={t_part} | rss={_rss_mb():.0f}MB",
                        flush=True,
                    )
            finally:
                reader.close()
        else:
            bbox_t = (
                bbox.lat_min,
                bbox.lat_max,
                bbox.lon_min,
                bbox.lon_max,
            )
            jobs = [
                (
                    cur.date().isoformat(),
                    str(chunk_out),
                    var_list,
                    args.z_level_hpa,
                    var_workers,
                    bbox_t,
                )
                for cur, chunk_out, _ in pending
            ]
            for _, chunk_out, _ in pending:
                if chunk_out.is_file():
                    chunk_out.unlink()
            with ProcessPoolExecutor(max_workers=min(day_workers, len(jobs))) as pool:
                for chunk_path, t_part in pool.map(_day_job, jobs):
                    total += t_part
                    print(f"[download] done {Path(chunk_path).name} T={t_part}", flush=True)

    manifest = {
        "backend": args.backend,
        "bbox": bbox.as_dict(),
        "start": start,
        "end": end,
        "var_set": var_set,
        "variables": var_list,
        "z_level_hpa": int(args.z_level_hpa),
        "chunks": [str((chunk_dir / f"{out.stem}_chunk{i:03d}{out.suffix}").relative_to(ROOT)) for i in range(idx)],
        "total_hours": total,
        "nc_glob": f"data/chunks/{out.stem}_chunk*.nc",
    }
    (chunk_dir / f"{out.stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[download] done T≈{total} h | {idx} day files\n"
        f"  era5-prepare list\n"
        f"  era5-prepare check data/chunks/{out.stem}_chunk000.nc",
        flush=True,
    )


def zarr_smoke_main() -> None:
    """Quick ARCO Zarr read test (no NetCDF write). Entry: era5-zarr-smoke."""
    p = argparse.ArgumentParser(description="Smoke-test ARCO Zarr regional read.")
    p.add_argument("--time", default="2023-07-01 00:00:00")
    p.add_argument("--z-level-hpa", type=int, default=500)
    add_bbox_arg(p)
    args = p.parse_args()
    bbox = parse_bbox(args.bbox)

    r = ZarrDirectReader(args.z_level_hpa, bbox=bbox)
    when = pd.Timestamp(args.time)
    sp = r.read_2d("surface_pressure", when)
    print(f"bbox={bbox}", flush=True)
    print(f"surface_pressure shape={sp.shape} mean={sp.mean():.2f} Pa", flush=True)
    u = r.read_2d("10m_u_component_of_wind", when)
    print(f"u10 mean={u.mean():.4f} m/s", flush=True)
    r.close()
    print("OK — Zarr reads look valid", flush=True)


if __name__ == "__main__":
    main()
