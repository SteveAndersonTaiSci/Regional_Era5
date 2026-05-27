#!/usr/bin/env python3
"""
ERA5 data preparation (after download) — no network fetch.

Subcommands: list, check, merge, manifest, stub
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import xarray as xr

from regional_era5.common import (
    D2M_VAR,
    RH2M_VAR,
    ROOT,
    SP_VAR,
    T2M_VAR,
    TCWV_VAR,
    U_VAR,
    VARIABLE_META,
    VARIABLE_SETS,
    Z_VAR,
    chunk_glob_for_stem,
    chunk_ok,
    detect_mode,
    resolve_nc_path,
    resolve_path,
)


def cmd_list(_args: argparse.Namespace) -> None:
    data = ROOT / "data"
    chunks = sorted((data / "chunks").glob("*.nc")) if (data / "chunks").is_dir() else []
    singles = sorted(data.glob("*.nc")) if data.is_dir() else []
    print(f"data/ singles ({len(singles)}):")
    for p in singles:
        print(f"  {p.name}")
    print(f"data/chunks/ ({len(chunks)}):")
    for p in chunks[:20]:
        print(f"  {p.name}")
    if len(chunks) > 20:
        print(f"  ... +{len(chunks) - 20} more")

    manifests = list((data / "chunks").glob("*_manifest.json")) if (data / "chunks").is_dir() else []
    for m in manifests:
        print(f"\nmanifest {m.name}:")
        print(json.dumps(json.loads(m.read_text(encoding="utf-8")), indent=2))

    if chunks:
        print(f"\nChunks on disk: {len(chunks)} daily file(s)")
        print("  Example: era5-prepare check data/chunks/<stem>_chunk000.nc")
    else:
        print("\nNo ERA5 chunks yet. Download:")
        print("  era5-download --preset week1 --max-days 1")
        raise SystemExit(1)


def cmd_describe(args: argparse.Namespace) -> None:
    """Print/visual summary of one chunk file (dims, vars, coords)."""
    path = resolve_nc_path(args.nc_file)
    with xr.open_dataset(path) as ds:
        print(f"\n=== ERA5 chunk format: {path.name} ===\n")
        print(f"  path: {path.resolve()}")
        print(f"  dims: {dict(ds.sizes)}")
        if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
            t0 = str(ds["time"].values[0])[:19]
            t1 = str(ds["time"].values[-1])[:19]
            print(f"  time: {t0}  →  {t1}  ({ds.sizes['time']} steps)")
        if "latitude" in ds.coords:
            lat = ds["latitude"].values
            print(f"  latitude: n={lat.size}  range=[{float(lat.min()):.2f}, {float(lat.max()):.2f}]")
        if "longitude" in ds.coords:
            lon = ds["longitude"].values
            print(f"  longitude: n={lon.size}  range=[{float(lon.min()):.2f}, {float(lon.max()):.2f}]")
        print("\n  variables (each: time × latitude × longitude):")
        for v in ds.data_vars:
            sh = tuple(ds[v].shape)
            meta = VARIABLE_META.get(str(v), {})
            unit = meta.get("unit", "")
            label = meta.get("label", v)
            extra = f"  ({label}" + (f", {unit}" if unit else "") + ")"
            if v == "geopotential" and "level_hpa" in ds[v].attrs:
                extra += f"  level_hpa={ds[v].attrs['level_hpa']}"
            print(f"    - {v}: shape={sh}{extra}")
        print(f"\n  detected profile: {detect_mode(path)}")
        print("\n  xarray:  ds = xr.open_dataset(...); ds['2m_temperature'].isel(time=0).plot()")
        print()


def cmd_check(args: argparse.Namespace) -> None:
    path = resolve_nc_path(args.nc_file)
    with xr.open_dataset(path) as ds:
        names = set(ds.data_vars)
        print(f"path: {path}")
        print(f"time: {ds.sizes.get('time')}")
        print(f"vars ({len(names)}): {sorted(names)}")
        for v in ds.data_vars:
            print(f"  {v}: {tuple(ds[v].dims)} {tuple(ds[v].shape)}")

    mode = detect_mode(path)
    print(f"detected_mode: {mode}")
    if mode == "standard":
        print("bundle ready: yes (standard — RH, dewpoint, TCWV, wind, Z)")
    elif mode == "multivar":
        print("bundle ready: yes (legacy multivar — no 2 m RH/dewpoint)")
    elif mode == "wind":
        print("bundle ready: wind only — use era5-download (default --var-set standard)")
    else:
        print("bundle ready: partial — closest presets:")
        for set_name, required in sorted(VARIABLE_SETS.items()):
            missing = [v for v in required if v not in names]
            if not missing:
                print(f"  matches preset: {set_name}")
            else:
                print(f"  {set_name}: missing {missing}")


def cmd_merge(args: argparse.Namespace) -> None:
    if args.glob:
        paths = sorted(ROOT.glob(args.glob))
    else:
        paths = [resolve_path(x) for x in args.inputs]
    if not paths:
        raise SystemExit(f"No input files (glob={args.glob!r})")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if not chunk_ok(path, min_hours=1):
            print(f"[prepare] warning: {path.name} may be incomplete", flush=True)

    ds = xr.open_mfdataset(paths, combine="by_coords", compat="override")
    if "time" in ds.dims:
        ds = ds.sortby("time")
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.load().to_netcdf(out)
    print(f"[prepare] merge saved {out.resolve()} | T={int(ds.sizes.get('time', 0))}", flush=True)


def cmd_manifest(args: argparse.Namespace) -> None:
    chunk_dir = resolve_path(args.chunk_dir)
    stem = args.stem
    pattern = f"{stem}_chunk*.nc"
    paths = sorted(chunk_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No chunks matching {chunk_dir}/{pattern}")

    total_h = 0
    rel_chunks: list[str] = []
    for p in paths:
        rel_chunks.append(str(p.relative_to(ROOT)))
        if chunk_ok(p):
            with xr.open_dataset(p) as ds:
                total_h += int(ds.sizes.get("time", 0))

    manifest = {
        "backend": args.backend,
        "chunks": rel_chunks,
        "total_hours": total_h,
        "nc_glob": chunk_glob_for_stem(stem),
    }
    out = chunk_dir / f"{stem}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[prepare] manifest {out.name} | chunks={len(paths)} | T≈{total_h}", flush=True)


def cmd_stub(args: argparse.Namespace) -> None:
    inp = resolve_path(args.in_nc)
    out = resolve_path(args.out_nc)
    with xr.open_dataset(inp) as ds:
        u = ds[U_VAR]
        shape = u.shape
        rng = np.random.default_rng(int(args.seed))
        sp = 101325.0 + rng.standard_normal(shape).astype(np.float32) * 200.0
        t2m = 290.0 + rng.standard_normal(shape).astype(np.float32) * 2.0
        rh = 70.0 + rng.standard_normal(shape).astype(np.float32) * 10.0
        d2m = 288.0 + rng.standard_normal(shape).astype(np.float32) * 2.0
        tcwv = 30.0 + rng.standard_normal(shape).astype(np.float32) * 5.0
        z = 50000.0 + rng.standard_normal(shape).astype(np.float32) * 100.0
        out_ds = ds.copy()
        out_ds[SP_VAR] = (("time", "latitude", "longitude"), sp)
        out_ds[T2M_VAR] = (("time", "latitude", "longitude"), t2m)
        out_ds[RH2M_VAR] = (("time", "latitude", "longitude"), rh)
        out_ds[D2M_VAR] = (("time", "latitude", "longitude"), d2m)
        out_ds[TCWV_VAR] = (("time", "latitude", "longitude"), tcwv)
        out_ds[Z_VAR] = (("time", "latitude", "longitude"), z)
        out_ds[Z_VAR].attrs["level_hpa"] = 500
        out.parent.mkdir(parents=True, exist_ok=True)
        out_ds.to_netcdf(out)
    print(f"[prepare] stub wrote {out.resolve()} (smoke only — use era5-download for real data)")


def main() -> None:
    p = argparse.ArgumentParser(description="ERA5 data preparation (post-download).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="List data/ chunks and manifests")
    sp_list.set_defaults(func=cmd_list)

    sp_check = sub.add_parser("check", help="Inspect one NetCDF")
    sp_check.add_argument("nc_file", nargs="?", default="data/chunks/example_chunk000.nc")
    sp_check.set_defaults(func=cmd_check)

    sp_desc = sub.add_parser("describe", help="Human-readable format summary (dims/vars)")
    sp_desc.add_argument("nc_file", nargs="?", default="data/chunks/taiwan_era5_202307_chunk000.nc")
    sp_desc.set_defaults(func=cmd_describe)

    sp_merge = sub.add_parser("merge", help="Concatenate chunks along time")
    sp_merge.add_argument("inputs", nargs="*", help="Input .nc files (if no --glob)")
    sp_merge.add_argument("--glob", default=None)
    sp_merge.add_argument("--out", required=True)
    sp_merge.set_defaults(func=cmd_merge)

    sp_man = sub.add_parser("manifest", help="Write manifest JSON from chunk files")
    sp_man.add_argument("--stem", required=True)
    sp_man.add_argument("--chunk-dir", default="data/chunks")
    sp_man.add_argument("--backend", default="zarr")
    sp_man.set_defaults(func=cmd_manifest)

    sp_stub = sub.add_parser("stub", help="UV-only → stub standard fields (dev smoke)")
    sp_stub.add_argument("--in-nc", required=True)
    sp_stub.add_argument("--out-nc", required=True)
    sp_stub.add_argument("--seed", type=int, default=42)
    sp_stub.set_defaults(func=cmd_stub)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
