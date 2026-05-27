# Regional ERA5 Chunks

**Cut a time window and a bounding box out of ERA5, one NetCDF per day, without loading the global archive into RAM.**

CLI tools for hourly **regional reanalysis** as standard NetCDF—usable with xarray, PyTorch, WRF, or any downstream stack. Not a forecast model; not an ECMWF service replacement.

---

## Install

```bash
git clone https://github.com/YOUR_ORG/regional-era5-chunks.git
cd regional-era5-chunks
pip install -e .

# Optional CDS backend
pip install -e ".[cds]"
```

Console commands: `era5-download`, `era5-prepare`, `era5-zarr-smoke`.

Or run modules directly:

```bash
pip install -r requirements.txt
python -m regional_era5.download --help
python -m regional_era5.prepare list
```

---

## Quick start

```bash
# One day, default bundle (standard = RH + dewpoint + TCWV + wind + Z500)
era5-download --preset week1 --backend zarr --max-days 1

# Custom region (Kanto example)
era5-download --start 2023-07-01 --end 2023-07-08 \
  --bbox 35 36 139 141 --out data/kanto_era5_202307.nc

# Inspect
era5-prepare list
era5-prepare describe data/chunks/kanto_era5_202307_chunk000.nc

# Test Zarr connectivity (no write)
era5-zarr-smoke --bbox 35 36 139 141
```

**Full month (resume):**

```bash
era5-download --preset july2023 --backend zarr --resume \
  --out data/taiwan_era5_202307.nc
era5-prepare manifest --stem taiwan_era5_202307
```

**Legacy TokaMind 6-field bundle (no 2 m RH):**

```bash
era5-download --preset week1 --multivar --max-days 1
```

---

## CLI reference

### `era5-download`

| Flag | Description |
|------|-------------|
| `--bbox LAT_MIN LAT_MAX LON_MIN LON_MAX` | Region (default `21 26 119 123`) |
| `--preset` | `week1`, `july2023`, `summer2023`, `q3_2023` |
| `--start` / `--end` | Override dates (`YYYY-MM-DD`) |
| `--var-set` | `standard` (default), `surface`, `wind`, `multivar` |
| `--vars` | Comma-separated ERA5 names (overrides `--var-set`) |
| `--multivar` | Shortcut for `--var-set multivar` (legacy 6 fields) |
| `--skip-vars` | Drop variables from the resolved set |
| `--backend` | `zarr` (ARCO GCS) or `cds` |
| `--resume` | Skip existing daily chunks |
| `--max-days` | Limit days (smoke / debug) |
| `--out` | Stem path; chunks → `data/chunks/<stem>_chunk*.nc` |

### `era5-prepare`

| Subcommand | Description |
|------------|-------------|
| `list` | Inventory `data/chunks/` |
| `check <nc>` | Variables and bundle readiness (`standard` / `multivar` / …) |
| `describe <nc>` | Print dims, coords, variable shapes (terminal) |
| `merge --glob ... --out` | Optional single-file concat |
| `manifest --stem` | Write `*_manifest.json` |
| `stub` | Dev-only: fake multivar from UV |

---

## Layout (four Python modules)

| File | Role |
|------|------|
| `regional_era5/common.py` | Bbox, presets, chunk checks, NetCDF path helpers |
| `regional_era5/download.py` | Zarr/CDS download + `zarr_smoke_main` |
| `regional_era5/prepare.py` | `list` / `check` / `merge` / `manifest` / `stub` |
| `regional_era5/__init__.py` | Version string |

```text
regional-era5-chunks/
  regional_era5/     # ← only these 4 .py files (+ tests/)
  data/chunks/       # user downloads (gitignored)
  tests/test_bbox.py
```

No duplicate `scripts/` layer: use `pip install -e .` → `era5-download` / `era5-prepare` / `era5-zarr-smoke`.

```text
  ARCO Zarr / CDS
        │
        ▼
  era5-download  ──►  data/chunks/*_chunkNNN.nc
        │
        ▼
  era5-prepare   ──►  list | check | merge | manifest
```

---

## Why not `xr.open_zarr().sel()`?

The usual tutorial pattern is:

```python
ds = xr.open_zarr("gs://gcp-public-data-arco-era5/...")
sub = ds.sel(time=when, latitude=slice(...), longitude=slice(...))
```

That builds a **Dask lazy graph over the full ERA5 store**. One careless step—`sub.mean()`, `.values`, `load()`, or even an implicit reduction—and Dask tries to pull **entire global chunks** into RAM. On a server, that often means OOM and a dead process. We hit exactly that failure mode in production.

**This package bypasses xarray’s high-level wrapper** for the Zarr backend and reads through `zarr.open()` with **orthogonal indexing**:

```python
node.oindex[tuple(idx)]   # idx uses precomputed _lat_idx / _lon_idx slices + one time index
```

Implementation: `ZarrDirectReader` in `regional_era5/download.py`.

| Step | What happens |
|------|----------------|
| Open store | `gcsfs` + `zarr.open(..., mode="r")` once per worker |
| Crop grid | `_lat_idx`, `_lon_idx` from `--bbox` (no full lat/lon arrays in memory) |
| Per hour × variable | `oindex` requests only the **raw binary blocks** for that timestep, that variable, and that lat/lon slice |

**Effect:** each HTTP read from Google Cloud is scoped to **one hour, one field, one regional tile**—not a lazy global array. Daily NetCDF chunks (~MB scale) are assembled in memory from those small 2D slices, then written once. Downstream you still use normal `xr.open_dataset()` on the local `.nc` files.

**Safe pattern for exploration (still risky on ARCO):** if you must use xarray on the public Zarr, keep everything lazy, never call `.values` / `.load()` / reductions until you have explicitly chunked and subset to a tiny domain—or use this tool and work from `data/chunks/*.nc` instead.

---

## `era5-prepare merge`: lazy multi-file concat

After download you have dozens of daily NetCDF files (`*_chunk000.nc`, `*_chunk001.nc`, …). Merging them is optional—most workflows read chunks directly—but `era5-prepare merge` stitches them into one file along `time`.

**Default xarray behaviour:** `open_mfdataset()` uses **Dask in the background** to build a **virtual long time series** across files. Until you call `.load()` (or trigger an equivalent materialization), **no chunk data sits in RAM**—only metadata and a lazy task graph.

```python
ds = xr.open_mfdataset(paths, combine="by_coords", compat="override")
ds = ds.sortby("time")
ds.load().to_netcdf(out)   # materialize only at the final write
```

Implementation: `cmd_merge` in `regional_era5/prepare.py`.

| Choice | Why |
|--------|-----|
| `combine="by_coords"` | Concatenate along `time` when each file is one calendar day |
| `compat="override"` | **Physical defence:** skip xarray’s cross-file checks that lat/lon coordinates match to floating-point equality. Adjacent daily chunks from the same bbox are logically identical grids, but tiny float representation differences would force expensive reconcile work. `override` trusts the grid and moves on—saving a lot of CPU on 30–90 file merges. |
| `ds.load()` only at export | One controlled point where memory use spikes; you choose when to pay that cost |

**Caveat:** `merge` is for convenience, not required. Keeping daily chunks avoids a single huge NetCDF and matches how `era5-download` writes data. If you only need a month of training samples, iterate `data/chunks/*.nc` in your loader instead of merging first.

---

## Data & license

- Scripts: **MIT** — see [LICENSE](LICENSE)
- ERA5 data: **not included** — users download under [ECMWF ERA5 terms](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- See [DATA_SOURCES.md](DATA_SOURCES.md) and [NOTICE](NOTICE)

---

## Variable bundles

| `--var-set` | Fields | Notes |
|-------------|--------|-------|
| **standard** (default) | sp, t2m, **2m RH**, **2m dewpoint**, tcwv, z500, u10, v10 | What most users expect for moisture |
| surface | sp, t2m, RH, dewpoint, tcwv | No wind / height |
| wind | u10, v10 | Minimal |
| multivar | sp, t2m, tcwv, z500, u10, v10 | Legacy TokaMind recipe (no RH) |

`total_column_water_vapour` (TCWV) is **column-integrated** water vapour, not 2 m relative humidity.

---

## Limitations

- Longitude antimeridian spans in one `--bbox` are not supported (split into two downloads).
- Default presets target 2023; any ERA5-covered dates work if the Zarr store has them.
- ~7–20 MB/day (standard, small regions); scales with area and variable count.

---

## Related work

Extracted from research tooling for [TokaMind](https://github.com/) downstream recipes. This package is **data-only**; no TokaMind weights required.

---

## Citation

If you use this software, cite **ERA5** (Hersbach et al., 2020) and link this repository. A Zenodo DOI may be added at release.
