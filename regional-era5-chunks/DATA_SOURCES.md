# Data sources

## ARCO ERA5 (default backend: Zarr)

- **Store:** `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- **Registry:** https://registry.opendata.aws/arco-era5/
- **Access:** anonymous read via `gcsfs` (public GCS)

## Copernicus CDS (optional backend)

- **API:** https://cds.climate.copernicus.eu/
- **Setup:** `pip install 'regional-era5-chunks[cds]'` and `~/.cdsapirc`
- **Datasets:** `reanalysis-era5-single-levels`, `reanalysis-era5-pressure-levels`

## Output files

Scripts write **derivative** regional NetCDF chunks under `data/chunks/`. You are responsible for compliance with ECMWF/CDS terms for the underlying reanalysis.
