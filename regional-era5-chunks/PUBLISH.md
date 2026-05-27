# Publishing this folder as its own GitHub repo

```bash
cd regional-era5-chunks
git init
git add .
git commit -m "Initial release: regional ERA5 chunk downloader v0.1.0"
git remote add origin git@github.com:YOUR_ORG/regional-era5-chunks.git
git push -u origin main
```

Before push:

1. Replace `YOUR_ORG` in `README.md` clone URL.
2. Confirm no secrets in `~/.cdsapirc` or local `data/*.nc` (gitignored).
3. Optional: `git tag v0.1.0 && git push --tags`

Monorepo users: this directory can stay nested; CI in `.github/workflows/ci.yml` assumes **this folder is the repository root**.
