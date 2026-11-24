# Data and offline installation guide

This guide prepares the Katabatic copy for use on a computer that will have no
internet connection. The project copy contains Python source only; you must
provide the input datasets and a prepared Python environment separately.

## 1. Prepare the Python environment

The supported baseline is Python 3.11 with packages from conda-forge. On a
computer with internet access, install Miniforge or Conda, then create the
environment:

```powershell
conda env create -f .\environment.yml
conda activate katabatic-offline
```

For a disconnected target computer, package the complete environment on a
connected computer with the same operating system and CPU architecture:

```powershell
conda install -n base -c conda-forge conda-pack
conda-pack -n katabatic-offline -o .\katabatic-offline-win-64.tar.gz
```

Transfer the archive to the target computer. Extract it into a new Conda
environment directory, then run `conda-unpack` from that extracted
environment. For example, replace the path below with the `envs` directory
under your Conda installation:

```powershell
$envPath = "$env:USERPROFILE\miniforge3\envs\katabatic-offline"
New-Item -ItemType Directory -Force -Path $envPath | Out-Null
tar -xzf .\katabatic-offline-win-64.tar.gz -C $envPath
conda activate $envPath
conda-unpack
```

Use the same Conda distribution and platform on both computers. Once
activated, the root-level `launch_katabatic.ps1` invokes that environment's
Python interpreter. If policy settings block a local PowerShell script, allow
it for the current PowerShell process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 2. Obtain and install CARRA2 input data

### Obtain the data

The project uses the C3S Arctic Regional Reanalysis second generation
(CARRA2), available from the official
[Copernicus Climate Data Store (CDS) sub-daily dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-pan-carra).
The dataset is free after creating a CDS account and accepting the terms of
use on its download page. It provides 3-hourly analyses at 2.5 km resolution.
Read the [CARRA2 Data User Guide](https://confluence.ecmwf.int/spaces/CKB/pages/583771203/Copernicus+pan-Arctic+Regional+Reanalysis+CARRA2+Data+User+Guide)
for details.

On a computer with internet access:

1. Sign in to CDS and open the sub-daily CARRA2 dataset above.
2. In the **Download** form, select single-level data and the **analysis**
   product. Select the June, July, and August dates and the available
   3-hourly analysis times for each study year.
3. Request these variables together for each year/month:
   - 2 m temperature
   - 10 m eastward (u) wind component
   - 10 m northward (v) wind component
   - Orography
   - Land-sea mask (needed when generating basin labels)
4. Select the Iceland area. The CDS API uses `[north, west, south, east]`
   order; `[69.5, -27.5, 60.5, -7.5]` is a practical starting bounding box
   around Iceland with surrounding terrain. Use the same area for every
   variable, year, and static field so the resulting grids match.
5. Select **NetCDF** output because the Python readers in this copy consume
   `.nc` files. CDS documents NetCDF conversion as experimental; check that
   the downloaded files open correctly before requesting the full period.
6. For large requests, download one year/month at a time. The CDS page's
   **Show API request code** button generates a request matching the current
   form. Follow the [CDS API setup guide](https://cds.climate.copernicus.eu/how-to-api)
   if using its API: install `cdsapi` on the connected computer, configure
   your personal CDS token there, and run the generated request. Do not put
   API tokens in this project or share them.

The main detector requires the NetCDF variables `t2m`, `u10`, and `v10`;
orography must be available as `orog`. Confirm the actual variable names in
the downloaded NetCDF files. If names differ, these readers will not silently
rename them. The 3 detector variables should be requested together so the
monthly data check can verify them consistently.

**Important:** the current `crop_iceland()` reader does not crop files during
loading. Limit the CDS request to the Iceland area above (or prepare an
equivalent Iceland subset before running the detector); downloading the full
2869-by-2869 pan-Arctic grid is unnecessary and can make processing very slow.
The glacier mask and basin arrays must use the exact same area and grid as
the downloaded CARRA2 files.

### Install the files

Transfer the downloaded files to this project and organize them as follows:

```text
Data/
  Brut_netcdf_files/
    land_mask.nc
    2022/
      CARRA2_June/
        data_2.nc        # NetCDF file containing the static orog field
        carra_june.nc    # NetCDF file(s) containing t2m, u10 and v10
        ...
    2023/
      CARRA2_June/
      CARRA2_July/
      CARRA2_August/
    2024/
      CARRA2_June/
      CARRA2_July/
      CARRA2_August/
```

Create one year folder per available year and a `CARRA2_<Month>` folder for
each required month. Put the `.nc` files for that month in the corresponding
folder. The land-sea mask is used by basin generation; save its NetCDF file
as `Data/Brut_netcdf_files/land_mask.nc`. If that file contains `lsm` rather
than `land_mask`, it should contain only that data variable so the loader
does not select a different variable by mistake.

The default reference-orography lookup uses
`2022/CARRA2_June/data_2.nc`. Put a file containing `orog` there, or select a
different reference with `--orog-year`, `--orog-month`, and
`--orog-file-index`. Keep a copy of the exact download request (years,
variables, area, product, and format) alongside your research records.

The main command defaults to at least five complete years. Use
`--expected-min-years 1` for a deliberately smaller test dataset. To check
the downloaded files before processing, run the quality checker. It checks
folder/year consistency, required variables, timestamps, and missing values
in the Iceland region:

```powershell
Push-Location ".\Scripts Pythons"
python -m katabatic.data_quality_check --path "..\Data\Brut_netcdf_files" --months June July August --required-vars t2m u10 v10 --fail-on-issues
python -m katabatic.list_available_years --path "..\Data\Brut_netcdf_files" --months June July August --required-vars t2m u10 v10
Pop-Location
```

The offline copy does not download CARRA2 data. Download it while connected,
then transfer the NetCDF files and request metadata to the disconnected
computer.

## 3. Install or generate derived arrays

The main classification requires basin labels at `Data/NPY/basins.npy`.
Basin generation also requires a glacier mask at
`Data/NPY/glacier_mask.npy` and the CARRA2 orography and land-sea mask
described above. The generated `glacier_labels.npy` and `steepest_ways.npy`
are intermediate outputs.

### Preferred: transfer the project's existing mask

If you have the full project workspace, copy its `Data/NPY/glacier_mask.npy`
into this copy's `Data/NPY/`. Only copy `basins.npy` or `glacier_labels.npy`
from that workspace if they were generated from the exact same CARRA area,
grid, and orography. Otherwise, regenerate them locally as described below.

```powershell
$source = "..\Data\NPY"
$target = ".\Data\NPY"
New-Item -ItemType Directory -Force -Path $target | Out-Null
Copy-Item (Join-Path $source "glacier_mask.npy") $target -Force
```

### Alternative: generate a mask from glacier outlines

`glacier_mask.npy` is a project-derived raster, not a file distributed by
CARRA2 or ICE-GAWS. The project rasterizes Iceland glacier outlines onto the
CARRA grid. The outlines used by the original full project come from the
[Randolph Glacier Inventory (RGI) 6.0](https://doi.org/10.7265/N5-RGI-60);
select region **06 (Iceland)** and download the complete Shapefile set
(`.shp`, `.shx`, `.dbf`, `.prj`, and any accompanying files).

The `geo/` preparation scripts are intentionally not included in this
Katabatic-only copy. If you need to create the mask rather than transfer it,
use the full source project: place the RGI files under
`Data/GEO/RGI/`, make the reference CARRA grid available at
`Data/Brut_netcdf_files/2025/CARRA2_June/`, activate that project's `geo`
environment, and run `Scripts Pythons/launch_geo.ps1 -Task raster`. The
current rasterizer uses the 2025 June grid as its reference. Copy the
resulting `Data/NPY/glacier_mask.npy` into this offline copy. If your chosen
CARRA subset differs from that reference grid, regenerate the mask on the
chosen grid; do not reuse an array with different dimensions or coordinates.

### Generate basin labels

Once the glacier mask, orography, and land-sea mask are installed, generate
basin labels from this copy:

```powershell
Push-Location ".\Scripts Pythons"
python -m katabatic.katabatic_basins_generator
Pop-Location
```

The generator writes `glacier_labels.npy`, `steepest_ways.npy`, and
`basins.npy` under `Data/NPY/`. Inspect the saved arrays and generated figures
before running the main classification. ICE-GAWS station data are a separate
input, described in Section 4.

## 4. Install ICE-GAWS data for station validation

The station-validation tools use the processed ICE-GAWS Level 3 (L3) CSV
files. Obtain them on a computer with internet access from the
[ICE-GAWS v2 data page](https://data.lv.is/ICE-GAWS-v2/). A versioned
2024 release is also available on
[Zenodo (DOI 10.5281/zenodo.15794332)](https://doi.org/10.5281/zenodo.15794332).
The [ICE-GAWS processing repository](https://github.com/andrigunn/ICE-GAWS-Processing)
describes the archive layout and file naming. Use one documented data release
and one consistent L3 gap-filling product for the analysis; record its
version and retrieval date for reproducibility.

Extract the archive and copy these items into `Data/ICE-GAWS/`:

- `ICE-GAWS-location-summary.csv` from the archive's `locations/` folder.
- The L3 station CSV files, named like
  `ICE-GAWS_<station>_L3_<product>_<year>.csv`. For example:
  `ICE-GAWS_Br01_L3_ICEBOX_2023.csv`.

The code searches for L3 CSV files directly inside `Data/ICE-GAWS/`, so copy
them out of any station-specific subfolders into that directory. Do not copy
L0/L1/L2 files or `.dat` files for this workflow. The station summary must
contain `station`, `lat_ave`, `lon_ave`, and `ele_ave` columns. The L3 files
must include a `Time` column and the default detector variables `t_2m`,
`ws_4m`, and `wd_4m`. Check the CSV headers before processing; use the
detector command's column options if your release uses different names.

If the original project data folder is next to this offline copy, this
PowerShell example copies the summary and L3 CSVs. Otherwise, set `$source` to
the extracted archive's ICE-GAWS folder:

```powershell
$source = "..\Data\ICE-GAWS"
$target = ".\Data\ICE-GAWS"
New-Item -ItemType Directory -Force -Path $target | Out-Null
Copy-Item (Join-Path $source "ICE-GAWS-location-summary.csv") $target -Force
Get-ChildItem -LiteralPath $source -Filter "ICE-GAWS_*_L3*.csv" -File -Recurse |
    Copy-Item -Destination $target -Force
```

For comparison, ICE-GAWS station-years must overlap the CARRA2 JJA years
being analysed. The mapping and detector JSON files are generated locally;
they are outputs, not files to download. Before running validation, make sure
`Data/NPY/basins.npy` and the CARRA2 reference-orography file described above
are installed as well.

### Build station mappings and run validation

Run these commands from the project root with the `katabatic-offline`
environment activated. First generate the station-to-CARRA pixel/basin
mapping and the station detector results:

```powershell
Push-Location ".\Scripts Pythons"
python -m katabatic.build_icegaws_carra_pixel_map
Pop-Location
```

Then run the full station time-series comparison and save the comparison
figures:

```powershell
Push-Location ".\Scripts Pythons"
python -m katabatic.compare_eval_icegaws --full-evaluation
python -m katabatic.compare_carra_icegaws --no-show --per-station
Pop-Location
```

The figure-producing commands use Cartopy. If you have already disconnected
the computer, complete Section 5 first so the local map cache is available.

The first command writes station mapping and detector JSON files under
`Data/ICE-GAWS/`. The full evaluation writes validation metrics and station
tables there, with diagnostic figures under `Figures/`. The map comparison
also writes station-overlay figures. These operations can take time and only
include stations and years with usable overlapping observations and CARRA2
data. Review the generated station counts and coverage before interpreting
the metrics.

If setup fails, inspect the available command-line options without processing
data:

```powershell
Push-Location ".\Scripts Pythons"
python -m katabatic.build_icegaws_carra_pixel_map --help
python -m katabatic.compare_eval_icegaws --help
Pop-Location
```

Please cite the ICE-GAWS dataset and its associated paper when reporting
results. The processing repository points to Gunnarsson et al. (2025),
*Surface climatology of glaciers in Iceland: Icelandic Glaciers Automatic
Weather Station Network (ICE-GAWS)*,
[doi:10.1029/2024JD043216](https://doi.org/10.1029/2024JD043216).

## 5. Prepare map assets before disconnecting

Map plotting uses Cartopy's Natural Earth land, ocean, and 10 m coastline
features. Cartopy may try to download these assets the first time they are
used, so prepare the cache while connected to the internet:

```powershell
conda activate katabatic-offline
python ".\Scripts Pythons\prepare_offline_maps.py"
```

The script stores the assets under `Data/cartopy/`. Transfer the complete
`Data/cartopy/` folder to the disconnected computer at the same relative
location. Do not clear it after setup. If you keep map assets elsewhere, set
`CARTOPY_DATA_DIR` to that directory before launching.

## 6. Verify the setup and run

From the project root, activate the environment and run:

```powershell
conda activate katabatic-offline
.\launch_katabatic.ps1 --help
.\launch_katabatic.ps1 --mode yearly --save-figures --no-show
```

For a small validation run, specify a single year and lower the minimum-year
requirement. Confirm that all three month folders exist for that year:

```powershell
.\launch_katabatic.ps1 --path ".\Data\Brut_netcdf_files" --mode yearly --expected-min-years 1 --save-figures --no-show
```

Outputs are written locally under `Data/NPY/`, `Data/CARRA_Detector/`, and
`Figures/`. No network access is required once the Python environment, input
data, derived arrays, and Cartopy map cache are present.
