# Katabatic analysis — offline copy

This folder contains a standalone copy of the Python katabatic winds detection tool and its
shared Python modules developed during my internship with Haraldur Olafson. 
It does not contain CARRA2 files, derived arrays, maps,
or a prebuilt Python environment.

The application can run without an internet connection after its Python
environment, input data, and Cartopy map assets have been installed locally.
See [DATA_INSTALLATION.md](./DATA_INSTALLATION.md) for the complete setup and
transfer procedure.

## Structure

- `Scripts Pythons/katabatic/` contains the analysis, validation, and data
  preparation commands.
- `Scripts Pythons/shared/` contains the readers, path configuration, plotting,
  and cache helpers used by those commands.
- `Data/` is reserved for local input data, derived arrays, and Cartopy's
  Natural Earth cache. No datasets are included.
- `Figures/` is the default location for generated plots.

## Quick start

1. Install or activate the `katabatic-offline` environment described in
   `environment.yml`.
2. Follow [DATA_INSTALLATION.md](./DATA_INSTALLATION.md) to copy the CARRA2
   inputs and prepare the local map cache.
3. Open PowerShell in this folder and check the command-line options:

   ```powershell
   .\launch_katabatic.ps1 --help
   ```

4. Run the default yearly JJA workflow and save its figures:

   ```powershell
   .\launch_katabatic.ps1 --mode yearly --save-figures --no-show
   ```

The launcher uses the currently active Python interpreter. Activate the
environment first, for example with `conda activate katabatic-offline`.

## Understanding the workflow

The main command is `katabatic.run_katabatic`. It checks which years contain
complete June, July, and August inputs, loads the temperature, wind, and
orography NetCDF variables, then detects and summarizes katabatic events.
Generated numerical outputs and detector caches go under `Data/NPY/` and
`Data/CARRA_Detector/`; figures go under `Figures/`.

Several workflows need glacier and basin arrays in addition to the raw
meteorological files. Their required files and setup are listed in
[DATA_INSTALLATION.md](./DATA_INSTALLATION.md). Use `--help` on an individual
module for its options, for example:

```powershell
Push-Location ".\Scripts Pythons"
python -m katabatic.list_available_years --help
python -m katabatic.data_quality_check --help
Pop-Location
```

For ICE-GAWS station validation, install the L3 station files and location
summary, then follow Section 4 of `DATA_INSTALLATION.md` to generate the
mappings and run the comparison.

## Portability and offline behavior

Paths default to folders inside this copy. The `COMMON_DATA_ROOT`,
`KATABATIC_DATA_ROOT`, `KATABATIC_FIGURES_ROOT`, `HMI_DATA_ROOT`,
`CARRA_DETECTOR_ROOT`, and `GLACIER_MASK_PATH` environment variables can
redirect data and output locations without changing Python source.

Cartopy's Natural Earth coastlines and land/ocean boundaries must be cached
before disconnecting from the internet. The preparation script writes them to
`Data/cartopy/`, which is also the launcher's default `CARTOPY_DATA_DIR`.
Transfer that directory together with this folder and the required datasets.
