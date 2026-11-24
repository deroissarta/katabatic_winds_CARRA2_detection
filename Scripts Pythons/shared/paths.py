import os
from pathlib import Path


# Keep the copied project self-contained while allowing external data volumes.
def _env_path(var_name, default_path):
    # Environment variables can override project defaults for portable runs.
    value = os.getenv(var_name)
    return Path(value) if value else Path(default_path)


SCRIPTS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
# All default paths below resolve to the root of this offline project copy.
DATA_DIR = _env_path("COMMON_DATA_ROOT", PROJECT_ROOT / "Data")

DEFAULT_KATABATIC_DATA_DIR = _env_path(
    "KATABATIC_DATA_ROOT", DATA_DIR / "Brut_netcdf_files"
)
DEFAULT_FIGURES_DIR = _env_path("KATABATIC_FIGURES_ROOT", PROJECT_ROOT / "Figures")
DEFAULT_OBS_FIGURES_DIR = _env_path("OBS_FIGURES_ROOT", DEFAULT_FIGURES_DIR / "OBS")
DEFAULT_GEO_DATA_DIR = _env_path("GEO_DATA_ROOT", DATA_DIR/"GEO")
DEFAULT_HMI_DATA_DIR = _env_path("HMI_DATA_ROOT", DATA_DIR/"NPY")
DEFAULT_NPY_DATA_DIR = _env_path("HMI_DATA_ROOT", DATA_DIR/"NPY")
DEFAULT_CARRA_DETECTOR_DIR = _env_path("CARRA_DETECTOR_ROOT", DATA_DIR/"CARRA_Detector")


def resolve_glacier_mask_path():
    # Optional override for experiments; fallback keeps legacy project behavior.
    candidate = os.getenv("GLACIER_MASK_PATH")
    if candidate:
        return Path(candidate)
    return DEFAULT_NPY_DATA_DIR / "glacier_mask.npy"
