"""Example plotting script for locally cached katabatic output arrays."""

import numpy as np
from shared.map_launcher import plot_iceland, animate_iceland
from shared.paths import DEFAULT_KATABATIC_DATA_DIR, DATA_DIR, resolve_glacier_mask_path, DEFAULT_NPY_DATA_DIR, DEFAULT_FIGURES_DIR

path = DEFAULT_NPY_DATA_DIR / "concat" / "window_3_dT_1.0_wind_5.0_dir_30.0" / "2020 - 2025"
per_kat = np.load(DEFAULT_NPY_DATA_DIR / "per_kat_dt1.5_ws15_wind2.5_dir0.87.npy")


plot_iceland(
        per_kat,
        "per_kat",
        typ="mask",
        ref=None,
        cmap="plasma",
        glaciers=np.load(resolve_glacier_mask_path()),
        save_path=DEFAULT_FIGURES_DIR / "katabatic_fraction.png",
        show=True,
        title=f"Katabatic fraction",
        coasts="white"
    )