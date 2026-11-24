"""Download Cartopy map features into the project's portable local cache."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARTOPY_CACHE = PROJECT_ROOT / "Data" / "cartopy"

# Cartopy reads this setting when it initializes its Natural Earth downloaders.
os.environ.setdefault("CARTOPY_DATA_DIR", str(CARTOPY_CACHE))

import cartopy.feature as cfeature


def main():
    """Fetch every map feature used by the plotting helpers."""
    features = (
        ("110 m land", cfeature.LAND),
        ("110 m ocean", cfeature.OCEAN),
        ("10 m coastline", cfeature.COASTLINE.with_scale("10m")),
    )

    CARTOPY_CACHE.mkdir(parents=True, exist_ok=True)
    for label, feature in features:
        # Accessing the geometries materializes a missing Natural Earth asset.
        geometry_count = sum(1 for _ in feature.geometries())
        print(f"Prepared {label}: {geometry_count} geometries")

    print(f"Offline Cartopy cache: {CARTOPY_CACHE}")


if __name__ == "__main__":
    main()
