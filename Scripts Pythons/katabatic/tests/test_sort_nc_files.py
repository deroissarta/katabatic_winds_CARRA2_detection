"""Regression tests for organizing raw NetCDF files."""

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr


MODULE_PATH = Path(__file__).resolve().parents[1] / "sort_nc_files.py"
SPEC = importlib.util.spec_from_file_location("sort_nc_files", MODULE_PATH)
sort_nc_files = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sort_nc_files)


class SortNcFilesTests(unittest.TestCase):
    def test_sort_nc_files_processes_zip_archives(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "source"
            target_dir = tmp_path / "target"
            source_dir.mkdir()
            target_dir.mkdir()

            ds = xr.Dataset(
                data_vars={"t2m": (("time",), np.array([273.15]))},
                coords={"time": np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")},
            )

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_file:
                ds.to_netcdf(tmp_file.name)
                tmp_path_for_archive = Path(tmp_file.name)

            archive_path = source_dir / "sample.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(tmp_path_for_archive, arcname="data_0.nc")

            tmp_path_for_archive.unlink(missing_ok=True)

            moved = sort_nc_files.sort_nc_files(source_dir=source_dir, target_dir=target_dir, dry_run=False)

            self.assertEqual(len(moved), 1)
            destination = target_dir / "2020" / "CARRA2_January" / "data_0.nc"
            self.assertTrue(destination.exists())
            self.assertFalse(archive_path.exists())


if __name__ == "__main__":
    unittest.main()
