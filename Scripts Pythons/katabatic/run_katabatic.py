"""Command-line entry point for yearly or concatenated katabatic processing."""

import argparse
import os
import subprocess
import sys
import re
from pathlib import Path

import numpy as np

# This launcher adds the containing Python directory to sys.path when run alone.
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# In headless runs, force a non-interactive backend before matplotlib is imported.
if "--no-show" in sys.argv and os.environ.get("KATABATIC_FORCE_AGG") is None:
    os.environ["KATABATIC_FORCE_AGG"] = "1"

from katabatic.Katabatic_classification import (
    process_jja_years,
    process_jja_concat,
    select_pixel_on_reference_map,
    export_fraction_evolution_graphs,
    export_yearly_maps_from_cache,
)
from katabatic.reference_map_picker import (
    build_selection_file_path,
    load_default_reference_map_from_cache,
    load_reference_map_from_cache,
)
from shared.cache_utils import compute_signature, load_manifest, outputs_exist, path_fingerprint, write_manifest
from shared.map_launcher import plot_iceland
from shared.paths import (
    DEFAULT_CARRA_DETECTOR_DIR,
    DEFAULT_FIGURES_DIR,
    DEFAULT_KATABATIC_DATA_DIR,
    DEFAULT_NPY_DATA_DIR,
    resolve_glacier_mask_path,
)
from shared.reader import load_static_orography, resolve_available_years

# Default thresholds used by the katabatic detection step.
DT_THRESHOLD = 1.5           # Minimum cold anomaly below the local mean in °C
WIND_THRESHOLD = 2.5         # Minimum 10 m wind speed in m/s
DIRECTION_THRESHOLD = 0.87    # Minimum cosine alignment with the downslope direction
WINDOW_SIZE = 15            # Size of the local smoothing window for temperature


def _jja_years_from_data(path, expected_min_years):
    return resolve_available_years(
        path,
        ["June", "July", "August"],
        required_vars=["t2m", "u10", "v10"],
        requested_year="multi",
        expected_min_years=expected_min_years,
        prefer_strict_timestamps=True,
    )


def _expected_output_paths(mode, param_tag, years):
    if not years:
        return []

    if mode == "yearly":
        period = f"{int(years[0])} - {int(years[-1])}"
        return [
            DEFAULT_NPY_DATA_DIR / "yearly" / param_tag / "aggregated" / period / f"per_kat_{param_tag}.npy",
            DEFAULT_NPY_DATA_DIR / "yearly" / param_tag / "aggregated" / period / f"label_map_{param_tag}.npy",
            DEFAULT_CARRA_DETECTOR_DIR / "carra2_katabatic_classification.json",
        ]

    period = f"{int(years[0])} - {int(years[-1])}"
    return [
        DEFAULT_NPY_DATA_DIR / "concat" / param_tag / period / f"per_kat_{param_tag}.npy",
        DEFAULT_NPY_DATA_DIR / "concat" / param_tag / period / f"label_map_{param_tag}.npy",
    ]


def _cache_manifest_path(mode, param_tag):
    base = Path(DEFAULT_CARRA_DETECTOR_DIR)
    return (
        base / f"carra2_katabatic_classification_manifest_{mode}_{param_tag}.json",
        base / f"carra2_katabatic_classification_manifest_{mode}_latest.json",
    )


def _discover_cached_param_tags(mode):
    base = Path(DEFAULT_CARRA_DETECTOR_DIR)
    pattern = f"carra2_katabatic_classification_manifest_{mode}_*.json"
    tags = []
    for path in sorted(base.glob(pattern)):
        name = path.name
        if name.endswith(f"{mode}_latest.json"):
            continue
        prefix = f"carra2_katabatic_classification_manifest_{mode}_"
        if not name.startswith(prefix):
            continue
        tag = name[len(prefix):-5]
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _parse_param_tag(param_tag):
    match = re.fullmatch(r"dt(?P<dt>[-+]?\d*\.?\d+)_ws(?P<ws>\d+)_wind(?P<wind>[-+]?\d*\.?\d+)_dir(?P<dir>[-+]?\d*\.?\d+)", str(param_tag))
    if not match:
        raise ValueError(f"Cannot parse param tag: {param_tag}")
    return {
        "dT_threshold": float(match.group("dt")),
        "window_size": int(match.group("ws")),
        "wind_threshold": float(match.group("wind")),
        "direction_threshold": float(match.group("dir")),
    }



def main():
    parser = argparse.ArgumentParser(description="Run katabatic classification for JJA data.")
    parser.add_argument("--path", default=str(DEFAULT_KATABATIC_DATA_DIR), help="Path to the CARRA2 data directory.")
    parser.add_argument("--mode", choices=["concat", "yearly"], default="yearly",
                        help="Data loading mode: concat full dataset or process year-by-year.")
    parser.add_argument("--output-ref-year", default="2022",
                        help="Reference year used for plotting metadata and coordinates.")
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURES_DIR),
                        help="Base directory where plotted figures are saved.")
    parser.add_argument("--orog-year", default="2022",
                        help="Year used to load the static orography file.")
    parser.add_argument("--orog-month", default="June",
                        help="Month folder used to load the static orography file.")
    parser.add_argument("--orog-file-index", type=int, default=2,
                        help="Index N for selecting data_N.nc as static orography source.")
    parser.add_argument("--save-figures", action="store_true",
                        help="Save figures to disk under the output directory.")
    parser.add_argument("--save-figure", dest="save_figures", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--save-animations", action="store_true",
                        help="Save katabatic animations to disk under the output directory.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display figures interactively.")
    parser.add_argument("--dT-threshold", type=float, default=DT_THRESHOLD,
                        help="Temperature anomaly threshold below local mean in °C.")
    parser.add_argument("--wind-threshold", type=float, default=WIND_THRESHOLD,
                        help="Minimum wind speed in m/s for katabatic detection.")
    parser.add_argument("--direction-threshold", type=float, default=DIRECTION_THRESHOLD,
                        help="Cosine overlap threshold for wind alignment with downslope.")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help="Smoothing window size used for local temperature mean.")
    parser.add_argument("--expected-min-years", type=int, default=5,
                        help="Minimum number of complete years that must be available before processing starts.")
    parser.add_argument("--reuse-if-unchanged", dest="reuse_if_unchanged", action="store_true",
                        help="Reuse existing outputs when configuration and inputs are unchanged.")
    parser.add_argument("--no-reuse-if-unchanged", dest="reuse_if_unchanged", action="store_false",
                        help="Disable cache reuse and recompute.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force recomputation even if cache signature matches.")
    parser.add_argument(
        "--all-cached-tags",
        action="store_true",
        help="Run the launcher once for every cached katabatic parameter tag found in the detector cache manifests.",
    )
    parser.add_argument("--fraction-evolution", choices=["basin", "pixel"], default=None,
                        help="Export the katabatic fraction evolution for a selected basin or a clicked pixel on the aggregated reference map.")
    parser.add_argument("--fraction-evolution-basin", type=int, default=None,
                        help="Basin label to use when exporting the basin fraction-evolution graph. If omitted in basin mode, all basins are exported.")
    parser.add_argument("--fraction-evolution-pixel", nargs=2, type=int, metavar=("Y", "X"), default=None,
                        help="Pixel coordinates (y x) to export, if not selected by click on the reference map.")
    parser.add_argument("--fraction-evolution-click", action="store_true",
                        help="Open the default aggregated fraction map and let the user click a pixel for the pixel-mode export.")
    parser.set_defaults(reuse_if_unchanged=True)

    args = parser.parse_args()
    show_plots = not args.no_show

    if args.all_cached_tags:
        cached_tags = _discover_cached_param_tags(args.mode)
        if not cached_tags:
            raise RuntimeError(f"No cached katabatic tags found for mode={args.mode}.")

        failures = []
        for cached_tag in cached_tags:
            try:
                parsed = _parse_param_tag(cached_tag)
            except Exception as exc:
                failures.append((cached_tag, str(exc)))
                print(f"[launcher][skip] {cached_tag}: {exc}")
                continue

            child_args = [
                "-m", "katabatic.run_katabatic",
                "--path", str(args.path),
                "--mode", args.mode,
                "--output-ref-year", str(args.output_ref_year),
                "--output-dir", str(args.output_dir),
                "--orog-year", str(args.orog_year),
                "--orog-month", str(args.orog_month),
                "--orog-file-index", str(args.orog_file_index),
                "--dT-threshold", str(parsed["dT_threshold"]),
                "--wind-threshold", str(parsed["wind_threshold"]),
                "--direction-threshold", str(parsed["direction_threshold"]),
                "--window-size", str(parsed["window_size"]),
                "--expected-min-years", str(args.expected_min_years),
                "--mk-cached-batch-placeholder",
            ]

            # Remove placeholder used only to keep the construction readable.
            child_args = [item for item in child_args if item != "--mk-cached-batch-placeholder"]

            if args.save_figures:
                child_args.append("--save-figures")
            if args.save_animations:
                child_args.append("--save-animations")
            if args.no_show:
                child_args.append("--no-show")
            if args.reuse_if_unchanged:
                child_args.append("--reuse-if-unchanged")
            else:
                child_args.append("--no-reuse-if-unchanged")
            if args.force_recompute:
                child_args.append("--force-recompute")
            if args.fraction_evolution == "basin":
                child_args.extend(["--fraction-evolution", "basin"])
                if args.fraction_evolution_basin is not None:
                    child_args.extend(["--fraction-evolution-basin", str(args.fraction_evolution_basin)])
            elif args.fraction_evolution == "pixel":
                child_args.extend(["--fraction-evolution", "pixel"])
                if args.fraction_evolution_pixel is not None:
                    child_args.extend(["--fraction-evolution-pixel", str(args.fraction_evolution_pixel[0]), str(args.fraction_evolution_pixel[1])])
                if args.fraction_evolution_click:
                    child_args.append("--fraction-evolution-click")

            print(f"[launcher] Batch cache run: {cached_tag}")
            completed = subprocess.run([sys.executable, *child_args], check=False)
            if completed.returncode != 0:
                failures.append((cached_tag, f"exit code {completed.returncode}"))

        if failures:
            print("[launcher] Batch cache mode completed with failures:")
            for tag, reason in failures:
                print(f"  - {tag}: {reason}")
            if len(failures) == len(cached_tags):
                raise RuntimeError("All cached-tag batch runs failed.")
        return

    param_tag = (
        f"dt{args.dT_threshold:g}_ws{args.window_size}"
        f"_wind{args.wind_threshold:g}_dir{args.direction_threshold:g}"
    )
    output_dir = Path(args.output_dir)
    if args.save_figures:
        output_dir.mkdir(parents=True, exist_ok=True)

    years = _jja_years_from_data(args.path, expected_min_years=args.expected_min_years)
    cache_payload = {
        "schema_version": 1,
        "stage": "carra2_katabatic_classification",
        "mode": args.mode,
        "path": path_fingerprint(args.path),
        "years": [str(y) for y in years],
        "orog": {
            "year": str(args.orog_year),
            "month": str(args.orog_month),
            "file_index": int(args.orog_file_index),
            "output_ref_year": str(args.output_ref_year),
        },
        "detector": {
            "window_size": int(args.window_size),
            "dT_threshold": float(args.dT_threshold),
            "wind_threshold": float(args.wind_threshold),
            "direction_threshold": float(args.direction_threshold),
        },
    }
    cache_signature = compute_signature(cache_payload)
    manifest_path, manifest_latest_path = _cache_manifest_path(args.mode, param_tag)
    expected_outputs = _expected_output_paths(args.mode, param_tag, years)
    manifest = load_manifest(manifest_path)
    years_for_period = [str(y) for y in years]
    period = None
    if years_for_period:
        years_int = sorted(int(y) for y in years_for_period)
        period = f"{years_int[0]} - {years_int[-1]}"

    if args.reuse_if_unchanged and (not args.force_recompute):
        if manifest is not None and manifest.get("signature") == cache_signature and outputs_exist(expected_outputs):
            if args.fraction_evolution is not None:
                if years_for_period:
                    ref_map, label_map = load_reference_map_from_cache(param_tag, years, DEFAULT_NPY_DATA_DIR)
                    if ref_map is not None and label_map is not None:
                        active_figures_root = (
                            Path(args.output_dir)
                            / "yearly"
                            / param_tag
                            / "aggregated"
                            / str(period)
                        )
                        selection_file = build_selection_file_path(
                            param_tag=param_tag,
                            years=years,
                            figures_root=Path(args.output_dir),
                        )
                        if args.fraction_evolution == "pixel":
                            pixel = tuple(args.fraction_evolution_pixel) if args.fraction_evolution_pixel else None
                            if args.fraction_evolution_click and pixel is None:
                                default_ref_map = load_default_reference_map_from_cache(years, DEFAULT_NPY_DATA_DIR)
                                if default_ref_map is not None:
                                    pixel = select_pixel_on_reference_map(
                                        default_ref_map,
                                        selection_file=selection_file,
                                        param_tag=param_tag,
                                        years=years,
                                        show=True,
                                    )
                                else:
                                    pixel = select_pixel_on_reference_map(
                                        ref_map,
                                        selection_file=selection_file,
                                        param_tag=param_tag,
                                        years=years,
                                        show=True,
                                    )
                            if pixel is not None:
                                export_fraction_evolution_graphs(
                                    mode="pixel",
                                    yearly_per_kat_by_year={
                                        str(y): np.load(DEFAULT_NPY_DATA_DIR / "yearly" / param_tag / str(y) / f"per_kat_{param_tag}.npy")
                                        for y in years
                                        if (DEFAULT_NPY_DATA_DIR / "yearly" / param_tag / str(y) / f"per_kat_{param_tag}.npy").exists()
                                    },
                                    label_map=label_map,
                                    output_dir=active_figures_root,
                                    pixel=pixel,
                                    reference_map=ref_map,
                                    title_prefix=f"{period} — ",
                                )
                        elif args.fraction_evolution == "basin":
                            basin_id = args.fraction_evolution_basin if args.fraction_evolution_basin is not None else "all"
                            export_fraction_evolution_graphs(
                                mode="basin",
                                yearly_per_kat_by_year={
                                    str(y): np.load(DEFAULT_NPY_DATA_DIR / "yearly" / param_tag / str(y) / f"per_kat_{param_tag}.npy")
                                    for y in years
                                    if (DEFAULT_NPY_DATA_DIR / "yearly" / param_tag / str(y) / f"per_kat_{param_tag}.npy").exists()
                                },
                                label_map=label_map,
                                output_dir=active_figures_root,
                                basin_label=basin_id,
                                reference_map=ref_map,
                                title_prefix=f"{period} — ",
                            )
            if args.save_figures and args.mode == "yearly":
                period_root = Path(args.output_dir) / "yearly" / param_tag / "aggregated" / str(period)
                maps_root = period_root / "maps"
                legacy_figures_root = period_root / "figures"
                graphs_root = period_root / "graphs"
                period_root.mkdir(parents=True, exist_ok=True)
                maps_root.mkdir(parents=True, exist_ok=True)
                graphs_root.mkdir(parents=True, exist_ok=True)

                # Migrate legacy aggregate maps from "figures" to "maps" once.
                legacy_fraction_map_path = legacy_figures_root / "katabatic_fraction.png"
                legacy_clusters_map_path = legacy_figures_root / "katabatic_clusters.png"
                fraction_map_path = maps_root / "katabatic_fraction.png"
                clusters_map_path = maps_root / "katabatic_clusters.png"
                if legacy_fraction_map_path.exists() and not fraction_map_path.exists():
                    legacy_fraction_map_path.replace(fraction_map_path)
                if legacy_clusters_map_path.exists() and not clusters_map_path.exists():
                    legacy_clusters_map_path.replace(clusters_map_path)
                if legacy_figures_root.exists() and not any(legacy_figures_root.iterdir()):
                    legacy_figures_root.rmdir()

                need_aggregate_maps = (not fraction_map_path.exists()) or (not clusters_map_path.exists())
                need_yearly_maps = any(
                    not ((Path(args.output_dir) / "yearly" / param_tag / str(year) / "maps" / "katabatic_fraction.png").exists())
                    or not ((Path(args.output_dir) / "yearly" / param_tag / str(year) / "maps" / "katabatic_clusters.png").exists())
                    for year in years
                )
                ds_orog_static = None
                if need_aggregate_maps or need_yearly_maps:
                    ds_orog_static = load_static_orography(
                        args.path,
                        year=args.orog_year or args.output_ref_year,
                        month=args.orog_month,
                        file_index=args.orog_file_index,
                    )
                if need_yearly_maps and ds_orog_static is not None:
                    export_yearly_maps_from_cache(
                        output_dir=Path(args.output_dir),
                        years=years,
                        param_tag=param_tag,
                        ref_ds=ds_orog_static,
                        show_plots=show_plots,
                        launch_params={
                            "dt_threshold": args.dT_threshold,
                            "window_size": args.window_size,
                            "wind_threshold": args.wind_threshold,
                            "direction_threshold": args.direction_threshold,
                        },
                        glaciers=np.load(resolve_glacier_mask_path()),
                    )
                aggregate_file = DEFAULT_NPY_DATA_DIR / "yearly" / param_tag / "aggregated" / period / f"per_kat_{param_tag}.npy"
                label_map_file = DEFAULT_NPY_DATA_DIR / "yearly" / param_tag / "aggregated" / period / f"label_map_{param_tag}.npy"
                if need_aggregate_maps and aggregate_file.exists() and label_map_file.exists() and ds_orog_static is not None:
                    aggregated_percent = np.load(aggregate_file)
                    label_map = np.load(label_map_file)
                    maps_root.mkdir(parents=True, exist_ok=True)
                    plot_iceland(
                        aggregated_percent,
                        "per_kat",
                        typ="mask",
                        ref=ds_orog_static,
                        cmap="plasma",
                        glaciers=np.load(resolve_glacier_mask_path()),
                        save_path=fraction_map_path,
                        show=show_plots,
                        title=f"Katabatic fraction - {period} (JJA)",
                        coasts="white",
                        launch_params={
                            "dt_threshold": args.dT_threshold,
                            "window_size": args.window_size,
                            "wind_threshold": args.wind_threshold,
                            "direction_threshold": args.direction_threshold,
                        },
                    )
                    plot_iceland(
                        label_map,
                        f"cluster_{int(np.max(label_map))}",
                        typ="mask",
                        ref=ds_orog_static,
                        unit={"cluster": "n"},
                        cmap="tab20",
                        glaciers=np.load(resolve_glacier_mask_path()),
                        save_path=clusters_map_path,
                        show=show_plots,
                        title=f"Katabatic clusters - {period} (JJA)",
                        launch_params={
                            "dt_threshold": args.dT_threshold,
                            "window_size": args.window_size,
                            "wind_threshold": args.wind_threshold,
                            "direction_threshold": args.direction_threshold,
                        },
                    )

            print(f"[launcher] Cache hit, skipping katabatic {args.mode} run.")
            print(f"[launcher] Manifest: {manifest_path}")
            return

    defaults_params = args.dT_threshold == 1.5 and args.wind_threshold == 2 and args.direction_threshold == 0.87 and args.window_size == 15

    if args.mode == "yearly":
        print(f"[laucher] Launching JJA data processing year-by-year mode with : " + 
            defaults_params * "default parameters :" + f" dt={args.dT_threshold:g} | ws={args.window_size} | wind={args.wind_threshold:g} | dir={args.direction_threshold:g} | " 
            + "figure saving enabled | " * args.save_figures + "figure saving disabled | " * (not args.save_figures) +
            "animation saving enabled | " * args.save_animations + "animation saving disabled | " * (not args.save_animations) +
            "interactive plotting enabled | " * show_plots + "interactive plotting disabled | " * (not show_plots))
        process_jja_years(
            path=args.path,
            output_ref_year=args.output_ref_year,
            orog_year=args.orog_year,
            orog_month=args.orog_month,
            orog_file_index=args.orog_file_index,
            expected_min_years=args.expected_min_years,
            window_size=args.window_size,
            dT_threshold=args.dT_threshold,
            wind_threshold=args.wind_threshold,
            direction_threshold=args.direction_threshold,
            output_dir=output_dir,
            param_tag=param_tag,
            save_figures=args.save_figures,
            save_animations=args.save_animations,
            show_plots=show_plots,
            fraction_evolution_mode=args.fraction_evolution,
            fraction_evolution_basin=args.fraction_evolution_basin,
            fraction_evolution_pixel=(tuple(args.fraction_evolution_pixel) if args.fraction_evolution_pixel else None),
            fraction_evolution_click=args.fraction_evolution_click,
        )
    else:
        print(f"[launcher] Launching JJA data processing in concat mode with : " + 
            defaults_params * "default parameters :" + f" dt={args.dT_threshold:g} | ws={args.window_size} | wind={args.wind_threshold:g} | dir={args.direction_threshold:g} | " 
            + "figure saving enabled | " * args.save_figures + "figure saving disabled | " * (not args.save_figures) +
            "animation saving enabled | " * args.save_animations + "animation saving disabled | " * (not args.save_animations) +
            "interactive plotting enabled | " * show_plots + "interactive plotting disabled | " * (not show_plots))
        process_jja_concat(
            path=args.path,
            output_ref_year=args.orog_year,
            orog_month=args.orog_month,
            orog_file_index=args.orog_file_index,
            expected_min_years=args.expected_min_years,
            window_size=args.window_size,
            dT_threshold=args.dT_threshold,
            wind_threshold=args.wind_threshold,
            direction_threshold=args.direction_threshold,
            output_dir=output_dir,
            param_tag=param_tag,
            save_figures=args.save_figures,
            save_animations=args.save_animations,
            show_plots=show_plots,
            fraction_evolution_mode=args.fraction_evolution,
            fraction_evolution_basin=args.fraction_evolution_basin,
            fraction_evolution_pixel=(tuple(args.fraction_evolution_pixel) if args.fraction_evolution_pixel else None),
            fraction_evolution_click=args.fraction_evolution_click,
        )

    manifest_payload = {
        "signature": cache_signature,
        "signature_payload": cache_payload,
        "mode": args.mode,
        "param_tag": param_tag,
        "expected_outputs": [str(Path(p).resolve()) for p in expected_outputs],
    }
    write_manifest(manifest_path, manifest_payload)
    write_manifest(manifest_latest_path, manifest_payload)


if __name__ == "__main__":
    main()
