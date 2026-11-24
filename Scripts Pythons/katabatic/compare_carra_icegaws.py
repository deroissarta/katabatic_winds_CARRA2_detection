"""Compare CARRA2 katabatic detections with ICE-GAWS station observations."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from shared.icegaws import load_station_overlay_points_from_json
from shared.map_launcher import plot_iceland
from shared.paths import DEFAULT_FIGURES_DIR, DEFAULT_KATABATIC_DATA_DIR, DEFAULT_NPY_DATA_DIR, DATA_DIR, resolve_glacier_mask_path
from shared.reader import load_static_orography
from shared.icegaws import load_station_locations, write_station_static_mapping_json, write_station_katabatic_detector_json


LEGACY_MAPPING_JSON_NAMES = [
    "icegaws_station_mapping.json",
    "test_pixel_map.json",
]
LEGACY_DETECTOR_JSON_NAMES = [
    "icegaws_katabatic_results.json",
]

COMPARE_CONFIG_STATE_PATH = DATA_DIR / "ICE-GAWS" / ".compare_last_compute_config.json"


def _discover_latest_per_kat_path():
    yearly_candidates = sorted(DEFAULT_NPY_DATA_DIR.glob("yearly/*/aggregated/*/per_kat_*.npy"))
    if yearly_candidates:
        return yearly_candidates[-1]

    candidates = sorted(DEFAULT_NPY_DATA_DIR.glob("concat/*/*/per_kat_*.npy"))
    if not candidates:
        raise FileNotFoundError(
            f"No saved per_kat files found under {DEFAULT_NPY_DATA_DIR / 'yearly'} or {DEFAULT_NPY_DATA_DIR / 'concat'}"
        )
    return candidates[-1]


def _safe_title_from_path(path: Path):
    return path.stem.replace("per_kat_", "").replace("_", " ")


def _load_points(mapping_json, detector_json, station_codes=None):
    points = load_station_overlay_points_from_json(
        static_mapping_path=mapping_json,
        detector_json_path=detector_json,
        station_codes=station_codes,
        value_key="event_fraction",
        normalize_value=False,
        value_scale=1.0,
    )
    points = [
        point
        for point in points
        if point.get("lon") is not None
        and point.get("lat") is not None
        and point.get("value") is not None
        and np.isfinite(point.get("value"))
    ]
    return points


def _ensure_json_file(path: Path, fallback_candidates=None):
    if path.exists():
        return path

    if fallback_candidates is not None:
        for candidate in fallback_candidates:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path

    return path


def _resolve_with_legacy_fallback_if_default(arg_value: str, default_value: str, fallback_candidates=None):
    requested = Path(arg_value)
    if requested.exists():
        return requested
    if Path(arg_value) == Path(default_value):
        return _ensure_json_file(requested, fallback_candidates=fallback_candidates)
    return requested


def _make_output_path(output_dir: Path, per_kat_path: Path, station_name: str | None = None):
    base = per_kat_path.stem
    if station_name:
        base = f"{base}_{station_name}"
    return output_dir / f"{base}.png"


def _run_module(module_name, args):
    command = [sys.executable, "-m", module_name, *args]
    print(f"[compare] Run: {' '.join(command)}")
    subprocess.run(command, check=True)


def _normalize_station_arg(stations):
    if not stations:
        return None
    return sorted(str(station) for station in stations)


def _build_compute_config(args):
    return {
        "carra_data_dir": str(args.carra_data_dir),
        "katabatic_mode": args.katabatic_mode,
        "per_kat_path": str(args.per_kat_path) if args.per_kat_path is not None else None,
        "station_mapping_json": str(args.station_mapping_json),
        "detector_json": str(args.detector_json),
        "station": _normalize_station_arg(args.station),
        "dT_threshold": float(args.dT_threshold),
        "wind_threshold": float(args.wind_threshold),
        "direction_threshold": float(args.direction_threshold),
        "window_size": int(args.window_size),
    }


def _load_last_compute_config(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        return None
    return payload.get("compute_config")


def _save_compute_config(path, compute_config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"compute_config": compute_config}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _apply_recompute_policy_from_config(args):
    compute_config = _build_compute_config(args)
    previous_config = _load_last_compute_config(COMPARE_CONFIG_STATE_PATH)
    config_changed = previous_config != compute_config

    if config_changed:
        print("[compare] New compute configuration detected -> force full recompute.")
        args.force_recompute = True
        args.refresh_detector = True
    else:
        print("[compare] Compute configuration unchanged -> cache reuse allowed.")

    return compute_config, config_changed


def _run_katabatic_if_needed(args):
    command_args = [
        "--path",
        str(args.carra_data_dir),
        "--mode",
        args.katabatic_mode,
        "--dT-threshold",
        str(args.dT_threshold),
        "--wind-threshold",
        str(args.wind_threshold),
        "--direction-threshold",
        str(args.direction_threshold),
        "--window-size",
        str(args.window_size),
    ]
    if args.force_recompute:
        command_args.append("--force-recompute")
    else:
        command_args.append("--reuse-if-unchanged")

    _run_module("katabatic.run_katabatic", command_args)


def _resolve_per_kat_path(args):
    if args.per_kat_path is not None:
        candidate = Path(args.per_kat_path)
        if candidate.exists():
            return candidate

        if not args.auto_run_katabatic_if_missing:
            raise FileNotFoundError(f"CARRA per_kat file not found: {candidate}")

        print(f"[compare] Missing per_kat file: {candidate}")
        _run_katabatic_if_needed(args)
        if candidate.exists():
            return candidate

        latest = _discover_latest_per_kat_path()
        return latest

    try:
        return _discover_latest_per_kat_path()
    except FileNotFoundError:
        if not args.auto_run_katabatic_if_missing:
            raise

    print("[compare] No saved per_kat found, auto-running katabatic classification.")
    _run_katabatic_if_needed(args)
    return _discover_latest_per_kat_path()


def _ensure_mapping_and_detector(args, ref, station_locations, location_summary_path, mapping_path, detector_path):
    if args.refresh_detector:
        _run_module(
            "katabatic.build_icegaws_carra_pixel_map",
            [
                "--carra-data-dir",
                str(args.carra_data_dir),
                "--output-mapping-json",
                str(mapping_path),
                "--output-results-json",
                str(detector_path),
                "--window-size",
                str(args.window_size),
                "--dT-threshold",
                str(args.dT_threshold),
                "--wind-threshold",
                str(args.wind_threshold),
                "--direction-threshold",
                str(args.direction_threshold),
                *( ["--station", *args.station] if args.station else [] ),
                *( ["--force-recompute"] if args.force_recompute else ["--reuse-if-unchanged"] ),
            ],
        )

    if not mapping_path.exists():
        mapping_path = write_station_static_mapping_json(
            output_path=mapping_path,
            carra_orography_ds=ref,
            station_locations=station_locations,
            station_codes=args.station,
            data_dir=Path(args.carra_data_dir).parent / "ICE-GAWS",
            location_summary_path=location_summary_path,
            basins_path=DEFAULT_NPY_DATA_DIR / "basins.npy",
        )

    if not detector_path.exists():
        if args.skip_detector_build:
            raise FileNotFoundError(f"Detector JSON not found and build is disabled: {detector_path}")
        detector_path = write_station_katabatic_detector_json(
            output_path=detector_path,
            carra_orography_ds=ref,
            station_locations=station_locations,
            station_codes=args.station,
            data_dir=Path(DATA_DIR / "ICE-GAWS"),
            location_summary_path=location_summary_path,
            basins_path=DEFAULT_NPY_DATA_DIR / "basins.npy",
            temperature_reference_mode="spatial_carra",
            carra_t2m_data_dir=Path(args.carra_data_dir),
        )

    return mapping_path, detector_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot CARRA2 per_kat maps with ICE-GAWS station comparisons using the same detector parameters."
        )
    )
    parser.add_argument(
        "--per-kat-path",
        default=None,
        help="Saved CARRA per_kat .npy file. If omitted, the latest saved concat per_kat file is used.",
    )
    parser.add_argument(
        "--carra-data-dir",
        default=str(DEFAULT_KATABATIC_DATA_DIR),
        help="CARRA2 data directory used to load the reference grid.",
    )
    parser.add_argument(
        "--carra-year",
        default="2022",
        help="Reference year used to load the CARRA grid.",
    )
    parser.add_argument(
        "--carra-month",
        default="June",
        help="Reference month used to load the CARRA grid.",
    )
    parser.add_argument(
        "--carra-file-index",
        type=int,
        default=2,
        help="Reference file index used to load the CARRA grid.",
    )
    default_mapping_json = str(DATA_DIR / "ICE-GAWS" / "icegaws_carra_pixel_basin_mapping.json")
    default_detector_json = str(DATA_DIR / "ICE-GAWS" / "icegaws_katabatic_detector_results.json")

    parser.add_argument(
        "--station-mapping-json",
        default=default_mapping_json,
        help="Static ICE-GAWS station-to-pixel mapping JSON.",
    )
    parser.add_argument(
        "--detector-json",
        default=default_detector_json,
        help="ICE-GAWS detector results JSON.",
    )
    parser.add_argument(
        "--skip-detector-build",
        action="store_true",
        help="Use the detector JSON only if it already exists; otherwise fail instead of recomputing it.",
    )
    parser.add_argument(
        "--station",
        nargs="*",
        default=None,
        help="Optional station codes to plot. If omitted, all available stations are used.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_FIGURES_DIR / "icegaws_comparisons"),
        help="Directory where comparison figures are written.",
    )
    parser.add_argument(
        "--annotate-points",
        action="store_true",
        help="Annotate station points with labels and values.",
    )
    parser.add_argument(
        "--per-station",
        action="store_true",
        help="Also write one figure per station in addition to the overview map.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures interactively.",
    )
    parser.add_argument(
        "--katabatic-mode",
        choices=["yearly", "concat"],
        default="yearly",
        help="Mode used when auto-running katabatic classification to build missing per_kat.",
    )
    parser.add_argument(
        "--auto-run-katabatic-if-missing",
        dest="auto_run_katabatic_if_missing",
        action="store_true",
        help="Auto-run katabatic classification when per_kat is missing.",
    )
    parser.add_argument(
        "--no-auto-run-katabatic-if-missing",
        dest="auto_run_katabatic_if_missing",
        action="store_false",
        help="Fail if per_kat is missing instead of auto-running classification.",
    )
    parser.add_argument(
        "--refresh-detector",
        action="store_true",
        help="Force detector/mapping refresh using the incremental builder before plotting.",
    )
    parser.add_argument("--dT-threshold", type=float, default=1.5, help="Detector dT threshold.")
    parser.add_argument("--wind-threshold", type=float, default=2.5, help="Detector wind threshold.")
    parser.add_argument("--direction-threshold", type=float, default=0.87, help="Detector direction threshold.")
    parser.add_argument("--window-size", type=int, default=15, help="Detector window size.")
    parser.add_argument("--force-recompute", action="store_true", help="Bypass cache and force recompute.")
    parser.set_defaults(auto_run_katabatic_if_missing=True)

    args = parser.parse_args()

    compute_config, config_changed = _apply_recompute_policy_from_config(args)
    if config_changed and args.per_kat_path is None and args.auto_run_katabatic_if_missing:
        print("[compare] Refresh katabatic classification for the new configuration.")
        _run_katabatic_if_needed(args)

    per_kat_path = _resolve_per_kat_path(args)
    if not per_kat_path.exists():
        raise FileNotFoundError(f"CARRA per_kat file not found: {per_kat_path}")

    per_kat = np.load(per_kat_path)
    ref = load_static_orography(
        Path(args.carra_data_dir),
        year=args.carra_year,
        month=args.carra_month,
        file_index=args.carra_file_index,
    )

    if per_kat.shape != ref["orog"].isel(time=0).shape:
        raise ValueError(
            f"Shape mismatch: per_kat has shape {per_kat.shape} but reference grid has shape {ref['orog'].isel(time=0).shape}."
        )

    mapping_path = _resolve_with_legacy_fallback_if_default(
        arg_value=args.station_mapping_json,
        default_value=default_mapping_json,
        fallback_candidates=[Path(DATA_DIR / "ICE-GAWS" / name) for name in LEGACY_MAPPING_JSON_NAMES],
    )
    detector_path = _resolve_with_legacy_fallback_if_default(
        arg_value=args.detector_json,
        default_value=default_detector_json,
        fallback_candidates=[Path(DATA_DIR / "ICE-GAWS" / name) for name in LEGACY_DETECTOR_JSON_NAMES],
    )
    location_summary_path = Path(DATA_DIR / "ICE-GAWS" / "ICE-GAWS-location-summary.csv")
    station_locations = load_station_locations(location_summary_path)

    mapping_path, detector_path = _ensure_mapping_and_detector(
        args=args,
        ref=ref,
        station_locations=station_locations,
        location_summary_path=location_summary_path,
        mapping_path=mapping_path,
        detector_path=detector_path,
    )

    points = _load_points(
        mapping_json=mapping_path,
        detector_json=detector_path,
        station_codes=args.station,
    )
    if not points:
        raise RuntimeError("No common ICE-GAWS stations with available detector results were found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_path = output_dir / f"{per_kat_path.stem}_overview.png"
    point_norm = Normalize(vmin=0.0, vmax=1.0)
    plot_iceland(
        per_kat,
        "per_kat",
        typ="mask",
        ref=ref,
        unit={"per_kat": "0-1"},
        cmap="plasma",
        glaciers=np.load(resolve_glacier_mask_path()),
        points=points,
        points_norm=point_norm,
        points_size=45,
        points_edgecolor="white",
        annotate_points=args.annotate_points,
        save_path=overview_path,
        show=not args.no_show,
        title=f"{_safe_title_from_path(per_kat_path)} - ICE-GAWS available stations",
        coasts="white",
    )

    if args.per_station:
        for point in points:
            station_name = str(point["station"])
            station_path = _make_output_path(output_dir, per_kat_path, station_name=station_name)
            plot_iceland(
                per_kat,
                "per_kat",
                typ="mask",
                ref=ref,
                unit={"per_kat": "0-1"},
                cmap="plasma",
                glaciers=np.load(resolve_glacier_mask_path()),
                points=[point],
                points_norm=point_norm,
                points_size=45,
                points_edgecolor="white",
                annotate_points=True,
                save_path=station_path,
                show=False,
                title=f"{_safe_title_from_path(per_kat_path)} - {station_name}",
                coasts="white",
            )

    print(f"[compare] Wrote overview figure to: {overview_path}")
    if args.per_station:
        print(f"[compare] Wrote {len(points)} per-station figures to: {output_dir}")

    _save_compute_config(COMPARE_CONFIG_STATE_PATH, compute_config)


if __name__ == "__main__":
    main()