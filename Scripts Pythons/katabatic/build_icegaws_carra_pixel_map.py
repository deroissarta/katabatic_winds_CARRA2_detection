"""Match ICE-GAWS station locations to CARRA2 grid cells and basins."""

import argparse
import hashlib
import json
from pathlib import Path

from shared.cache_utils import compute_signature, path_fingerprint
from shared.icegaws import (
    ICEGAWS_LOCATION_SUMMARY,
    build_station_static_mapping_json,
    build_station_katabatic_detector_json,
    discover_l3_files,
    load_station_locations,
)
from shared.paths import DEFAULT_KATABATIC_DATA_DIR, DATA_DIR
from shared.reader import load_static_orography


LEGACY_MAPPING_JSON_NAMES = [
    "icegaws_station_mapping.json",
    "test_pixel_map.json",
]
LEGACY_DETECTOR_JSON_NAMES = [
    "icegaws_katabatic_results.json",
]


def _load_json_payload(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_json_path(path: Path, legacy_names):
    path = Path(path)
    if path.exists():
        return path
    for name in legacy_names:
        candidate = path.parent / name
        if candidate.exists():
            return candidate
    return path


def _resolve_with_legacy_fallback_if_default(arg_value: str, default_value: str, legacy_names):
    requested = Path(arg_value)
    if requested.exists():
        return requested
    # Only fall back to legacy names when the user kept the default path.
    if Path(arg_value) == Path(default_value):
        return _resolve_existing_json_path(requested, legacy_names)
    return requested


def _write_json_payload(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _station_codes_from_payload(payload):
    if not payload:
        return set()
    return {str(code).strip() for code in payload.get("stations", {}).keys()}


def _merge_station_payloads(existing_payload, new_payload, generated_at_utc=None):
    if existing_payload is None:
        return new_payload

    merged = dict(existing_payload)
    merged["stations"] = dict(existing_payload.get("stations", {}))
    merged["stations"].update(new_payload.get("stations", {}))

    if "carra_reference" not in merged and new_payload.get("carra_reference") is not None:
        merged["carra_reference"] = new_payload["carra_reference"]

    if generated_at_utc is not None:
        merged["generated_at_utc"] = generated_at_utc
    elif new_payload.get("generated_at_utc") is not None:
        merged["generated_at_utc"] = new_payload["generated_at_utc"]

    return merged


def _detector_config(
    window_size,
    dT_threshold,
    wind_threshold,
    direction_threshold,
    temperature_column,
    wind_speed_column,
    wind_direction_column,
    temperature_reference_mode,
    carra_time_match_tolerance,
    resample_step,
):
    return {
        "name": "katabatic_station_detector",
        "window_size": int(window_size),
        "dT_threshold": float(dT_threshold),
        "wind_threshold": float(wind_threshold),
        "direction_threshold": float(direction_threshold),
        "temperature_column": temperature_column,
        "wind_speed_column": wind_speed_column,
        "wind_direction_column": wind_direction_column,
        "temperature_reference_mode": temperature_reference_mode,
        "carra_time_match_tolerance": carra_time_match_tolerance,
        "resample_step": resample_step,
    }


def _detector_run_id(detector_config):
    serialized = json.dumps(detector_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]
    return f"cfg_{digest}"


def _migrate_legacy_detector_payload(existing_payload):
    if existing_payload is None:
        return None

    if isinstance(existing_payload.get("detector_runs"), dict):
        return existing_payload

    legacy_detector = existing_payload.get("detector", {})
    legacy_stations = existing_payload.get("stations", {})
    if not legacy_detector and not legacy_stations:
        return existing_payload

    run_id = _detector_run_id(legacy_detector)
    migrated = dict(existing_payload)
    migrated["detector_runs"] = {
        run_id: {
            "detector": legacy_detector,
            "stations": dict(legacy_stations),
            "carra_reference": existing_payload.get("carra_reference"),
            "generated_at_utc": existing_payload.get("generated_at_utc"),
        }
    }
    migrated["active_run_id"] = run_id
    return migrated


def _detector_station_codes_for_config(existing_payload, detector_config):
    payload = _migrate_legacy_detector_payload(existing_payload)
    if payload is None:
        return set()

    run_id = _detector_run_id(detector_config)
    run_payload = payload.get("detector_runs", {}).get(run_id, {})
    stations = run_payload.get("stations", {})
    return {str(code).strip() for code in stations.keys()}


def _detector_station_signatures_for_config(existing_payload, detector_config):
    payload = _migrate_legacy_detector_payload(existing_payload)
    if payload is None:
        return {}

    run_id = _detector_run_id(detector_config)
    run_payload = payload.get("detector_runs", {}).get(run_id, {})
    return dict(run_payload.get("station_signatures", {}))


def _set_detector_station_signatures_for_config(detector_payload, detector_config, station_signatures):
    run_id = _detector_run_id(detector_config)
    payload = _migrate_legacy_detector_payload(detector_payload)
    if payload is None:
        return detector_payload

    detector_runs = dict(payload.get("detector_runs", {}))
    run_payload = dict(detector_runs.get(run_id, {}))
    run_payload["station_signatures"] = dict(station_signatures)
    detector_runs[run_id] = run_payload
    payload["detector_runs"] = detector_runs
    return payload


def _station_signature(station_code, station_files, detector_config, carra_data_dir):
    payload = {
        "schema_version": 1,
        "station": str(station_code),
        "detector_config": detector_config,
        "carra_data_dir": path_fingerprint(carra_data_dir),
        "l3_files": [path_fingerprint(path) for path in sorted(station_files)],
    }
    return compute_signature(payload)


def _merge_detector_payloads(existing_payload, new_payload, detector_config):
    run_id = _detector_run_id(detector_config)

    if existing_payload is None:
        merged = dict(new_payload)
        merged["detector_runs"] = {
            run_id: {
                "detector": detector_config,
                "stations": dict(new_payload.get("stations", {})),
                "carra_reference": new_payload.get("carra_reference"),
                "generated_at_utc": new_payload.get("generated_at_utc"),
            }
        }
        merged["active_run_id"] = run_id
        merged["detector"] = detector_config
        merged["stations"] = dict(new_payload.get("stations", {}))
        return merged

    normalized = _migrate_legacy_detector_payload(existing_payload)
    merged = dict(normalized)

    detector_runs = dict(merged.get("detector_runs", {}))
    existing_run = dict(detector_runs.get(run_id, {}))
    merged_run_stations = dict(existing_run.get("stations", {}))
    merged_run_stations.update(new_payload.get("stations", {}))

    merged_run = {
        "detector": detector_config,
        "stations": merged_run_stations,
        "carra_reference": new_payload.get("carra_reference", existing_run.get("carra_reference")),
        "generated_at_utc": new_payload.get("generated_at_utc", existing_run.get("generated_at_utc")),
    }
    detector_runs[run_id] = merged_run

    merged["detector_runs"] = detector_runs
    merged["active_run_id"] = run_id
    # Keep root-level aliases for backward compatibility with downstream readers.
    merged["detector"] = merged_run["detector"]
    merged["stations"] = merged_run["stations"]
    if merged_run.get("carra_reference") is not None:
        merged["carra_reference"] = merged_run["carra_reference"]
    if merged_run.get("generated_at_utc") is not None:
        merged["generated_at_utc"] = merged_run["generated_at_utc"]
    return merged


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a JSON map from ICE-GAWS stations to the nearest CARRA grid pixels and "
            "compute a station-level katabatic event detector."
        )
    )
    parser.add_argument(
        "--icegaws-data-dir",
        default=str(DATA_DIR / "ICE-GAWS"),
        help="Folder containing ICE-GAWS L3 CSV files.",
    )
    parser.add_argument(
        "--location-summary",
        default=str(ICEGAWS_LOCATION_SUMMARY),
        help="ICE-GAWS location summary CSV.",
    )
    parser.add_argument(
        "--carra-data-dir",
        default=str(DEFAULT_KATABATIC_DATA_DIR),
        help="Root folder containing CARRA NetCDF files.",
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
        help="Reference NetCDF file index used to load the CARRA grid.",
    )
    default_mapping_json = str(DATA_DIR / "ICE-GAWS" / "icegaws_carra_pixel_basin_mapping.json")
    default_results_json = str(DATA_DIR / "ICE-GAWS" / "icegaws_katabatic_detector_results.json")

    parser.add_argument(
        "--output-mapping-json",
        default=default_mapping_json,
        help="Output JSON path for the station-to-pixel and basin mapping.",
    )
    parser.add_argument(
        "--output-results-json",
        default=default_results_json,
        help="Output JSON path for the detector results.",
    )
    parser.add_argument(
        "--basins-path",
        default=str(DATA_DIR / "NPY" / "basins.npy"),
        help="Path to the saved katabatic basins.npy file produced by the katabatic pipeline.",
    )
    parser.add_argument(
        "--station",
        nargs="*",
        default=None,
        help="Optional station codes to include only a subset of stations.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=15,
        help="Window size used by the station temperature anomaly detector.",
    )
    parser.add_argument(
        "--dT-threshold",
        type=float,
        default=1.5,
        help="Temperature anomaly threshold in °C.",
    )
    parser.add_argument(
        "--wind-threshold",
        type=float,
        default=2.5,
        help="Minimum wind speed in m/s.",
    )
    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=0.87,
        help="Minimum cosine alignment with downslope direction.",
    )
    parser.add_argument(
        "--temperature-column",
        default="t_2m",
        help="Temperature column to use from the ICE-GAWS CSV.",
    )
    parser.add_argument(
        "--wind-speed-column",
        default="ws_4m",
        help="Wind speed column to use from the ICE-GAWS CSV.",
    )
    parser.add_argument(
        "--wind-direction-column",
        default="wd_4m",
        help="Wind direction column to use from the ICE-GAWS CSV.",
    )
    parser.add_argument(
        "--temperature-reference-mode",
        default="spatial_carra",
        choices=["spatial_carra"],
        help=(
            "Temperature reference mode: 'spatial_carra' uses CARRA2 spatial local mean "
            "around the mapped station pixel."
        ),
    )
    parser.add_argument(
        "--carra-time-match-tolerance",
        default="2h",
        help="Nearest-time tolerance used to align station timestamps with CARRA times in spatial_carra mode.",
    )
    parser.add_argument(
        "--reuse-if-unchanged",
        dest="reuse_if_unchanged",
        action="store_true",
        help="Reuse station detector outputs when the station input signature is unchanged.",
    )
    parser.add_argument(
        "--no-reuse-if-unchanged",
        dest="reuse_if_unchanged",
        action="store_false",
        help="Disable cache reuse and recompute detector outputs for requested stations.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force detector recomputation for all requested stations.",
    )
    parser.set_defaults(reuse_if_unchanged=True)

    args = parser.parse_args()

    detector_config = _detector_config(
        args.window_size,
        args.dT_threshold,
        args.wind_threshold,
        args.direction_threshold,
        args.temperature_column,
        args.wind_speed_column,
        args.wind_direction_column,
        args.temperature_reference_mode,
        args.carra_time_match_tolerance,
        "3h",
    )

    location_summary_path = Path(args.location_summary)
    station_locations = load_station_locations(location_summary_path)

    carra_grid = load_static_orography(
        Path(args.carra_data_dir),
        year=args.carra_year,
        month=args.carra_month,
        file_index=args.carra_file_index,
    )

    mapping_path = _resolve_with_legacy_fallback_if_default(
        arg_value=args.output_mapping_json,
        default_value=default_mapping_json,
        legacy_names=LEGACY_MAPPING_JSON_NAMES,
    )
    detector_path = _resolve_with_legacy_fallback_if_default(
        arg_value=args.output_results_json,
        default_value=default_results_json,
        legacy_names=LEGACY_DETECTOR_JSON_NAMES,
    )
    existing_mapping = _load_json_payload(mapping_path)
    existing_detector = _load_json_payload(detector_path)

    if args.station is not None:
        requested_stations = {str(code).strip() for code in args.station}
    else:
        requested_stations = set(station_locations["station"].astype(str).map(str.strip))

    mapping_existing = _station_codes_from_payload(existing_mapping)
    detector_existing = _detector_station_codes_for_config(existing_detector, detector_config)
    detector_signatures_existing = _detector_station_signatures_for_config(existing_detector, detector_config)
    l3_files_by_station = discover_l3_files(data_dir=Path(args.icegaws_data_dir))

    station_signatures_current = {}
    for station_code in sorted(requested_stations):
        station_files = l3_files_by_station.get(station_code, [])
        if not station_files:
            continue
        station_signatures_current[station_code] = _station_signature(
            station_code=station_code,
            station_files=station_files,
            detector_config=detector_config,
            carra_data_dir=Path(args.carra_data_dir),
        )

    mapping_missing = sorted(requested_stations - mapping_existing)
    detector_stale = sorted(
        station
        for station in requested_stations
        if station in detector_existing
        and station_signatures_current.get(station) is not None
        and detector_signatures_existing.get(station) != station_signatures_current.get(station)
    )

    if args.force_recompute or (not args.reuse_if_unchanged):
        detector_missing = sorted(requested_stations)
    else:
        detector_missing = sorted((requested_stations - detector_existing).union(detector_stale))

    if not mapping_missing and not detector_missing:
        print("[icegaws] All requested stations already exist in both JSON files; nothing to compute.")
        print(f"[icegaws] Mapping JSON: {mapping_path}")
        print(f"[icegaws] Detector JSON: {detector_path}")
        return

    mapping_payload = existing_mapping
    if mapping_missing:
        new_mapping = build_station_static_mapping_json(
            carra_orography_ds=carra_grid,
            station_locations=station_locations,
            station_codes=mapping_missing,
            data_dir=Path(args.icegaws_data_dir),
            location_summary_path=location_summary_path,
            basins_path=Path(args.basins_path),
        )
        mapping_payload = _merge_station_payloads(mapping_payload, new_mapping)
        _write_json_payload(mapping_path, mapping_payload)
        print(f"[icegaws] Added {len(mapping_missing)} missing station(s) to mapping JSON.")
    else:
        print("[icegaws] Mapping JSON already contains all requested stations.")

    detector_payload = existing_detector
    if detector_missing:
        new_detector = build_station_katabatic_detector_json(
            carra_orography_ds=carra_grid,
            station_locations=station_locations,
            station_codes=detector_missing,
            data_dir=Path(args.icegaws_data_dir),
            location_summary_path=location_summary_path,
            basins_path=Path(args.basins_path),
            window_size=args.window_size,
            dT_threshold=args.dT_threshold,
            wind_threshold=args.wind_threshold,
            direction_threshold=args.direction_threshold,
            temperature_column=args.temperature_column,
            wind_speed_column=args.wind_speed_column,
            wind_direction_column=args.wind_direction_column,
            temperature_reference_mode=args.temperature_reference_mode,
            carra_t2m_data_dir=Path(args.carra_data_dir),
            carra_time_match_tolerance=args.carra_time_match_tolerance,
        )
        detector_payload = _merge_detector_payloads(detector_payload, new_detector, detector_config)
        merged_signatures = dict(detector_signatures_existing)
        for station_code in detector_missing:
            if station_code in station_signatures_current:
                merged_signatures[station_code] = station_signatures_current[station_code]
        detector_payload = _set_detector_station_signatures_for_config(
            detector_payload,
            detector_config,
            merged_signatures,
        )
        _write_json_payload(detector_path, detector_payload)
        print(f"[icegaws] Added {len(detector_missing)} missing station(s) to detector JSON.")
    else:
        print("[icegaws] Detector JSON already contains all requested stations.")

    mapping_path = mapping_path
    results_path = detector_path
    print(f"[icegaws] Wrote station mapping JSON to: {mapping_path}")
    print(f"[icegaws] Wrote detector results JSON to: {results_path}")


if __name__ == "__main__":
    main()