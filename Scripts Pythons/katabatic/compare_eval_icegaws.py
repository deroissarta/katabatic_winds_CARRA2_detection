"""Summarize and compare saved ICE-GAWS evaluation results."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from shared.paths import DATA_DIR, DEFAULT_KATABATIC_DATA_DIR, DEFAULT_NPY_DATA_DIR


COMPARE_EVAL_CONFIG_STATE_PATH = DATA_DIR / "ICE-GAWS" / ".compare_eval_last_compute_config.json"


def _discover_latest_per_kat_path():
    yearly_candidates = sorted(DEFAULT_NPY_DATA_DIR.glob("yearly/*/aggregated/*/per_kat_*.npy"))
    if yearly_candidates:
        return yearly_candidates[-1]

    concat_candidates = sorted(DEFAULT_NPY_DATA_DIR.glob("concat/*/*/per_kat_*.npy"))
    if concat_candidates:
        return concat_candidates[-1]

    return None


def _run_module(module_name, args):
    command = [sys.executable, "-m", module_name, *args]
    print(f"[compare_eval] Run: {' '.join(command)}")
    subprocess.run(command, check=True)


def _normalize_csv_arg(value):
    if value is None:
        return None
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    return sorted(parts)


def _normalize_station_arg(stations):
    if not stations:
        return None
    return sorted(str(station) for station in stations)


def _build_compute_config(args):
    return {
        "carra_data_dir": str(args.carra_data_dir),
        "katabatic_mode": args.katabatic_mode,
        "per_kat_path": str(args.per_kat_path) if args.per_kat_path is not None else None,
        "mapping_json": str(args.mapping_json),
        "detector_json": str(args.detector_json),
        "output_json": str(args.output_json),
        "output_csv": str(args.output_csv),
        "stat_output_dir": str(args.stat_output_dir) if args.stat_output_dir is not None else None,
        "station": _normalize_station_arg(args.station),
        "years": _normalize_csv_arg(args.years),
        "dT_threshold": float(args.dT_threshold),
        "wind_threshold": float(args.wind_threshold),
        "direction_threshold": float(args.direction_threshold),
        "window_size": int(args.window_size),
        "full_evaluation": bool(args.full_evaluation),
        "generate_stat_maps": bool(args.generate_stat_maps),
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
    previous_config = _load_last_compute_config(COMPARE_EVAL_CONFIG_STATE_PATH)
    config_changed = previous_config != compute_config

    if config_changed:
        print("[compare_eval] New compute configuration detected -> force full recompute.")
        args.force_recompute = True
        args.refresh_detector = True
    else:
        print("[compare_eval] Compute configuration unchanged -> cache reuse allowed.")

    return compute_config, config_changed


def _resolve_per_kat_path(args):
    if args.per_kat_path is not None:
        candidate = Path(args.per_kat_path)
        if candidate.exists():
            return candidate

        if not args.auto_run_katabatic_if_missing:
            raise FileNotFoundError(f"per_kat file not found: {candidate}")

        print(f"[compare_eval] Missing per_kat file: {candidate}")
        print("[compare_eval] Launch katabatic classification to build missing per_kat.")
        _run_katabatic_if_needed(args)
        if candidate.exists():
            return candidate

        latest = _discover_latest_per_kat_path()
        if latest is None:
            raise FileNotFoundError("No per_kat file found after katabatic classification run.")
        return latest

    latest = _discover_latest_per_kat_path()
    if latest is not None:
        return latest

    if not args.auto_run_katabatic_if_missing:
        raise FileNotFoundError(
            f"No per_kat files found under {DEFAULT_NPY_DATA_DIR / 'yearly'} or {DEFAULT_NPY_DATA_DIR / 'concat'}"
        )

    print("[compare_eval] No per_kat found. Launch katabatic classification.")
    _run_katabatic_if_needed(args)
    latest = _discover_latest_per_kat_path()
    if latest is None:
        raise FileNotFoundError("No per_kat file found after katabatic classification run.")
    return latest


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


def _ensure_mapping_and_detector(args):
    mapping_path = Path(args.mapping_json)
    detector_path = Path(args.detector_json)

    should_refresh = (
        args.force_recompute
        or args.refresh_detector
        or (not mapping_path.exists())
        or (not detector_path.exists())
    )
    if not should_refresh:
        print("[compare_eval] Mapping and detector files found. Skip refresh.")
        return mapping_path, detector_path

    print("[compare_eval] Build or refresh ICE-GAWS mapping/detector prerequisites.")
    command_args = [
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
    ]

    if args.station:
        command_args.append("--station")
        command_args.extend(args.station)

    if args.force_recompute:
        command_args.append("--force-recompute")
    else:
        command_args.append("--reuse-if-unchanged")

    _run_module("katabatic.build_icegaws_carra_pixel_map", command_args)

    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping JSON not found after build: {mapping_path}")
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector JSON not found after build: {detector_path}")

    return mapping_path, detector_path


def _run_evaluation(args, per_kat_path, mapping_path):
    command_args = [
        "--carra-data-dir",
        str(args.carra_data_dir),
        "--mapping-json",
        str(mapping_path),
        "--detector-json",
        str(args.detector_json),
        "--output-json",
        str(args.output_json),
        "--output-csv",
        str(args.output_csv),
        "--carra-per-kat-path",
        str(per_kat_path),
        "--dT-thresholds",
        str(args.dT_threshold),
        "--wind-thresholds",
        str(args.wind_threshold),
        "--direction-thresholds",
        str(args.direction_threshold),
    ]

    if args.years:
        command_args.extend(["--years", args.years])

    if args.station:
        command_args.append("--station")
        command_args.extend(args.station)

    if args.stat_output_dir is not None:
        command_args.extend(["--stat-output-dir", str(args.stat_output_dir)])

    if args.full_evaluation:
        command_args = [arg for arg in command_args if arg != "--detector-json" and arg != str(args.detector_json)]
        if args.generate_stat_maps:
            command_args.append("--generate-stat-maps")
        else:
            command_args.append("--no-generate-stat-maps")
    else:
        command_args.append("--no-generate-stat-maps")

    if args.force_recompute:
        command_args.append("--force-recompute")
    else:
        command_args.append("--reuse-if-unchanged")

    try:
        _run_module("katabatic.evaluate_icegaws_katabatic", command_args)
    except subprocess.CalledProcessError as exc:
        # Some Windows environments can crash natively in plotting libs.
        if int(exc.returncode) == 3228369023 and args.generate_stat_maps and args.full_evaluation:
            print(
                "[compare_eval] Native plotting crash detected (3228369023). "
                "Retry evaluation without stat maps."
            )
            retry_args = [arg for arg in command_args if arg != "--generate-stat-maps"]
            if "--no-generate-stat-maps" not in retry_args:
                retry_args.append("--no-generate-stat-maps")
            _run_module("katabatic.evaluate_icegaws_katabatic", retry_args)
            return
        raise


def main():
    default_mapping_json = DATA_DIR / "ICE-GAWS" / "icegaws_carra_pixel_basin_mapping.json"
    default_detector_json = DATA_DIR / "ICE-GAWS" / "icegaws_katabatic_detector_results.json"
    default_output_json = DATA_DIR / "ICE-GAWS" / "icegaws_katabatic_validation_metrics.json"
    default_output_csv = DATA_DIR / "ICE-GAWS" / "icegaws_katabatic_threshold_sweep.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Eval launcher similar to compare: ensure missing prerequisites, then run "
            "ICE-GAWS evaluation and generate graphs."
        )
    )
    parser.add_argument("--per-kat-path", default=None, help="Optional explicit per_kat .npy path.")
    parser.add_argument("--carra-data-dir", default=str(DEFAULT_KATABATIC_DATA_DIR), help="CARRA2 data root.")
    parser.add_argument(
        "--katabatic-mode",
        choices=["yearly", "concat"],
        default="yearly",
        help="Mode used when auto-running katabatic classification to build missing per_kat.",
    )

    parser.add_argument("--mapping-json", default=str(default_mapping_json), help="Mapping JSON path.")
    parser.add_argument("--detector-json", default=str(default_detector_json), help="Detector JSON path.")
    parser.add_argument("--output-json", default=str(default_output_json), help="Evaluation output JSON path.")
    parser.add_argument(
        "--output-csv",
        default=str(default_output_csv),
        help="Evaluation station CSV path in detector mode, or single-config metrics CSV in full evaluation mode.",
    )
    parser.add_argument("--stat-output-dir", default=None, help="Optional STAT output directory override.")

    parser.add_argument("--station", nargs="*", default=None, help="Optional station subset.")
    parser.add_argument("--years", default=None, help="Optional years filter as comma-separated list.")

    parser.add_argument("--dT-threshold", type=float, default=1.5, help="Detector dT threshold.")
    parser.add_argument("--wind-threshold", type=float, default=2.5, help="Detector wind threshold.")
    parser.add_argument("--direction-threshold", type=float, default=0.87, help="Detector direction threshold.")
    parser.add_argument("--window-size", type=int, default=15, help="Detector window size.")

    parser.add_argument(
        "--generate-stat-maps",
        dest="generate_stat_maps",
        action="store_true",
        help="Generate and save evaluation station maps (used in full evaluation mode).",
    )
    parser.add_argument(
        "--no-generate-stat-maps",
        dest="generate_stat_maps",
        action="store_false",
        help="Disable evaluation station maps generation.",
    )
    parser.set_defaults(generate_stat_maps=True)

    parser.add_argument(
        "--full-evaluation",
        action="store_true",
        help=(
            "Run full station-timeseries evaluation. By default this launcher runs detector-fraction-only "
            "evaluation for a fast first-pass."
        ),
    )

    parser.add_argument(
        "--refresh-detector",
        action="store_true",
        help="Force a detector/mapping refresh step even if JSON files already exist.",
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
    parser.set_defaults(auto_run_katabatic_if_missing=True)

    parser.add_argument("--force-recompute", action="store_true", help="Bypass cache and force recompute.")

    args = parser.parse_args()
    args.carra_data_dir = Path(args.carra_data_dir)
    args.mapping_json = Path(args.mapping_json)
    args.detector_json = Path(args.detector_json)
    args.output_json = Path(args.output_json)
    args.output_csv = Path(args.output_csv)
    args.stat_output_dir = Path(args.stat_output_dir) if args.stat_output_dir else None

    compute_config, config_changed = _apply_recompute_policy_from_config(args)
    if config_changed and args.per_kat_path is None and args.auto_run_katabatic_if_missing:
        print("[compare_eval] Refresh katabatic classification for the new configuration.")
        _run_katabatic_if_needed(args)

    per_kat_path = _resolve_per_kat_path(args)
    print(f"[compare_eval] Use per_kat: {per_kat_path}")
    if args.full_evaluation:
        print("[compare_eval] Mode: full station-timeseries evaluation")
    else:
        print("[compare_eval] Mode: detector-fraction-only evaluation (default)")

    mapping_path, _detector_path = _ensure_mapping_and_detector(args)
    _run_evaluation(args, per_kat_path=per_kat_path, mapping_path=mapping_path)
    _save_compute_config(COMPARE_EVAL_CONFIG_STATE_PATH, compute_config)


if __name__ == "__main__":
    main()
