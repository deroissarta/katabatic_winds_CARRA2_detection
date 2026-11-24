"""Compute and plot quantiles from cached katabatic classification outputs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import os

# Force non-interactive plotting backend for batch quantile map generation.
os.environ.setdefault("KATABATIC_FORCE_AGG", "1")

from shared.map_launcher import plot_iceland
from shared.paths import (
    DEFAULT_CARRA_DETECTOR_DIR,
    DEFAULT_FIGURES_DIR,
    DEFAULT_NPY_DATA_DIR,
    resolve_glacier_mask_path,
)


def _compute_quantiles(values):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "p05": float(np.quantile(valid, 0.05)),
        "p25": float(np.quantile(valid, 0.25)),
        "p50": float(np.quantile(valid, 0.50)),
        "p75": float(np.quantile(valid, 0.75)),
        "p90": float(np.quantile(valid, 0.90)),
        "p95": float(np.quantile(valid, 0.95)),
        "max": float(np.max(valid)),
    }


def _resolve_aggregate_dir(base_npy_dir, param_tag=None, period=None):
    root = Path(base_npy_dir) / "yearly"
    if param_tag and period:
        candidate = root / str(param_tag) / "aggregated" / str(period)
        if not candidate.exists():
            raise FileNotFoundError(f"Aggregate directory not found: {candidate}")
        return candidate

    if param_tag and not period:
        tag_dir = root / str(param_tag) / "aggregated"
        if not tag_dir.exists():
            raise FileNotFoundError(f"Param tag directory not found: {tag_dir}")
        period_dirs = [p for p in tag_dir.iterdir() if p.is_dir()]
        if not period_dirs:
            raise FileNotFoundError(f"No period directories found under: {tag_dir}")
        return max(period_dirs, key=lambda p: p.stat().st_mtime)

    all_period_dirs = [p for p in root.glob("*/aggregated/*") if p.is_dir()]
    if not all_period_dirs:
        raise FileNotFoundError(f"No aggregated NPY directories found under: {root}")
    return max(all_period_dirs, key=lambda p: p.stat().st_mtime)


def _load_times_count(times_path):
    if times_path is None:
        return None
    times_path = Path(times_path)
    if not times_path.exists():
        return None
    times = np.load(times_path, allow_pickle=True)
    return int(len(times))


def _infer_period_from_times(times_path, fallback_period):
    if times_path is None:
        return str(fallback_period)

    times_path = Path(times_path)
    if not times_path.exists():
        return str(fallback_period)

    try:
        times = np.load(times_path, allow_pickle=True)
    except Exception:
        return str(fallback_period)

    years = []
    for value in np.ravel(times):
        text = str(value)
        if "_" in text:
            prefix = text.split("_", 1)[0]
        else:
            prefix = text[:4]
        if len(prefix) == 4 and prefix.isdigit():
            years.append(int(prefix))

    if not years:
        return str(fallback_period)

    years = sorted(set(years))
    return f"{years[0]} - {years[-1]}"


def build_quantiles_payload(per_kat, label_map, n_timestamps, param_tag, period, input_sources):
    values = np.asarray(per_kat, dtype=float)
    labels = np.asarray(label_map)
    if values.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch between per_kat {values.shape} and label_map {labels.shape}"
        )

    basin_quantiles_all = []
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    for basin_label in unique_labels:
        basin_mask = labels == basin_label
        basin_values = values[basin_mask]
        basin_values = basin_values[np.isfinite(basin_values)]
        if basin_values.size == 0:
            continue

        nb_pixels_total = int(basin_values.size)
        nb_pixels_actifs = int(np.sum(basin_values > 0.0))
        intensity_relative = float(np.sum(basin_values))

        if n_timestamps is None:
            intensity = intensity_relative
            intensity_mode = "relative_sum_fraction"
        else:
            intensity = float(intensity_relative * int(n_timestamps))
            intensity_mode = "pixel_timestamps"

        basin_entry = {
            "label": int(basin_label),
            "nb_pixels_total": nb_pixels_total,
            "nb_pixels_actifs": nb_pixels_actifs,
            "fraction_active_pixels": float(nb_pixels_actifs / nb_pixels_total),
            "fraction_catabatique_moyenne_bassin": float(np.mean(basin_values)),
            "intensite_temporelle_bassin": intensity,
            "intensite_mode": intensity_mode,
        }
        basin_entry.update(_compute_quantiles(basin_values))
        basin_quantiles_all.append(basin_entry)

    basin_quantiles_all.sort(key=lambda row: row["label"])

    return {
        "metadata": {
            "dataset": "CARRA2",
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "param_tag": str(param_tag),
            "period": str(period),
            "quantiles_scope": "whole_aggregated_period",
            "grid_shape": [int(values.shape[0]), int(values.shape[1])],
            "nb_timestamps_detectes": None if n_timestamps is None else int(n_timestamps),
        },
        "input_sources": {
            "per_kat": str(input_sources["per_kat"]),
            "label_map": str(input_sources["label_map"]),
            "times": None if input_sources.get("times") is None else str(input_sources["times"]),
        },
        "global_quantiles_iceland": _compute_quantiles(values),
        "basin_quantiles_all": basin_quantiles_all,
    }


def _build_quantile_maps(per_kat, label_map):
    values = np.asarray(per_kat, dtype=float)
    labels = np.asarray(label_map)
    if values.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch between per_kat {values.shape} and label_map {labels.shape}"
        )

    q50_map = np.full(values.shape, np.nan, dtype=np.float32)
    q90_map = np.full(values.shape, np.nan, dtype=np.float32)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    for basin_label in unique_labels:
        basin_mask = labels == basin_label
        basin_values = values[basin_mask]
        basin_values = basin_values[np.isfinite(basin_values)]
        if basin_values.size == 0:
            continue
        q50 = float(np.quantile(basin_values, 0.50))
        q90 = float(np.quantile(basin_values, 0.90))
        q50_map[basin_mask] = q50
        q90_map[basin_mask] = q90

    return q50_map, q90_map


def _plot_quantile_maps(per_kat, label_map, period, param_tag, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    q50_map, q90_map = _build_quantile_maps(per_kat, label_map)
    per_kat_values = np.asarray(per_kat, dtype=float)
    label_values = np.asarray(label_map)
    basin_valid = np.isfinite(per_kat_values) & (label_values > 0)
    if np.any(basin_valid):
        global_q20 = float(np.quantile(per_kat_values[basin_valid], 0.20))
        global_q50 = float(np.quantile(per_kat_values[basin_valid], 0.50))
        global_p95 = float(np.quantile(per_kat_values[basin_valid], 0.95))
    else:
        global_q20 = np.nan
        global_q50 = np.nan
        global_p95 = np.nan

    try:
        glaciers = np.load(resolve_glacier_mask_path())
    except Exception:
        glaciers = None

    q20_mask = np.where(basin_valid & (per_kat_values <= global_q20), per_kat_values, np.nan)
    plot_iceland(
        q20_mask,
        "per_kat",
        typ="mask",
        ref=None,
        unit="fraction",
        cmap="Blues",
        glaciers=glaciers,
        save_path=figures_dir / "global_q20_coldspots_map.png",
        show=False,
        title=f"Global q20 coldspots - {period}",
        coasts="white",
    )

    q50_mask = np.where(basin_valid & (per_kat_values <= global_q50), per_kat_values, np.nan)
    plot_iceland(
        q50_mask,
        "per_kat",
        typ="mask",
        ref=None,
        unit="fraction",
        cmap="Blues",
        glaciers=glaciers,
        save_path=figures_dir / "global_q50_coldspots_map.png",
        show=False,
        title=f"Global q50 coldspots - {period}",
        coasts="white",
    )

    plot_iceland(
        q50_map,
        "per_kat",
        typ="mask",
        ref=None,
        unit="fraction",
        cmap="magma",
        glaciers=glaciers,
        save_path=figures_dir / "basin_q50_map.png",
        show=False,
        title=f"Basin q50 map - {period}",
        coasts="white",
    )

    plot_iceland(
        q90_map,
        "per_kat",
        typ="mask",
        ref=None,
        unit="fraction",
        cmap="inferno",
        glaciers=glaciers,
        save_path=figures_dir / "basin_q90_map.png",
        show=False,
        title=f"Basin q90 map - {period}",
        coasts="white",
    )

    p95_mask = np.where(basin_valid & (per_kat_values >= global_p95), per_kat_values, np.nan)
    plot_iceland(
        p95_mask,
        "per_kat",
        typ="mask",
        ref=None,
        unit="fraction",
        cmap="plasma",
        glaciers=glaciers,
        save_path=figures_dir / "global_p95_hotspots_map.png",
        show=False,
        title=f"Global p95 hotspots - {period}",
        coasts="white",
    )

    summary = {
        "param_tag": str(param_tag),
        "period": str(period),
        "global_q20": None if np.isnan(global_q20) else float(global_q20),
        "global_q50": None if np.isnan(global_q50) else float(global_q50),
        "global_p95": None if np.isnan(global_p95) else float(global_p95),
        "global_quantile_scope": "basin_pixels_only",
        "q20_pixel_count": int(np.sum(np.isfinite(q20_mask))),
        "q50_pixel_count": int(np.sum(np.isfinite(q50_mask))),
        "q95_pixel_count": int(np.sum(np.isfinite(p95_mask))),
    }
    (figures_dir / "quantile_map_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_quantiles_npy(payload, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_q = payload["global_quantiles_iceland"]
    global_order = ["mean", "p05", "p25", "p50", "p75", "p90", "p95", "max"]
    global_values = np.array(
        [np.nan if global_q.get(k) is None else float(global_q[k]) for k in global_order],
        dtype=np.float64,
    )
    np.save(output_dir / "global_quantiles_values.npy", global_values)

    basin_rows = payload.get("basin_quantiles_all", [])
    basin_labels = np.array([int(row["label"]) for row in basin_rows], dtype=np.int32)
    np.save(output_dir / "basin_labels.npy", basin_labels)

    basin_order = [
        "min",
        "p05",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "max",
        "mean",
        "fraction_active_pixels",
        "intensite_temporelle_bassin",
    ]

    basin_matrix = np.full((len(basin_rows), len(basin_order)), np.nan, dtype=np.float64)
    for i, row in enumerate(basin_rows):
        for j, key in enumerate(basin_order):
            value = row.get(key)
            basin_matrix[i, j] = np.nan if value is None else float(value)
    np.save(output_dir / "basin_quantiles_matrix.npy", basin_matrix)

    np.save(output_dir / "global_quantiles_keys.npy", np.array(global_order, dtype=object))
    np.save(output_dir / "basin_quantiles_keys.npy", np.array(basin_order, dtype=object))


def main():
    parser = argparse.ArgumentParser(
        description="Compute CARRA2 katabatic quantiles from aggregated NPY outputs."
    )
    parser.add_argument("--param-tag", default=None, help="Parameter tag (e.g. dt1.5_ws15_wind2.5_dir0.87)")
    parser.add_argument("--period", default=None, help="Period folder name (e.g. 2020 - 2025)")
    parser.add_argument("--aggregate-dir", default=None, help="Explicit aggregate directory containing NPY files")
    parser.add_argument("--per-kat", default=None, help="Explicit per_kat NPY path")
    parser.add_argument("--label-map", default=None, help="Explicit label_map NPY path")
    parser.add_argument("--times", default=None, help="Optional times NPY path")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CARRA_DETECTOR_DIR / "carra2_katabatic_quantiles.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(DEFAULT_FIGURES_DIR / "yearly"),
        help="Base directory where quantile plots are saved",
    )
    parser.add_argument(
        "--save-plots",
        dest="save_plots",
        action="store_true",
        help="Save quantiles plots",
    )
    parser.add_argument(
        "--no-save-plots",
        dest="save_plots",
        action="store_false",
        help="Disable quantiles plot saving",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Save quantiles arrays as NPY files in the aggregate directory",
    )
    parser.set_defaults(save_plots=True)
    args = parser.parse_args()

    if args.aggregate_dir:
        aggregate_dir = Path(args.aggregate_dir)
    else:
        aggregate_dir = _resolve_aggregate_dir(
            base_npy_dir=DEFAULT_NPY_DATA_DIR,
            param_tag=args.param_tag,
            period=args.period,
        )

    if not aggregate_dir.exists():
        raise FileNotFoundError(f"Aggregate directory not found: {aggregate_dir}")

    if args.param_tag:
        resolved_param_tag = args.param_tag
    else:
        resolved_param_tag = aggregate_dir.parent.parent.name
    resolved_period = args.period or aggregate_dir.name

    per_kat_path = Path(args.per_kat) if args.per_kat else aggregate_dir / f"per_kat_{resolved_param_tag}.npy"
    label_map_path = Path(args.label_map) if args.label_map else aggregate_dir / f"label_map_{resolved_param_tag}.npy"

    if args.times:
        times_path = Path(args.times)
    else:
        candidate_times = aggregate_dir / f"times_{resolved_param_tag}.npy"
        times_path = candidate_times if candidate_times.exists() else None

    resolved_period = _infer_period_from_times(times_path, resolved_period)

    if not per_kat_path.exists():
        raise FileNotFoundError(f"Missing per_kat file: {per_kat_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing label_map file: {label_map_path}")

    per_kat = np.load(per_kat_path)
    label_map = np.load(label_map_path)
    n_timestamps = _load_times_count(times_path)

    payload = build_quantiles_payload(
        per_kat=per_kat,
        label_map=label_map,
        n_timestamps=n_timestamps,
        param_tag=resolved_param_tag,
        period=resolved_period,
        input_sources={
            "per_kat": per_kat_path,
            "label_map": label_map_path,
            "times": times_path,
        },
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[katabatic] Quantiles JSON written to: {output_path}")

    if args.save_plots:
        figures_dir = Path(args.plots_dir) / str(resolved_param_tag) / "aggregated" / "STAT"
        _plot_quantile_maps(
            per_kat=per_kat,
            label_map=label_map,
            period=resolved_period,
            param_tag=resolved_param_tag,
            figures_dir=figures_dir,
        )
        print(f"[katabatic] Quantiles figures written to: {figures_dir}")

    if args.save_npy:
        quantiles_npy_dir = aggregate_dir / "quantiles_npy"
        _save_quantiles_npy(payload, quantiles_npy_dir)
        print(f"[katabatic] Quantiles NPY written to: {quantiles_npy_dir}")


if __name__ == "__main__":
    main()
