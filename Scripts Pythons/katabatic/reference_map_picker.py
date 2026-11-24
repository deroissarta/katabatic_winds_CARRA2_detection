"""Select and load cached maps used as spatial analysis references."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared.paths import DEFAULT_FIGURES_DIR, DEFAULT_NPY_DATA_DIR

DEFAULT_REFERENCE_PARAM_TAG = "dt1.5_ws15_wind2.5_dir0.87"


def build_selection_file_path(param_tag: str | None = None, years=None, figures_root: Path | str = DEFAULT_FIGURES_DIR):
    """Build the active run-specific JSON path for a selected pixel.

    The selector itself may use a default cached reference map, but the persisted
    selection must remain under the same scientific output branch as the active run.
    """
    base_root = Path(figures_root)
    if param_tag is None or years is None:
        return base_root / "last_reference_pixel.json"

    years_int = sorted(int(str(y)) for y in years)
    if not years_int:
        return base_root / "last_reference_pixel.json"

    period = f"{years_int[0]} - {years_int[-1]}"
    return (
        base_root
        / "yearly"
        / str(param_tag)
        / "aggregated"
        / str(period)
        / "graphs"
        / "selection"
        / "last_reference_pixel.json"
    )


def _save_selected_pixel(selection_path: Path | str, pixel):
    path = Path(selection_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pixel": [int(pixel[0]), int(pixel[1])],
        "y": int(pixel[0]),
        "x": int(pixel[1]),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _event_to_pixel(event, shape):
    if event is None or event.xdata is None or event.ydata is None:
        return None

    x = int(round(event.xdata))
    y = int(round(event.ydata))
    if 0 <= y < shape[0] and 0 <= x < shape[1]:
        return (y, x)
    return None


def load_reference_map_from_cache(param_tag: str, years, base_dir: Path | str = DEFAULT_NPY_DATA_DIR):
    """Load the aggregated reference map that matches the default yearly cache layout."""
    base_dir = Path(base_dir)
    years_int = sorted(int(str(y)) for y in years)
    if not years_int:
        return None, None
    period = f"{years_int[0]} - {years_int[-1]}"
    aggregate_dir = base_dir / "yearly" / param_tag / "aggregated" / period
    per_kat_path = aggregate_dir / f"per_kat_{param_tag}.npy"
    label_map_path = aggregate_dir / f"label_map_{param_tag}.npy"
    if not per_kat_path.exists() or not label_map_path.exists():
        return None, None
    return np.load(per_kat_path), np.load(label_map_path)


def load_default_reference_map_from_cache(years, base_dir: Path | str = DEFAULT_NPY_DATA_DIR, param_tag: str = DEFAULT_REFERENCE_PARAM_TAG):
    """Load a single default cached reference map for the click-only workflow.

    This intentionally ignores the active run parameters and uses the project's default
    aggregated katabatic map as a lightweight reference structure for the pixel picker.
    """
    return load_reference_map_from_cache(param_tag, years, base_dir)[0]


def pick_pixel_from_reference_map(reference_map, title="Click a pixel on the katabatic fraction reference map", fallback_prompt=True, selection_file=None, param_tag=None, years=None, show=True):
    """Display a cached katabatic fraction map and return the clicked pixel as (y, x).

    The selector is intentionally lightweight and independent from the full scientific plot
    pipeline. It can still be forced to display even when the global --no-show option is set,
    because that option is meant to suppress heavy plotting during processing, not the manual
    pixel-selection map itself.
    """
    reference_map = np.asarray(reference_map)
    if reference_map.ndim != 2:
        raise ValueError("reference_map must be a 2D array")

    clicked = {}
    if selection_file is not None:
        target_file = Path(selection_file)
    else:
        target_file = build_selection_file_path(param_tag=param_tag, years=years, figures_root=DEFAULT_FIGURES_DIR)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(reference_map, cmap="plasma", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()

    def on_click(event):
        pixel = _event_to_pixel(event, reference_map.shape)
        if pixel is None:
            return
        clicked["pixel"] = pixel
        _save_selected_pixel(target_file, pixel)
        print(f"[reference_map_picker] Selected pixel recorded: y={pixel[0]}, x={pixel[1]} -> {target_file}")
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    if show:
        try:
            plt.show(block=True)
        except Exception:
            pass

    if "pixel" in clicked:
        return clicked["pixel"]

    if fallback_prompt:
        while True:
            raw = input("No click was received. Enter pixel as 'y x' (or 'q' to cancel): ").strip()
            if raw.lower() in {"q", "quit", "cancel"}:
                raise RuntimeError("Pixel selection cancelled by the user")
            try:
                y, x = map(int, raw.split())
            except ValueError:
                print("Invalid format. Please enter two integers: y x")
                continue
            if 0 <= y < reference_map.shape[0] and 0 <= x < reference_map.shape[1]:
                _save_selected_pixel(target_file, (y, x))
                print(f"[reference_map_picker] Manual pixel recorded: y={y}, x={x} -> {target_file}")
                return (y, x)
            print(f"Pixel out of bounds for this map: shape={reference_map.shape}, got=({y}, {x})")

    raise RuntimeError("No pixel was selected in the reference map")
