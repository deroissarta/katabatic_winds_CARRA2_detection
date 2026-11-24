"""List years with complete CARRA2 files for the requested months and fields."""

import argparse
from pathlib import Path

from shared.paths import DEFAULT_KATABATIC_DATA_DIR
from shared.reader import resolve_available_years


def main():
    parser = argparse.ArgumentParser(
        description="List years that have complete monthly data for a given set of months."
    )
    parser.add_argument(
        "--path",
        default=str(DEFAULT_KATABATIC_DATA_DIR),
        help="Root folder containing one subfolder per year.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=["June", "July", "August"],
        help="Required month names, for example: June July August",
    )
    parser.add_argument(
        "--required-vars",
        nargs="+",
        default=["t2m", "u10", "v10"],
        help="Variables that must be present for each required month.",
    )

    args = parser.parse_args()

    years = resolve_available_years(Path(args.path), args.months, required_vars=args.required_vars)
    if not years:
        print("No complete years found for the requested months and variables.")
        return

    print("Complete years:")
    for year in years:
        print(f"- {year}")


if __name__ == "__main__":
    main()
