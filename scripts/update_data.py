"""Command line interface for refreshing Statiz-based pitcher datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.update_pitcher_data import update_pitcher_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update KBO pitcher datasets from Statiz")
    parser.add_argument("--start-year", type=int, required=True, help="첫 시즌 (예: 2018)")
    parser.add_argument("--end-year", type=int, required=True, help="마지막 시즌 (예: 2023)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baseball_data"),
        help="CSV 파일을 저장할 경로 (기본값: baseball_data)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_pitcher_data(args.start_year, args.end_year, args.output_dir)
    print(
        f"Saved pitcher datasets for seasons {args.start_year}–{args.end_year} to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
