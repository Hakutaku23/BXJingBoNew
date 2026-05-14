from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .interface import SafeBandRecommender
except Exception:
    from interface import SafeBandRecommender  # type: ignore


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor-only calcium safe-band MVP runtime example.")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=["production", "replay"], default="production")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--input-parquet", type=Path)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--raw-input-csv", type=Path)
    parser.add_argument("--raw-input-parquet", type=Path)
    parser.add_argument("--raw-time-col", default="time")
    parser.add_argument("--end-time")
    parser.add_argument("--min-valid-points", type=int, default=30)
    parser.add_argument("--include-optional-ir", default="true")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-parquet", type=Path)
    return parser.parse_args()


def read_json_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("rows")
        return rows if isinstance(rows, list) else [payload]
    raise ValueError("JSON input must be an object, list of objects, or {'rows': [...]} structure.")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_engineered_input(args: argparse.Namespace) -> Any:
    if args.input_json:
        return read_json_rows(args.input_json)
    if args.input_csv:
        return read_csv_rows(args.input_csv)
    if args.input_parquet:
        import pandas as pd
        return pd.read_parquet(args.input_parquet)
    raise ValueError("Provide engineered input or raw input.")


def read_raw_input(args: argparse.Namespace) -> Any:
    import pandas as pd
    if args.raw_input_csv:
        return pd.read_csv(args.raw_input_csv)
    if args.raw_input_parquet:
        return pd.read_parquet(args.raw_input_parquet)
    raise ValueError("Provide --raw-input-csv or --raw-input-parquet.")


def write_outputs(result: Any, args: argparse.Namespace) -> None:
    rows = result.to_dict(orient="records") if hasattr(result, "to_dict") else result
    if isinstance(rows, dict):
        rows = [rows]
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if args.output_parquet:
        import pandas as pd
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        frame = result if hasattr(result, "to_parquet") else pd.DataFrame(rows)
        frame.to_parquet(args.output_parquet, index=False)


def main() -> None:
    # Plant DCS fetch logic should be implemented by the plant adapter owner.
    # Plant writeback logic should be implemented by the plant adapter owner.
    # Current script does not write DCS and does not perform automatic control.
    args = parse_args()
    recommender = SafeBandRecommender(args.model_dir, mode=args.mode).load()
    if args.raw_input_csv or args.raw_input_parquet:
        raw_df = read_raw_input(args)
        if args.end_time:
            result = recommender.predict_from_raw_dataframe(
                raw_df,
                end_time=args.end_time,
                time_col=args.raw_time_col,
                min_valid_points=args.min_valid_points,
                include_optional_ir=parse_bool(args.include_optional_ir),
            )
        else:
            result = recommender.predict_batch_from_raw_dataframe(
                raw_df,
                evaluation_times=None,
                time_col=args.raw_time_col,
                min_valid_points=args.min_valid_points,
                include_optional_ir=parse_bool(args.include_optional_ir),
            )
    else:
        input_data = read_engineered_input(args)
        result = recommender.predict_batch(input_data, mode=args.mode)
    write_outputs(result, args)
    count = len(result) if hasattr(result, "__len__") else 0
    print("Scored rows: {}".format(count))
    print("Mode: {}; monitor-only; no DCS writeback; no automatic control.".format(args.mode))


if __name__ == "__main__":
    main()
