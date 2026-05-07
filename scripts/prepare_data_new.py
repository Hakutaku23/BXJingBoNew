from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SUPPORTED_DATA_EXTENSIONS = {".csv", ".xlsx", ".txt"}


@dataclass(frozen=True)
class PointSpec:
    variable: str
    tag: str
    status: str
    lower: float | None
    upper: float | None
    raw_left_bound: float | None
    raw_right_bound: float | None


def normalize_tag(value: object) -> str:
    text = str(value).strip()
    text = text.replace("/", ".").replace("-", "_").replace(".", "_")
    text = re.sub(r"_+", "_", text)
    text = text.upper().strip("_")
    if text.startswith("B4_"):
        text = text[3:]
    return text


def find_point_file(data_dir: Path) -> Path:
    candidates = [
        path
        for path in data_dir.glob("*.xlsx")
        if not path.name.startswith("~$")
        and "\u526f\u672c" in path.name
        and "\u6570\u636e\u70b9\u4f4d" in path.name
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one point file, found {len(candidates)}: {candidates}")
    return candidates[0]


def as_optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_specs(point_file: Path) -> list[PointSpec]:
    sheet = pd.read_excel(point_file, sheet_name="Sheet3")
    variable_col, tag_col, status_col, left_bound_col, right_bound_col = sheet.columns[:5]

    specs: list[PointSpec] = []
    for row in sheet.itertuples(index=False):
        variable = str(getattr(row, variable_col)).strip()
        tag = str(getattr(row, tag_col)).strip()
        status = str(getattr(row, status_col)).strip()
        left_bound = as_optional_float(getattr(row, left_bound_col))
        right_bound = as_optional_float(getattr(row, right_bound_col))

        if left_bound is None or right_bound is None:
            lower = None
            upper = None
        else:
            lower = min(left_bound, right_bound)
            upper = max(left_bound, right_bound)

        specs.append(
            PointSpec(
                variable=variable,
                tag=tag,
                status=status,
                lower=lower,
                upper=upper,
                raw_left_bound=left_bound,
                raw_right_bound=right_bound,
            )
        )
    return specs


def read_csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)


def discover_txt_matches(source_dir: Path, specs: list[PointSpec]) -> dict[str, list[Path]]:
    needed = {normalize_tag(spec.tag) for spec in specs}
    matches: dict[str, list[Path]] = {key: [] for key in needed}

    for path in sorted(source_dir.rglob("*.txt")):
        if path.name.startswith("~$"):
            continue
        normalized = normalize_tag(path.stem)
        if normalized in needed:
            matches[normalized].append(path)
    return matches


def discover_wide_csv_matches(source_dir: Path, specs: list[PointSpec]) -> dict[str, list[tuple[Path, str]]]:
    needed = {normalize_tag(spec.tag) for spec in specs}
    matches: dict[str, list[tuple[Path, str]]] = {key: [] for key in needed}

    for path in sorted(source_dir.rglob("*.csv")):
        if path.name.startswith("~$"):
            continue
        for column in read_csv_header(path):
            normalized = normalize_tag(column)
            if normalized in needed:
                matches[normalized].append((path, column))
    return matches


def discover_wide_xlsx_matches(source_dir: Path, specs: list[PointSpec], point_file: Path) -> dict[str, list[tuple[Path, str, str]]]:
    needed = {normalize_tag(spec.tag) for spec in specs}
    matches: dict[str, list[tuple[Path, str, str]]] = {key: [] for key in needed}

    for path in sorted(source_dir.rglob("*.xlsx")):
        if path.name.startswith("~$") or path.resolve() == point_file.resolve():
            continue
        workbook = pd.ExcelFile(path)
        for sheet_name in workbook.sheet_names:
            header = pd.read_excel(path, sheet_name=sheet_name, nrows=0)
            for column in header.columns:
                normalized = normalize_tag(column)
                if normalized in needed:
                    matches[normalized].append((path, sheet_name, column))
    return matches


def discover_long_xlsx_matches(source_dir: Path, specs: list[PointSpec], point_file: Path) -> dict[str, list[tuple[Path, str]]]:
    needed = {normalize_tag(spec.tag) for spec in specs}
    matches: dict[str, list[tuple[Path, str]]] = {key: [] for key in needed}

    for path in sorted(source_dir.rglob("*.xlsx")):
        if path.name.startswith("~$") or path.resolve() == point_file.resolve():
            continue
        workbook = pd.ExcelFile(path)
        for sheet_name in workbook.sheet_names:
            sample = pd.read_excel(path, sheet_name=sheet_name, nrows=5000)
            if sample.shape[1] < 3:
                continue
            first_column = sample.iloc[:, 0].map(normalize_tag)
            present = set(first_column.dropna()) & needed
            for normalized in present:
                matches[normalized].append((path, sheet_name))
    return matches


def pick_source(
    normalized: str,
    txt_matches: dict[str, list[Path]],
    csv_matches: dict[str, list[tuple[Path, str]]],
    xlsx_wide_matches: dict[str, list[tuple[Path, str, str]]],
    xlsx_long_matches: dict[str, list[tuple[Path, str]]],
) -> tuple[str, tuple] | None:
    if txt_matches.get(normalized):
        return "txt_long", (txt_matches[normalized][0],)
    if csv_matches.get(normalized):
        return "csv_wide", csv_matches[normalized][0]
    if xlsx_wide_matches.get(normalized):
        return "xlsx_wide", xlsx_wide_matches[normalized][0]
    if xlsx_long_matches.get(normalized):
        return "xlsx_long", xlsx_long_matches[normalized][0]
    return None


def read_txt_sources(sources: Iterable[tuple[Path, PointSpec]], chunksize: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path, spec in sources:
        chunk_frames: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["tag", "time", "value", "quality"],
            usecols=[0, 1, 2],
            chunksize=chunksize,
            encoding="utf-8",
        ):
            chunk = chunk[chunk["tag"].map(normalize_tag) == normalize_tag(spec.tag)].copy()
            chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
            chunk[spec.variable] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk = chunk[["time", spec.variable]].dropna(subset=["time"])
            chunk_frames.append(chunk)

        if chunk_frames:
            frame = pd.concat(chunk_frames, ignore_index=True)
            frame = frame.groupby("time", as_index=False).last()
            frames.append(frame)

    return merge_on_time(frames)


def read_csv_sources(sources: Iterable[tuple[Path, str, str]], chunksize: int) -> pd.DataFrame:
    by_path: dict[Path, list[tuple[str, str]]] = {}
    for path, source_col, output_col in sources:
        by_path.setdefault(path, []).append((source_col, output_col))

    frames: list[pd.DataFrame] = []
    for path, columns in by_path.items():
        usecols = ["time"] + [source_col for source_col, _ in columns]
        rename_map = {source_col: output_col for source_col, output_col in columns}

        chunk_frames: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, encoding="utf-8-sig"):
            chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
            chunk = chunk.dropna(subset=["time"])
            for source_col, _ in columns:
                chunk[source_col] = pd.to_numeric(chunk[source_col], errors="coerce")
            chunk = chunk.rename(columns=rename_map)
            chunk_frames.append(chunk)

        if chunk_frames:
            frame = pd.concat(chunk_frames, ignore_index=True)
            frame = frame.groupby("time", as_index=False).last()
            frames.append(frame)

    return merge_on_time(frames)


def read_xlsx_wide_source(path: Path, sheet_name: str, source_col: str, output_col: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet_name, usecols=lambda col: col in {"time", source_col})
    if "time" not in frame.columns:
        return pd.DataFrame(columns=["time", output_col])
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame[output_col] = pd.to_numeric(frame[source_col], errors="coerce")
    return frame[["time", output_col]].dropna(subset=["time"])


def read_xlsx_long_source(path: Path, sheet_name: str, spec: PointSpec) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet_name)
    if frame.shape[1] < 3:
        return pd.DataFrame(columns=["time", spec.variable])

    tag_col = frame.columns[0]
    time_col = frame.columns[1]
    value_col = frame.columns[2]
    normalized_tag = normalize_tag(spec.tag)

    filtered = frame[frame[tag_col].map(normalize_tag) == normalized_tag].copy()
    filtered["time"] = pd.to_datetime(filtered[time_col], errors="coerce")
    filtered[spec.variable] = pd.to_numeric(filtered[value_col], errors="coerce")
    return filtered[["time", spec.variable]].dropna(subset=["time"])


def merge_on_time(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["time"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="time", how="outer")
    return merged


def apply_limits(frame: pd.DataFrame, specs: list[PointSpec]) -> dict[str, int]:
    cleaned_counts: dict[str, int] = {}
    for spec in specs:
        if spec.variable not in frame.columns:
            continue
        if spec.lower is None or spec.upper is None:
            cleaned_counts[spec.variable] = 0
            continue

        values = pd.to_numeric(frame[spec.variable], errors="coerce")
        invalid = values.notna() & ((values < spec.lower) | (values > spec.upper))
        cleaned_counts[spec.variable] = int(invalid.sum())
        frame.loc[invalid, spec.variable] = np.nan
    return cleaned_counts


def missing_metrics(frame: pd.DataFrame, specs: list[PointSpec]) -> dict[str, dict[str, float | int]]:
    row_count = int(len(frame))
    metrics: dict[str, dict[str, float | int]] = {}
    for spec in specs:
        if spec.variable not in frame.columns:
            missing_count = row_count
        else:
            missing_count = int(frame[spec.variable].isna().sum())
        metrics[spec.variable] = {
            "missing_count": missing_count,
            "missing_rate": float(missing_count / row_count) if row_count else 0.0,
        }
    return metrics


def filter_rows(frame: pd.DataFrame, drop_missing: bool, start_date: str | None) -> tuple[pd.DataFrame, dict[str, object]]:
    before_rows = int(len(frame))
    filtered = frame

    rows_removed_by_missing = 0
    if drop_missing:
        before_dropna = int(len(filtered))
        filtered = filtered.dropna(axis=0, how="any").copy()
        rows_removed_by_missing = before_dropna - int(len(filtered))

    rows_removed_by_start_date = 0
    parsed_start_date = None
    if start_date:
        parsed_start_date = pd.to_datetime(start_date)
        before_date_filter = int(len(filtered))
        filtered = filtered[filtered["time"] >= parsed_start_date].copy()
        rows_removed_by_start_date = before_date_filter - int(len(filtered))

    filtered = filtered.sort_values("time").reset_index(drop=True)
    return filtered, {
        "before_rows": before_rows,
        "after_rows": int(len(filtered)),
        "drop_missing": drop_missing,
        "rows_removed_by_missing": rows_removed_by_missing,
        "start_date": parsed_start_date.isoformat() if parsed_start_date is not None else None,
        "rows_removed_by_start_date": rows_removed_by_start_date,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare halogen section data_new.parquet from Sheet3 point specs.")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--source-dir", default=None, type=Path)
    parser.add_argument("--raw-output", default=Path("data/data_tot.parquet"), type=Path)
    parser.add_argument("--output", default=Path("data/data_new.parquet"), type=Path)
    parser.add_argument("--report", default=Path("data/data_new_report.json"), type=Path)
    parser.add_argument("--chunksize", default=200_000, type=int)
    parser.add_argument("--drop-missing", action="store_true", help="Drop rows with any missing value after limit cleaning.")
    parser.add_argument("--start-date", default=None, help="Keep rows whose time is greater than or equal to this date.")
    args = parser.parse_args()

    data_dir = args.data_dir
    source_dir = args.source_dir or data_dir
    point_file = find_point_file(data_dir)
    specs = load_specs(point_file)

    txt_matches = discover_txt_matches(source_dir, specs)
    csv_matches = discover_wide_csv_matches(source_dir, specs)
    xlsx_wide_matches = discover_wide_xlsx_matches(source_dir, specs, point_file)
    xlsx_long_matches = discover_long_xlsx_matches(source_dir, specs, point_file)

    selected_sources: dict[str, dict[str, object]] = {}
    missing_specs: list[PointSpec] = []
    txt_sources: list[tuple[Path, PointSpec]] = []
    csv_sources: list[tuple[Path, str, str]] = []
    xlsx_frames: list[pd.DataFrame] = []

    for spec in specs:
        normalized = normalize_tag(spec.tag)
        source = pick_source(normalized, txt_matches, csv_matches, xlsx_wide_matches, xlsx_long_matches)
        if source is None:
            missing_specs.append(spec)
            selected_sources[spec.variable] = {"tag": spec.tag, "source_type": None, "source": None}
            continue

        source_type, payload = source
        if source_type == "txt_long":
            (path,) = payload
            txt_sources.append((path, spec))
            selected_sources[spec.variable] = {
                "tag": spec.tag,
                "source_type": source_type,
                "source": str(path),
            }
        elif source_type == "csv_wide":
            path, source_col = payload
            csv_sources.append((path, source_col, spec.variable))
            selected_sources[spec.variable] = {
                "tag": spec.tag,
                "source_type": source_type,
                "source": str(path),
                "source_column": source_col,
            }
        elif source_type == "xlsx_wide":
            path, sheet_name, source_col = payload
            xlsx_frames.append(read_xlsx_wide_source(path, sheet_name, source_col, spec.variable))
            selected_sources[spec.variable] = {
                "tag": spec.tag,
                "source_type": source_type,
                "source": str(path),
                "sheet": sheet_name,
                "source_column": source_col,
            }
        else:
            path, sheet_name = payload
            xlsx_frames.append(read_xlsx_long_source(path, sheet_name, spec))
            selected_sources[spec.variable] = {
                "tag": spec.tag,
                "source_type": source_type,
                "source": str(path),
                "sheet": sheet_name,
            }

    frames = []
    txt_frame = read_txt_sources(txt_sources, chunksize=args.chunksize)
    if not txt_frame.empty:
        frames.append(txt_frame)
    csv_frame = read_csv_sources(csv_sources, chunksize=args.chunksize)
    if not csv_frame.empty:
        frames.append(csv_frame)
    frames.extend(frame for frame in xlsx_frames if not frame.empty)

    result = merge_on_time(frames)
    for spec in specs:
        if spec.variable not in result.columns:
            result[spec.variable] = np.nan

    output_columns = ["time"] + [spec.variable for spec in specs]
    result = result[output_columns].sort_values("time").reset_index(drop=True)
    raw_result = result.copy()
    missing_before_limits = missing_metrics(raw_result, specs)

    cleaned_result = result.copy()
    cleaned_counts = apply_limits(cleaned_result, specs)
    missing_after_limits = missing_metrics(cleaned_result, specs)
    final_result, row_filter_report = filter_rows(cleaned_result, args.drop_missing, args.start_date)

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_result.to_parquet(args.raw_output, index=False)
    final_result.to_parquet(args.output, index=False)

    report = {
        "point_file": str(point_file),
        "source_dir": str(source_dir),
        "raw_output": str(args.raw_output),
        "output": str(args.output),
        "rows": int(len(final_result)),
        "rows_after_limit_cleaning_before_row_filters": int(len(cleaned_result)),
        "row_filters": row_filter_report,
        "columns": output_columns,
        "missing_points": [
            {"variable": spec.variable, "tag": spec.tag, "status": spec.status} for spec in missing_specs
        ],
        "selected_sources": selected_sources,
        "limits": {
            spec.variable: {
                "tag": spec.tag,
                "raw_left_bound": spec.raw_left_bound,
                "raw_right_bound": spec.raw_right_bound,
                "interpreted_lower": spec.lower,
                "interpreted_upper": spec.upper,
                "out_of_range_to_nan": cleaned_counts.get(spec.variable, 0),
                "missing_before_limits_count": missing_before_limits[spec.variable]["missing_count"],
                "missing_before_limits_rate": missing_before_limits[spec.variable]["missing_rate"],
                "missing_after_limits_count": missing_after_limits[spec.variable]["missing_count"],
                "missing_after_limits_rate": missing_after_limits[spec.variable]["missing_rate"],
            }
            for spec in specs
        },
        "missing_rates": {
            spec.variable: {
                "before_limits": missing_before_limits[spec.variable],
                "after_limits": missing_after_limits[spec.variable],
            }
            for spec in specs
        },
        "raw_non_null_counts": {column: int(raw_result[column].notna().sum()) for column in output_columns},
        "non_null_counts": {column: int(cleaned_result[column].notna().sum()) for column in output_columns},
        "final_non_null_counts": {column: int(final_result[column].notna().sum()) for column in output_columns},
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
