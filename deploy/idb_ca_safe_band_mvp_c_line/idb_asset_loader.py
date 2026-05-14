from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json_asset(local_path: str) -> Dict[str, Any]:
    path = Path(local_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_asset(local_path: str) -> List[Dict[str, str]]:
    path = Path(local_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_f3fs_downloaded_path(local_path: str) -> str:
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"External reference asset not found: {local_path}")
    return str(path)


def try_load_optional_reference_manifest(local_path: str) -> Dict[str, Any]:
    try:
        return load_json_asset(resolve_f3fs_downloaded_path(local_path))
    except Exception as exc:
        return {"available": False, "error": str(exc), "basic_scoring_available": True}
