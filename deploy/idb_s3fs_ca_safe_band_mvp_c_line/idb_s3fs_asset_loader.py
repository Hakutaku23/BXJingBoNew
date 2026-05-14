from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class AssetLoadError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise AssetLoadError("missing_asset", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise AssetLoadError("invalid_json", f"{path}: {exc}") from exc


def load_config(config_path: Optional[Any] = None) -> Dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise AssetLoadError("missing_asset", f"config not found: {path}")
    return _read_json(path)


def _candidate_dirs(config: Dict[str, Any], base_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    asset_dir = config.get("asset_dir")
    if asset_dir:
        dirs.append(Path(asset_dir))
    env_dir = os.environ.get("S3FS_ASSET_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    dirs.append(Path.cwd())
    dirs.append(base_dir)
    unique: list[Path] = []
    seen: set[str] = set()
    for item in dirs:
        key = str(item.resolve()) if item.exists() else str(item)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def resolve_asset_path(name: str, config: Dict[str, Any], base_dir: Optional[Any] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    explicit = config.get(name + "_path")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise AssetLoadError("missing_asset", f"{name}: {path}")
    filename = {"artifact": "safe_band_artifact.json", "support": "support.json", "schema": "schema.json"}[name]
    for directory in _candidate_dirs(config, base):
        path = directory / filename
        if path.exists():
            return path
    raise AssetLoadError("missing_asset", f"{filename} not found in explicit paths, S3FS_ASSET_DIR, cwd, or package dir")


def load_runtime_assets(config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None, base_dir: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    cfg = dict(config or {})
    if config_path is not None:
        cfg.update(load_config(config_path))
    artifact_path = resolve_asset_path("artifact", cfg, base_dir=base_dir)
    support_path = resolve_asset_path("support", cfg, base_dir=base_dir)
    schema_path = resolve_asset_path("schema", cfg, base_dir=base_dir)
    artifact = _read_json(artifact_path)
    support = _read_json(support_path)
    schema = _read_json(schema_path)
    if artifact.get("final_strategy") not in {None, "top_rule_only"}:
        raise AssetLoadError("wrong_artifact_version", f"unexpected strategy: {artifact.get('final_strategy')}")
    return {"artifact": artifact, "support": support, "schema": schema, "paths": {"artifact": str(artifact_path), "support": str(support_path), "schema": str(schema_path)}}
