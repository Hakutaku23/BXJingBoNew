from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_DEFAULT_MODEL_S3_DIR = "s3://data/t90"
_DEFAULT_CACHE_ROOT = "/tmp/idb_algos"
_ALGO_NAME = "ca_safe_band_mvp_c_line"
_RUNTIME: Dict[str, Any] = {"ready": False}
_LOCK = threading.Lock()


class RuntimeLoadError(RuntimeError):
    pass


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def _download_s3_file(remote_path: str, local_path: Path) -> None:
    try:
        import s3fs  # type: ignore
    except Exception as exc:
        raise RuntimeLoadError("s3fs is required when asset_dir is not configured") from exc
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".tmp")
    fs = s3fs.S3FileSystem()
    fs.download(remote_path, str(tmp))
    tmp.replace(local_path)


def _stage_file(filename: str, source_dir: Optional[Path], remote_dir: Optional[str], runtime_dir: Path) -> Path:
    target = runtime_dir / filename
    if source_dir is not None:
        source = source_dir / filename
        if not source.exists():
            raise RuntimeLoadError(f"missing local runtime asset: {source}")
        _atomic_copy(source, target)
    else:
        if not remote_dir:
            raise RuntimeLoadError("model_s3_dir is not configured")
        _download_s3_file(remote_dir.rstrip("/") + "/" + filename, target)
    return target


def _load_manifest_to_temp(cfg: Dict[str, Any], cache_root: Path) -> tuple[Path, Optional[Path], Optional[str]]:
    asset_dir = cfg.get("asset_dir")
    model_s3_dir = cfg.get("model_s3_dir") or _DEFAULT_MODEL_S3_DIR
    if asset_dir:
        source_dir = Path(asset_dir)
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeLoadError(f"missing manifest: {manifest_path}")
        return manifest_path, source_dir, None
    temp_dir = cache_root / _ALGO_NAME / "_manifest"
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = temp_dir / "manifest.json"
    _download_s3_file(model_s3_dir.rstrip("/") + "/manifest.json", manifest_path)
    return manifest_path, None, model_s3_dir


def _prepare_runtime_dir(cfg: Dict[str, Any]) -> tuple[Path, Dict[str, Any]]:
    cache_root = Path(cfg.get("cache_root") or _DEFAULT_CACHE_ROOT)
    manifest_path, source_dir, remote_dir = _load_manifest_to_temp(cfg, cache_root)
    manifest = _read_json(manifest_path)
    if manifest.get("algorithm_name") != _ALGO_NAME:
        raise RuntimeLoadError(f"wrong algorithm asset: {manifest.get('algorithm_name')}")
    expected_tag = manifest.get("python_tag")
    if expected_tag and expected_tag != sys.implementation.cache_tag:
        raise RuntimeLoadError(f"Python ABI mismatch: manifest={expected_tag}, runtime={sys.implementation.cache_tag}")
    artifact_hash = manifest.get("artifact_hash") or "unknown_artifact"
    version = manifest.get("version") or "unknown_version"
    runtime_dir = cache_root / _ALGO_NAME / str(version) / str(artifact_hash) / sys.implementation.cache_tag
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _atomic_copy(manifest_path, runtime_dir / "manifest.json")

    for filename in manifest.get("files", {}):
        path = runtime_dir / filename
        expected = manifest["files"][filename].get("sha256")
        if path.exists() and expected and _sha256(path) == expected:
            continue
        _stage_file(filename, source_dir, None if source_dir else remote_dir, runtime_dir)
        if expected and _sha256(path) != expected:
            try:
                path.unlink()
            except Exception:
                pass
            raise RuntimeLoadError(f"hash mismatch for runtime asset: {filename}")
    return runtime_dir, manifest


def _load_modules(runtime_dir: Path, manifest: Dict[str, Any]) -> tuple[Any, Any]:
    p = str(runtime_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    names = manifest.get("module_names") or {}
    core_name = names.get("core", "ca_safe_band_mvp_c_line_core")
    features_name = names.get("features", "ca_safe_band_mvp_c_line_features")
    core = importlib.import_module(core_name)
    features = importlib.import_module(features_name)
    return core, features


def _load_assets(runtime_dir: Path, manifest: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    asset_files = manifest.get("asset_files") or {}
    artifact = _read_json(runtime_dir / asset_files.get("artifact", "safe_band_artifact.json"))
    support = _read_json(runtime_dir / asset_files.get("support", "support.json"))
    schema = _read_json(runtime_dir / asset_files.get("schema", "schema.json"))
    strategy = artifact.get("final_strategy") or (artifact.get("aggregation_policy") or {}).get("strategy")
    if strategy != "top_rule_only":
        raise RuntimeLoadError(f"unexpected artifact strategy: {strategy}")
    return artifact, support, schema


def init(config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None, mode: str = "production") -> Dict[str, Any]:
    cfg = dict(config or {})
    if config_path:
        cfg.update(_read_json(Path(config_path)))
    with _LOCK:
        runtime_dir, manifest = _prepare_runtime_dir(cfg)
        core, features = _load_modules(runtime_dir, manifest)
        artifact, support, schema = _load_assets(runtime_dir, manifest)
        _RUNTIME.update({"ready": True, "manifest": manifest, "runtime_dir": str(runtime_dir), "core": core, "features": features, "artifact": artifact, "support": support, "schema": schema, "mode": mode})
    return {"ready": True, "runtime_dir": str(runtime_dir), "manifest": manifest}


def _ensure_ready() -> None:
    if not _RUNTIME.get("ready"):
        init()


def predict_one(row: Dict[str, Any], config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config_path is not None or config is not None or not _RUNTIME.get("ready"):
        init(config_path=config_path, config=config)
    core = _RUNTIME["core"]
    output = core.recommend_one(row, _RUNTIME["artifact"], _RUNTIME["support"], schema=_RUNTIME["schema"], mode="production")
    return core.postprocess_output_semantics(output, row)


def predict_batch(rows: Iterable[Dict[str, Any]], config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
    if config_path is not None or config is not None or not _RUNTIME.get("ready"):
        init(config_path=config_path, config=config)
    return [predict_one(dict(row)) for row in rows]


def _append_warning(existing: Any, warning: str) -> str:
    parts = []
    if isinstance(existing, str) and existing.strip():
        parts.extend([p.strip() for p in existing.split(";") if p.strip()])
    elif isinstance(existing, (list, tuple, set)):
        parts.extend([str(p).strip() for p in existing if str(p).strip()])
    if warning not in parts:
        parts.append(warning)
    return ";".join(sorted(set(parts)))


def run_once(raw_df: Optional[Any] = None, end_time: Optional[Any] = None, row: Optional[Dict[str, Any]] = None, config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None, column_mapping: Optional[Dict[str, str]] = None, min_valid_points: int = 30, include_optional_ir: bool = False, point_bounds: Any = None) -> Dict[str, Any]:
    if config_path is not None or config is not None or not _RUNTIME.get("ready"):
        init(config_path=config_path, config=config)
    if row is not None:
        return predict_one(row)
    if raw_df is None:
        raise ValueError("run_once requires row or raw_df")
    features = _RUNTIME["features"].build_runtime_features_from_dataframe(
        raw_df,
        end_time=end_time,
        time_col="time",
        column_mapping=column_mapping,
        min_valid_points=min_valid_points,
        include_optional_ir=include_optional_ir,
        point_bounds=point_bounds,
    )
    out = predict_one(features)
    out["adapter_time"] = features.get("time")
    out["adapter_feature_quality"] = features.get("feature_quality")
    out["adapter_warning_flags"] = ";".join(features.get("warning_flags") or []) if isinstance(features.get("warning_flags"), list) else features.get("warning_flags")
    out["adapter_missing_raw_columns"] = ";".join(features.get("missing_raw_columns") or []) if isinstance(features.get("missing_raw_columns"), list) else features.get("missing_raw_columns")
    out["adapter_insufficient_window_features"] = ";".join(features.get("insufficient_window_features") or []) if isinstance(features.get("insufficient_window_features"), list) else features.get("insufficient_window_features")
    if features.get("feature_quality") != "ok":
        out["input_valid"] = False
        out["recommendation_status"] = "no_recommendation_input_quality"
        out["interval_position"] = "missing"
        out["warning_flags"] = _append_warning(out.get("warning_flags"), "input_quality_incomplete")
    return out
