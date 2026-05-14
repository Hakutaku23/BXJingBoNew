# C-line safe-band MVP IDB/s3fs package

This package is for C-line monitor-only guidance testing. The old merged-line package is invalid for this C-line deployment evidence.

Runtime assets are JSON files: `safe_band_artifact.json`, `support.json`, and `schema.json`. They can be loaded from explicit local paths in `idb_config_template.json`, from `S3FS_ASSET_DIR`, from the current working directory, or from the package directory for local smoke tests. The runtime does not require parquet and does not import parquet engines.

Entry file: `ca_safe_band_mvp_c_line.py`. Release zip: `ca_safe_band_mvp_c_line.zip`.

The user-facing recommendation target is calcium stearate feed (`硬脂酸钙加注量`). The normalized calcium-consumption fields (`ca_consumption`, `ca_per_rubber_flow`) are internal diagnostic metrics used to compute and compare safe bands. Before output, normalized recommendation limits are de-normalized with `rubber_flow_2_win_60_mean`, then protected by the point-config bounds for `B4-FIC-C51401.PV.F_CV`: `700 <= 硬脂酸钙加注量 <= 1300`. Unbounded feed fields are diagnostic only. T90 outputs are risk warnings only, not exact current or future T90 value predictions.

Required raw input points are the 11 C-line DCS tags used by the C-line safe-band package. Outputs include recommendation status, current calcium stearate feed, bounded recommended calcium stearate feed interval, interval position, T90 high/low risk warning, action visibility, engineering review flag, and warning flags.

Safety: advisory display/logging only. No automatic calcium adjustment. No DCS setpoint writeback. Plant operators and process engineers retain all control authority.
