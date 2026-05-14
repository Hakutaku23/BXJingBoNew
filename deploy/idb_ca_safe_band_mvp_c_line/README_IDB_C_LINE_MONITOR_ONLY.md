# C-line calcium safe-band MVP IDB package

Purpose: monitor-only guidance testing for the C-line calcium-consumption safe-band MVP.

Entry file: `ca_safe_band_mvp_c_line.py`. The release zip name is `ca_safe_band_mvp_c_line.zip`.

This package is parquet-free at runtime. Critical scoring assets are embedded in `runtime_assets_embedded.py` as Python constants, so basic scoring does not require reading JSON, CSV, or parquet from inside the uploaded zip.

Required raw input points:
- B4-FIC-C51001.PV.F_CV / rubber_flow_2
- B4-FIC-C51004.PV.CV / bromine_feed
- B4-AT-C50002A-BIIR.PV.CV / tank_rubber_conc
- B4-TI-C51007A_S.PV.CV / r510a_temp
- B4-TI-C51101A_S.PV.CV / r511a_temp
- B4-TI-C51702A.PV.F_CV / r512a_temp
- B4-FIC-C51401.PV.F_CV / ca_feed
- B4-FIC-C51801.PV.F_CV / esbo_feed
- B4-FIC-C51605.PV.F_CV / neutral_alkali_feed
- B4-TI-C51301_S.PV.CV / r513_temp
- B4-TI-C51401_S.PV.CV / r514_temp

Main output fields include recommendation status, current calcium consumption, recommended calcium-consumption min/max/target, interval position, action visibility, review flag, explanation, and warning flags.

Safety mode:
- Advisory display/logging only.
- No DCS setpoint output is produced.
- No calcium setpoint command is produced.
- Plant staff retain all operating authority.

Optional external reference assets can be prepared in f3fs for dashboards and drift monitoring. They are not required for basic scoring.

Local smoke test:
```bash
python -B ca_safe_band_mvp_c_line.py
```
