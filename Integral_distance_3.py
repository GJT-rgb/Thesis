print("script start")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# 1) Function MUST come first
# -----------------------
def extract_separation_um(filepath: Path) -> float:
    """
    Extract separation from filename like 'sensor_intensity_100.csv' -> 100 (µm).
    """
    stem = filepath.stem
    parts = stem.split("_")
    try:
        return float(parts[-1])
    except ValueError:
        raise ValueError(f"Cannot extract separation from {filepath.name}")

# -----------------------
# 2) User settings
# -----------------------
data_folder = Path(r"E:\EMBL\Master EMBL\Matlab\final\ASM2DcombinednoNxattenuationhalfradius_sep_0_to_1000_step_10")

# Physical constants for true acoustic intensity
RHO = 1000.0   # kg/m^3
C0  = 1500.0   # m/s

# Integration domain:
# Use FULL arc by default (recommended). If you want to restrict, set S_MIN_MM / S_MAX_MM.
S_MIN_MM = None   # e.g., -2.5   (mm)  or None for full arc
S_MAX_MM = None   # e.g., +2.5   (mm)  or None for full arc

# If you want to keep the old x-window pressure integral for comparison (legacy)
EXPORT_LEGACY_PRMS_X_INT = True
X_MIN_MM = -1
X_MAX_MM =  1

# -----------------------
# 3) Collect files
# -----------------------
files = sorted(data_folder.glob("*.csv"), key=extract_separation_um)
print("Files found:", len(files))

# -----------------------
# 4) Main loop
# -----------------------
seps_um = []
Iint_W_per_m_list = []
Iavg_W_per_m2_list = []  # optional: arc-length averaged intensity
legacy_prms_int_list = []  # optional legacy

for filepath in files:
    print("Processing:", filepath.name)

    sep_um = extract_separation_um(filepath)
    df = pd.read_csv(filepath)

    # Required columns check
    required = {"arc_pos_mm", "intensity_rms_window_Pa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {filepath.name}")

    # Sort by arc coordinate s (arc_pos_mm)
    df = df.sort_values("arc_pos_mm")

    # Optional restriction on arc coordinate s
    if S_MIN_MM is not None and S_MAX_MM is not None:
        df_s = df[(df["arc_pos_mm"] >= S_MIN_MM) & (df["arc_pos_mm"] <= S_MAX_MM)].copy()
    else:
        df_s = df.copy()

    if len(df_s) < 2:
        print(f"  Skipping {filepath.name}: not enough data points in s-range")
        continue

    # Convert s from mm -> m
    s_m = df_s["arc_pos_mm"].to_numpy(dtype=float) * 1e-3

    # p_rms in Pa (despite the word "intensity" in the column name)
    p_rms = df_s["intensity_rms_window_Pa"].to_numpy(dtype=float)

    # True acoustic intensity I = p_rms^2 / (rho*c)  [W/m^2]
    I_Wm2 = (p_rms ** 2) / (RHO * C0)

    # Arc-length integral over s: Iint = ∫ I(s) ds  [W/m]
    Iint_W_per_m = np.trapz(I_Wm2, s_m)

    # Optional: average intensity along the integrated arc segment [W/m^2]
    arc_len_m = s_m[-1] - s_m[0]
    Iavg_W_per_m2 = Iint_W_per_m / arc_len_m if arc_len_m > 0 else np.nan

    seps_um.append(sep_um)
    Iint_W_per_m_list.append(Iint_W_per_m)
    Iavg_W_per_m2_list.append(Iavg_W_per_m2)

    # Optional legacy metric: ∫ p_rms(x) dx over x window [Pa·mm]
    if EXPORT_LEGACY_PRMS_X_INT:
        if "x_mm" not in df.columns:
            raise ValueError(f"Legacy export requested, but x_mm not found in {filepath.name}")
        df_x = df[(df["x_mm"] >= X_MIN_MM) & (df["x_mm"] <= X_MAX_MM)].sort_values("x_mm")
        if len(df_x) >= 2:
            prms_int = np.trapz(df_x["intensity_rms_window_Pa"].to_numpy(dtype=float),
                                df_x["x_mm"].to_numpy(dtype=float))
        else:
            prms_int = np.nan
        legacy_prms_int_list.append(prms_int)

# -----------------------
# 5) Results table
# -----------------------
result_df = pd.DataFrame({
    "separation_um": seps_um,
    "separation_mm": np.array(seps_um) * 1e-3,   # convenience for plotting
    "Iint_W_per_m": Iint_W_per_m_list,
    "Iavg_W_per_m2": Iavg_W_per_m2_list,
})

if EXPORT_LEGACY_PRMS_X_INT:
    result_df["integrated_prms_Pa_mm"] = legacy_prms_int_list  # old metric (optional)

result_df = result_df.sort_values("separation_um")
print(result_df)

out_path = data_folder / "integrated_true_intensity_vs_separation.csv"
result_df.to_csv(out_path, index=False)
print("Saved:", out_path)

# -----------------------
# 6) Plotting
# -----------------------
plt.figure(figsize=(7, 4))
plt.plot(result_df["separation_mm"], result_df["Iint_W_per_m"], marker="o")
plt.xlabel("Separation d (mm)")
plt.ylabel(r"$I_{\mathrm{int}}$ (W/m)  with  $I=p_{\mathrm{rms}}^2/(\rho c)$")
title = "Arc-integrated true intensity vs separation"
if S_MIN_MM is not None and S_MAX_MM is not None:
    title += f"  (s in [{S_MIN_MM},{S_MAX_MM}] mm)"
plt.title(title)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

if EXPORT_LEGACY_PRMS_X_INT:
    plt.figure(figsize=(7, 4))
    plt.plot(result_df["separation_mm"], result_df["integrated_prms_Pa_mm"], marker="o")
    plt.xlabel("Separation d (mm)")
    plt.ylabel(r"$\int p_{\mathrm{rms}}(x)\,dx$  (Pa·mm)")
    plt.title(f"Legacy metric over x in [{X_MIN_MM},{X_MAX_MM}] mm")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("script end")