print("script start")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# 1️⃣ Function MUST come first
# -----------------------
def extract_distance(filepath):
    stem = filepath.stem  # e.g. 'sensor_intensity_100'
    parts = stem.split("_")
    try:
        return float(parts[-1])
    except ValueError:
        raise ValueError(f"Cannot extract distance from {filepath.name}")

# -----------------------
# 2️⃣ User settings
# -----------------------
data_folder = Path(r"E:\EMBL\Master EMBL\Matlab\final\ASM2DcombinednoNx_sep_0_to_2000_step_5")

x_min = -1
x_max = 1

# -----------------------
# 3️⃣ Now you can use extract_distance safely
# -----------------------
files = sorted(data_folder.glob("*.csv"), key=extract_distance)

print("Files found:", len(files))

# -----------------------
# 4️⃣ Main loop
# -----------------------
distances = []
integrals = []

for filepath in files:
    print("Processing:", filepath.name)

    distance = extract_distance(filepath)
    df = pd.read_csv(filepath)

    df_range = df[
        (df["x_mm"] >= x_min) &
        (df["x_mm"] <= x_max)
    ].sort_values("x_mm")

    if len(df_range) < 2:
        print(f"  Skipping {filepath.name}: no data in x-range")
        continue

    integral = np.trapz(
        df_range["intensity_rms_window_Pa"],
        df_range["x_mm"]
    )

    distances.append(distance)
    integrals.append(integral)

# -----------------------
# 5️⃣ Results
# -----------------------
result_df = pd.DataFrame({
    "distance": distances,
    "integrated_intensity_Pa_mm": integrals
}).sort_values("distance")

print(result_df)

result_df.to_csv("integrated_intensity_vs_distance.csv", index=False)

# -----------------------
# 6️⃣ Plotting
# -----------------------
plt.figure(figsize=(6, 4))
plt.plot(
    result_df["distance"],
    result_df["integrated_intensity_Pa_mm"],
    marker="o"
)
plt.xlabel("Distance (mm)")
plt.ylabel("Integrated intensity (Pa·mm)")
plt.title(f"∫ I(x) dx from {x_min} mm to {x_max} mm")
plt.grid(True)
plt.tight_layout()
plt.show()

print("script end")