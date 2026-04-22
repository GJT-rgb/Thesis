import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------
# CONFIGURATION
# --------------------------
STEP_SIZE = 5.0       # <-- set your real separation distance step
DATA_FILE = "commercialchannel_test2.txt"
OUTPUT_FILE = "commercialchannel_test2_peaks.csv"
# --------------------------

script_dir = Path(__file__).resolve().parent
data_path = script_dir / DATA_FILE
output_path = script_dir / OUTPUT_FILE

# --------------------------
# LOAD EXPERIMENTAL MATRIX
# --------------------------
rows = []
with data_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().replace(",", ".")
        if not line:
            continue
        values = [float(v) for v in line.split("\t")]
        rows.append(values)

matrix = np.array(rows, dtype=float)
print("Loaded matrix shape:", matrix.shape)

# --------------------------
# COMPUTE (max - min) PER ROW
# --------------------------
peak_to_peak = np.max(matrix, axis=1) - np.min(matrix, axis=1)

# Create separation/distance axis
num_rows = matrix.shape[0]
distances = np.arange(num_rows) * STEP_SIZE

# --------------------------
# SAVE RESULT
# --------------------------
output = np.column_stack((distances, peak_to_peak))

np.savetxt(
    output_path,
    output,
    delimiter=",",
    fmt="%.6f",
    header="distance,peak_to_peak_intensity",
    comments=""
)

print("Saved peak-to-peak intensity CSV:", output_path)

# --------------------------
# VISUAL OUTPUT
# --------------------------
plt.figure(figsize=(12, 6))
plt.plot(distances, peak_to_peak, marker="o", linewidth=2)
plt.xlabel("Distance (um)")
plt.ylabel("Peak-to-peak Intensity (max-min)")
plt.title("Experimental Peak-to-Peak Intensity vs Distance")
plt.grid(True)
plt.tight_layout()
plt.show()