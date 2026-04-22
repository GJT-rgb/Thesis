import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Use the folder where this script is located
script_dir = Path(__file__).resolve().parent

# Input and output paths
input_file = script_dir / "gold_test1.txt"
output_csv = script_dir / "processed_data.csv"

rows = []

# Read and parse the data
with input_file.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().replace(",", ".")
        if not line:
            continue
        values = line.split("\t")
        rows.append([float(v) for v in values])

# Convert to matrix
matrix = np.array(rows, dtype=float)

# Save CSV
np.savetxt(output_csv, matrix, delimiter=",", fmt="%.6f")
print(f"CSV saved as: {output_csv}")


# Plot each line
plt.figure(figsize=(12,6))
for i, row in enumerate(matrix):
    plt.plot(row, label=f"Row {i}")

plt.title("Intensity Profiles by Row")
plt.xlabel("X Position (column index)")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot heatmap
plt.imshow(matrix, aspect="auto", cmap="turbo", origin="lower")
plt.colorbar(label="Intensity")
plt.title("Intensity Heatmap")
plt.xlabel("Column index")
plt.ylabel("Row index")
plt.show()