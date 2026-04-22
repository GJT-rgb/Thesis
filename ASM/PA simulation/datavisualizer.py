import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ---------------------------------------
# LOAD DATA (expects columns: separation_um, integrated_value)
# ---------------------------------------
csv_path = r"E:\EMBL\Master EMBL\Python\PA simulation\integrated_intensity_vs_separation_um_-1.0_to_+1.0mm.csv"
df = pd.read_csv(csv_path)

# New column names from your latest integration
x = df["separation_um"].to_numpy()     # separation in µm
y = df["integrated_value"].to_numpy()  # integrated metric (same meaning as before)

# If you prefer to work in mm (thus FFT in cycles/mm), uncomment:
# x = x / 1000.0

# ---------------------------------------
# STEP 1 — BASELINE REMOVAL (same logic as before)
# ---------------------------------------
# Keep your Savitzky–Golay parameters; ensure the window is odd and <= len(y)
win = min(101, len(y) - (1 - len(y) % 2))  # 101 is what you used; auto-safe against short series
if win < 5:
    baseline = np.zeros_like(y)
    print("NOTE: series too short for Savitzky–Golay; skipping baseline removal.")
else:
    baseline = savgol_filter(y, window_length=win, polyorder=2)

osc = y - baseline

dx = x[1] - x[0]  # sampling step in µm (or mm if you converted above)
N = len(osc)

# ---------------------------------------
# STEP 2 — FFT (units consistent with x)
# ---------------------------------------
Y = np.fft.fft(osc)
freqs = np.fft.fftfreq(N, d=dx)  # cycles per µm (or per mm if you converted x)
amp = (2.0 / N) * np.abs(Y)

pos_mask = freqs > 0
pfreqs = freqs[pos_mask]
pamp = amp[pos_mask]

# Report top peaks (exclude DC)
if len(pfreqs):
    top_idx = np.argsort(pamp)[-5:][::-1]
    top_peaks = [(pfreqs[i], pamp[i]) for i in top_idx]
    print("Top spectral peaks (cycles/µm):")
    for fpk, apk in top_peaks:
        print(f"  {fpk:.6f}  (ampl {apk:.0f})")
else:
    top_peaks = []
    print("No positive-frequency content found (check data).")

# Dominant (for reference)
dominant_freq = top_peaks[0][0] if top_peaks else np.nan
if np.isfinite(dominant_freq):
    print(f"Dominant spatial freq: {dominant_freq:.6f} cycles/µm")
    print(f"Period: {1.0/dominant_freq:.3f} µm")

# ---------------------------------------
# STEP 3 — MANUAL BAND-PASS CROP (same structure)
# ---------------------------------------
# IMPORTANT: bounds must match the frequency units of `freqs`.
# With x in µm (default), use cycles/µm here.
# Tip: to capture your strongest line, try around 0.0085–0.0093 cycles/µm.
f_low  = 0.0085   # lower bound (cycles/µm)
f_high = 0.0093   # upper bound (cycles/µm)

print(f"\nUsing manual band-pass crop: {f_low:.6f} to {f_high:.6f} cycles/µm")

# Band-pass mask (two-sided, preserves complex symmetry)
band_mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)

# Apply band-pass to the full spectrum
Y_bp = np.zeros_like(Y, dtype=complex)
Y_bp[band_mask] = Y[band_mask]

# ---------------------------------------
# STEP 4 — IFFT RECONSTRUCTION (same)
# ---------------------------------------
osc_rec = np.fft.ifft(Y_bp).real
simulated_signal = baseline + osc_rec

# ---------------------------------------
# STEP 5 — METRICS (same)
# ---------------------------------------
def r2_score(y_true, y_hat):
    ss_res = np.sum((y_true - y_hat)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

mse_full = np.mean((y - simulated_signal)**2)
r2_full  = r2_score(y, simulated_signal)

mse_osc  = np.mean((osc - osc_rec)**2)
r2_osc   = r2_score(osc, osc_rec)

print(f"\nMSE (full signal): {mse_full:.6f}")
print(f"R²  (full signal): {r2_full:.4f}")
print(f"MSE (oscillation): {mse_osc:.6f}")
print(f"R²  (oscillation): {r2_osc:.4f}")

# ---------------------------------------
# STEP 6 — PLOTS (identical layout: 3 panels)
# ---------------------------------------
plt.figure(figsize=(14,10))

# Panel 1: Original + baseline (+ optional reconstructed)
plt.subplot(3,1,1)
plt.plot(x, y, label="Original")
plt.plot(x, baseline, label="Baseline")
# If your prior figures showed it, uncomment next line:
# plt.plot(x, simulated_signal, label="Reconstructed (band-pass IFFT)")
plt.legend()
plt.title("Original vs Baseline vs Reconstructed")
plt.xlabel(f"Separation ({'µm'})")
plt.ylabel("Integrated value")
plt.grid(True)

# Panel 2: Detrended oscillation and band-passed reconstruction
plt.subplot(3,1,2)
plt.plot(x, osc, label="Detrended")
plt.plot(x, osc_rec, label="Band-passed IFFT")
plt.legend()
plt.title("Oscillatory Component")
plt.xlabel(f"Separation ({'µm'})")
plt.ylabel("Amplitude")
# Place metrics at mid x (adjust if needed)
x_mid = x[len(x)//2]
y_span = np.max(np.abs(osc)) if np.any(osc) else 1.0
#plt.text(x_mid, 0.8*y_span, f"\MSE (full): {mse_full:.6f}\\nR² (full): {r2_full:.4f}", fontsize=9)
#plt.text(x_mid, 0.6*y_span, f"\MSE (osc):  {mse_osc:.6f}\\nR² (osc):  {r2_osc:.4f}", fontsize=9)
plt.grid(True)

# Panel 3: Spectrum (one-sided) + band-pass span
plt.subplot(3,1,3)
plt.plot(pfreqs, pamp, label="Amplitude Spectrum")
plt.axvspan(f_low, f_high, color='red', alpha=0.3, label="Band-pass")
plt.xlabel("Spatial frequency (cycles/µm)")
plt.ylabel("Amplitude")
plt.grid(False)
plt.legend()

plt.tight_layout()
plt.show()