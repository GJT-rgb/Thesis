import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ---------------------------------------
# LOAD DATA
# ---------------------------------------
df = pd.read_csv(r"E:\EMBL\Master EMBL\Python\PA simulation\integrated_intensity_vs_separation_um_-1.0_to_+1.0mm.csv")

x = df["separation_um"].values
y = df["integrated_value"].values

# ---------------------------------------
# STEP 1 — BASELINE REMOVAL
# ---------------------------------------
baseline = savgol_filter(y, window_length=101, polyorder=2)
osc = y - baseline

dx = x[1] - x[0]
N = len(osc)
Fs = 1.0 / dx  # samples per mm

# ---------------------------------------
# STEP 2 — FFT
# ---------------------------------------
Y = np.fft.fft(osc)
freqs = np.fft.fftfreq(N, d=dx)
amp = (2.0 / N) * np.abs(Y)

pos_mask = freqs > 0
pfreqs = freqs[pos_mask]
pamp = amp[pos_mask]

# Determine dominant frequency
dominant_freq = pfreqs[np.argmax(pamp)]
print(f"Dominant spatial frequency: {dominant_freq:.6f} cycles/mm")
print(f"Corresponding spatial period: {1/dominant_freq:.3f} mm")

# ---------------------------------------
# STEP 3 — MANUAL BAND-PASS CROP
# ---------------------------------------
# Set your own frequency crop range (in cycles/mm):
# Example: crop between 0.010 and 0.020 c/mm
f_low = 0.011     # <<< choose your lower bound
f_high = 0.0122    # <<< choose your upper bound

print(f"\nUsing manual band-pass crop: {f_low:.6f} to {f_high:.6f} cycles/mm")

# Create band-pass mask
band_mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)

# Apply band-pass to the full complex spectrum
Y_bp = np.zeros_like(Y, dtype=complex)
Y_bp[band_mask] = Y[band_mask]

# ---------------------------------------
# STEP 4 — IFFT RECONSTRUCTION
# ---------------------------------------
osc_rec = np.fft.ifft(Y_bp).real
simulated_signal = baseline + osc_rec

# ---------------------------------------
# STEP 5 — METRICS
# ---------------------------------------
def r2_score(y, y_hat):
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

mse_full = np.mean((y - simulated_signal)**2)
r2_full = r2_score(y, simulated_signal)

mse_osc = np.mean((osc - osc_rec)**2)
r2_osc = r2_score(osc, osc_rec)

print(f"\nMSE (full signal): {mse_full:.6f}")
print(f"R²  (full signal): {r2_full:.4f}")
print(f"MSE (oscillation): {mse_osc:.6f}")
print(f"R²  (oscillation): {r2_osc:.4f}")

# ---------------------------------------
# STEP 6 — PLOTS
# ---------------------------------------
plt.figure(figsize=(14,10))

# Raw + baseline + reconstruction
plt.subplot(3,1,1)
plt.plot(x, y, label="Original")
plt.plot(x, baseline, label="Baseline")
#plt.plot(x, simulated_signal, label="Reconstructed (band-pass IFFT)")
plt.legend()
plt.title("Original vs Baseline vs Reconstructed")

# Detrended oscillation vs reconstructed oscillation
plt.subplot(3,1,2)
plt.plot(x, osc, label="Detrended")
plt.plot(x, osc_rec, label="Band-passed IFFT")
plt.legend()
plt.text(500,20, f"\nMSE (full signal): {mse_full:.6f}" f"R²  (full signal): {r2_full:.4f}")
plt.text(500,15, f"MSE (oscillation): {mse_osc:.6f}" f"R²  (oscillation): {r2_osc:.4f}")
plt.title("Oscillatory Component")

# Amplitude spectrum + bandpass region
plt.subplot(3,1,3)
plt.plot(pfreqs, pamp, label="Amplitude Spectrum")
plt.axvspan(f_low, f_high, color='red', alpha=0.3, label="Band-pass")
plt.xlabel("Spatial frequency (cycles/mm)")
plt.ylabel("Amplitude")
plt.grid(False)
plt.legend()
plt.text(0.04,12, f"highpass: {f_low:.4f}" f" lowpass: {f_high:.4f}")
plt.tight_layout()
plt.show()