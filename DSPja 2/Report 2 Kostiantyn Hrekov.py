import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# PARAMETERS
fs = 1000        # sampling frequency
N = 4096         # number of samples

# =========================
# GENERATE RANDOM SIGNALS
# =========================
np.random.seed(0)

signal1 = np.random.normal(0, 1, N)   # lower power signal
signal2 = np.random.normal(0, 3, N)   # higher power signal

# ENERGY AND STD
energy1 = np.sum(signal1**2)
energy2 = np.sum(signal2**2)

std1 = np.std(signal1)
std2 = np.std(signal2)

print("----- SIGNAL 1 -----")
print("Energy:", energy1)
print("Standard deviation:", std1)

print("\n----- SIGNAL 2 -----")
print("Energy:", energy2)
print("Standard deviation:", std2)

# PLOT SIGNALS
plt.figure(figsize=(10,4))
plt.plot(signal1, label="Signal 1 (low power)")
plt.plot(signal2, label="Signal 2 (high power)", alpha=0.7)
plt.title("Random Signals Comparison")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# PSD - WELCH (HANN WINDOW)
f1_hann, Pxx1_hann = welch(signal1, fs=fs, window='hann', nperseg=512)
f2_hann, Pxx2_hann = welch(signal2, fs=fs, window='hann', nperseg=512)

plt.figure()
plt.semilogy(f1_hann, Pxx1_hann, label="Signal1 Hann")
plt.semilogy(f2_hann, Pxx2_hann, label="Signal2 Hann")
plt.title("PSD using Welch (Hann window)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency")
plt.legend()
plt.grid(True)
plt.show()

# PSD - WELCH (HAMMING WINDOW)
f1_hamming, Pxx1_hamming = welch(signal1, fs=fs, window='hamming', nperseg=512)
f2_hamming, Pxx2_hamming = welch(signal2, fs=fs, window='hamming', nperseg=512)

plt.figure()
plt.semilogy(f1_hamming, Pxx1_hamming, label="Signal1 Hamming")
plt.semilogy(f2_hamming, Pxx2_hamming, label="Signal2 Hamming")
plt.title("PSD using Welch (Hamming window)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency")
plt.legend()
plt.grid(True)
plt.show()