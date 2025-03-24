import numpy as np
import matplotlib.pyplot as plt

# Parameters
w = 60 * 2 * np.pi / 60  # Angular velocity (radians per second)
L = 1.4 # Radius (meters)
R = (12*25.4/1000)/2 # Length (meters)
r=171.45/1000/2
A = np.pi * ((2 * r)**2) / 4  # Area (square meters)

NB = 10000

# Theta values
teta = np.linspace(0, 20 * np.pi, NB)
teta1 = teta + 2 * np.pi / 3
teta2 = teta + 4 * np.pi / 3

# Flow rate calculations
Q = -2 * w * R * A * 3600 * (np.sin(teta) + (R / (2 * L)))
Q1 = -2 * w * R * A * 3600 * (np.sin(teta1) + (R / (2 * L)))
Q2 = -2 * w * R * A * 3600 * (np.sin(teta2) + (R / (2 * L)))

# Find element-wise maximum
Qn = np.maximum.reduce([Q, Q1, Q2])*2

# Calculate mean of Qn
Qm =np.mean(Qn)

SR = NB
T = (20 * np.pi) / SR

# Compute the FFT 
signal_fft = np.fft.fft(Qn) 
fft_freq = np.fft.fftfreq(len(Qn), T) 

# Only keep the positive half of the spectrum
positive_freq_indices = np.where(fft_freq >= 0) 
fft_freq = fft_freq[positive_freq_indices] 
signal_fft = signal_fft[positive_freq_indices] 

# Plot the signal 
plt.figure() 
plt.subplot(2, 1, 1) 
plt.plot(teta, Qn)
plt.axhline(y=Qm, color='r', linestyle='--', label=f'Mean Q = {Qm:.2f}')
plt.xlabel('Theta (radians)') 
plt.ylabel('Amplitude') 
plt.title('Original Signal') 
plt.grid(True)
plt.legend()

# Plot the FFT result 
plt.subplot(2, 1, 2) 
plt.plot(fft_freq, np.abs(signal_fft)) 
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Magnitude') 
plt.title('FFT of the Signal')
plt.grid(True)

plt.tight_layout()
plt.show()




