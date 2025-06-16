import sys
sys.path.append(r'C:\\Users\\MR99924\\workspace\\vscode\\Projects\\assetallocation-research\\data_etl')

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from macrobond import Macrobond

# Initialize Macrobond class
mb = Macrobond()

# Fetch GDP series
gdp = mb.FetchOneSeries("usgdp")

# Calculate QoQ change
gdp['qoq_change'] = gdp['usgdp'].pct_change() * 100

# Normalize GDP data
gdp['normalized_gdp'] = 2 * (gdp['usgdp'] - np.min(gdp['usgdp'])) / (np.max(gdp['usgdp']) - np.min(gdp['usgdp'])) - 1

# Fit a sine wave to the normalized GDP data
def fit_sine_wave(data):
    n = len(data)
    t = np.arange(n)
    guess_freq = 1 / (n / 4)  # Initial guess for frequency
    guess_amplitude = np.std(data) * 2**0.5
    guess_offset = np.mean(data)
    guess = np.array([guess_amplitude, 2 * np.pi * guess_freq, 0, guess_offset])

    def sine_wave(t, amplitude, frequency, phase, offset):
        return amplitude * np.sin(frequency * t + phase) + offset

    from scipy.optimize import curve_fit
    params, _ = curve_fit(sine_wave, t, data, p0=guess)
    return params

params = fit_sine_wave(gdp['normalized_gdp'])
t = np.arange(len(gdp))
fitted_sine_wave = params[0] * np.sin(params[1] * t + params[2]) + params[3]
gdp['fitted_sine_wave'] = fitted_sine_wave

# Identify the latest data point's position in the cycle
latest_position = gdp['fitted_sine_wave'].iloc[-1]

# Adjust the cycle phase thresholds to ensure more balanced bands
if latest_position > -0.15:
    cycle_phase = 'Late Cycle'
elif latest_position > -0.33:
    cycle_phase = 'Mid Cycle'
else:
    cycle_phase = 'Early Cycle'

print(f"The latest data point indicates that the economy is currently in the {cycle_phase} phase.")

# Plotting the stages using seaborn

# Plot original GDP data and normalized GDP data side by side
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# Original GDP data
sns.lineplot(x=gdp.index, y=gdp['usgdp'], ax=axs[0])
axs[0].set_title('Original GDP Data')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('GDP')

# Normalized GDP data
sns.lineplot(x=gdp.index, y=gdp['normalized_gdp'], ax=axs[1], label='Normalized GDP')
sns.lineplot(x=gdp.index, y=gdp['fitted_sine_wave'], ax=axs[1], label='Fitted Sine Wave', linestyle='--')
axs[1].set_title('Normalized GDP Data with Fitted Sine Wave')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Normalized GDP')

plt.savefig('gdp_data_comparison.png')
plt.show()

# Find peaks and troughs in QoQ change data
peaks, _ = find_peaks(gdp['qoq_change'])
troughs, _ = find_peaks(-gdp['qoq_change'])

# Plot QoQ change with peaks and troughs using seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(x=gdp.index, y=gdp['qoq_change'], label='QoQ Change')
sns.scatterplot(x=gdp.index[peaks], y=gdp['qoq_change'][peaks], color='red', label='Peaks')
sns.scatterplot(x=gdp.index[troughs], y=gdp['qoq_change'][troughs], color='blue', label='Troughs')
plt.title('Quarter-on-Quarter Change in GDP')
plt.xlabel('Date')
plt.ylabel('QoQ Change (%)')
plt.legend()
plt.savefig('qoq_change.png')
plt.show()

# Plot fitted sine wave with cycle phase indication and adjusted thresholds using seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(x=gdp.index, y=gdp['fitted_sine_wave'], label='Fitted Sine Wave')
plt.axhline(y=-0.15, color='green', linestyle='--', label='Mid/Late Cycle Threshold')
plt.axhline(y=-0.33, color='red', linestyle='--', label='Early/Mid Cycle Threshold')
sns.scatterplot(x=[gdp.index[-1]], y=[latest_position], color='orange', label=f'Latest Position: {cycle_phase}')
plt.title('Fitted Sine Wave Representation of GDP Data with Cycle Phases')
plt.xlabel('Date')
plt.ylabel('Fitted Sine Wave Position')
plt.legend()
plt.savefig('fitted_sine_wave.png')
plt.show()