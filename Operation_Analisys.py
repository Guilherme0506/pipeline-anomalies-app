import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# === 1. Load Excel File ===
file_path = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Dados_Operacionais_03-25.xlsx"  # ✅ Update with actual filename

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip().str.upper()  # Normalize column names
df['DATA'] = pd.to_datetime(df['DATA'])  # Ensure datetime format

# Define required columns
required_cols = ['DATA', 'PEB', 'QEB', 'DEB', 'P45']
optional_cols = ['PT', 'QT', 'DT']

# Convert to numeric, handling errors
for col in required_cols[1:] + [c for c in optional_cols if c in df.columns]:  # Skip 'DATA'
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === 2. Calculate Weight Concentration ===
rho_s = 3.2  # solids density [kg/m³]
rho_l = 1.0  # water density [kg/m³]
df['WEIGHT_CONCENTRATION'] = (rho_s * (df['DEB'] - rho_l)) / (df['DEB'] * (rho_s - rho_l))
df['PRODUCTION_TPH'] = df['QEB'].abs() * df['DEB'] * (df['WEIGHT_CONCENTRATION'] / 100) / 1000

# === 4. Plot Raw Variables Before Filtering ===
fig, axs = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
axs = axs.flatten()

variables = ['PEB', 'QEB', 'DEB', 'P45', 'PT', 'QT', 'DT', 'WEIGHT_CONCENTRATION']
titles = [
    'Pressure @ Pump Station (PEB)', 'Flow Rate @ Pump Station (QEB)', 'Density @ Pump Station (DEB)',
    'Pressure @ Km 55 (P45)', 'Pressure @ Terminal (PT)', 'Flow Rate @ Terminal (QT)',
    'Density @ Terminal (DT)', 'Weight Concentration (%)'
]

# Remove missing optional columns
valid_variables = [var for var in variables if var in df.columns]
valid_titles = [titles[i] for i, var in enumerate(variables) if var in df.columns]

for i, var in enumerate(valid_variables):
    axs[i].plot(df['DATA'], df[var], label=var, color='tab:blue')
    axs[i].set_title(valid_titles[i])
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("Operational Data Over Time (Before Filtering)", fontsize=16, y=1.02)
plt.show()


####################################################################################################
# Plotando Diagrama de Operação

# === 2. Constants ===
rho_s = 3.2  # Solids density [g/cm³]
rho_l = 1.0  # Liquid density [g/cm³]
density_threshold = 1.05  # Minimum slurry density to exclude water operation

# === 3. Filter out water operation ===
df_valid = df[df['DEB'] > density_threshold].copy()

# === 4. Compute Weight Concentration (Cw) ===
df_valid['WEIGHT_CONCENTRATION'] = ((df_valid['DEB'] - rho_l) / df_valid['DEB']) * (rho_s / (rho_s - rho_l))

# === 5. Compute Actual Total Solids Production (TSPH) ===
df_valid['TSPH'] = df_valid['QEB'] / ((1 / df_valid['WEIGHT_CONCENTRATION']) - 1 + (1 / rho_s))
df_valid['TSPH'] = df_valid['TSPH'].abs()  # Ensure all values are positive

# === 6. Theoretical Reference Curves ===
flow_range = np.linspace(200, 290, 200)
cw_55 = 0.55
cw_62 = 0.62
tsp_low = flow_range / ((1 / cw_55) - 1 + (1 / rho_s))
tsp_high = flow_range / ((1 / cw_62) - 1 + (1 / rho_s))

# === 7. Prepare Envelope Region Between 220–260 m³/h ===
mask = (flow_range >= 220) & (flow_range <= 260)
flow_fill = flow_range[mask]
tsp_low_fill = tsp_low[mask]
tsp_high_fill = tsp_high[mask]

# === 8. Plotting: Clean & Professional ===
plt.figure(figsize=(12, 6))
plt.xlim(200, 300)
plt.ylim(150, 300)

# Operational envelope fill
plt.fill_between(flow_fill, tsp_low_fill, tsp_high_fill,
                 color='mediumseagreen', alpha=0.3, label='Operational Envelope (60–62%)')

# Theoretical lines
plt.plot(flow_range, tsp_low, 'g--', linewidth=1.5, label='60% CW (Theoretical)')
plt.plot(flow_range, tsp_high, 'g--', linewidth=1.5, label='62% CW (Theoretical)')

# Actual operational data
plt.scatter(df_valid['QEB'], df_valid['TSPH'],
            color='royalblue', s=10, alpha=0.6, label='Operational Data')

# Labels, title, legend, grid
plt.xlabel("Flow Rate [m³/h]", fontsize=12)
plt.ylabel("Total Solids Production (TSPH) [t/h]", fontsize=12)
plt.title("Operational Diagram with Actual Total Solids Production (TSPH)", fontsize=14, weight='bold')
plt.xlim(210, 280)
plt.ylim(180, 300)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper left', fontsize=10, frameon=True)
plt.tight_layout()
plt.show()

#############################################################################################################

import matplotlib.pyplot as plt

# === 6. Determine Operational Status (Inside or Outside Range) ===
df_valid['IN_RANGE'] = (
    (df_valid['QEB'] >= 220) & (df_valid['QEB'] <= 260) &
    (df_valid['WEIGHT_CONCENTRATION'] >= 0.55) & (df_valid['WEIGHT_CONCENTRATION'] <= 0.62)
)

# Count total points
total_points = len(df_valid)
inside_range = df_valid['IN_RANGE'].sum()
outside_range = total_points - inside_range

# === 7. Stylish Pie Chart ===
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#2E8B57', '#E74C3C']  # dark green and strong red
labels = [f'Inside Range ({inside_range})', f'Outside Range ({outside_range})']
explode = (0.05, 0)  # Slightly explode the first slice

wedges, texts, autotexts = ax.pie(
    [inside_range, outside_range],
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    explode=explode,
    wedgeprops=dict(edgecolor='white'),
    textprops=dict(color='black', fontsize=12),
    pctdistance=0.85
)

# Style center circle for donut look
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig.gca().add_artist(centre_circle)

# Title
plt.title("Operational Compliance Summary", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# === 8. Professional Histogram of Weight Concentration ===
import matplotlib.pyplot as plt
import numpy as np

# === Histogram Data ===
data = df_valid['WEIGHT_CONCENTRATION']
bins = np.linspace(0.45, 0.7, 40)
counts, bin_edges = np.histogram(data, bins=bins)
percentages = (counts / counts.sum()) * 100
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# === Plot Setup ===
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(bin_centers, percentages, width=(bin_edges[1] - bin_edges[0]), 
              color='royalblue', edgecolor='black', alpha=0.85)

# === Add Percentage Labels ===
for bar, percent in zip(bars, percentages):
    if percent > 0:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{percent:.1f}%', 
                ha='center', va='bottom', fontsize=9, color='black')

# === Highlight Operational Range 0.55–0.62 ===
ax.axvspan(0.55, 0.62, color='green', alpha=0.1, label='Target Range (55%–62%)')

# === Styling ===
ax.set_xlim(0.45, 0.7)
ax.set_ylim(0, max(percentages) * 1.2)
ax.set_xlabel('Weight Concentration (Cw)', fontsize=12)
ax.set_ylabel('Percentage of Data [%]', fontsize=12)
ax.set_title('Weight Concentration (Cw) Distribution — Operational Histogram', fontsize=14, weight='bold')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.show()



##################################################################################################################


# === Exclude water periods ===
df['exclude_water'] = False
transit_time = timedelta(hours=24)
i = 0
while i < len(df):
    if df.loc[i, 'DEB'] < 1.050:
        start_time = df.loc[i, 'DATA']
        end_time = start_time + transit_time
        mask = (df['DATA'] >= start_time) & (df['DATA'] <= end_time)
        df.loc[mask, 'exclude_water'] = True
        next_valid = df[df['DATA'] > end_time]
        i = next_valid.index.min() if not next_valid.empty else len(df)
    else:
        i += 1

df_valid = df[~df['exclude_water']].copy()

# === Apply moving average filter for steady-state ===
window = 15  # Rolling window size
tolerance = 0.5  # Allowed deviation from moving average [m³/h]

df_valid['QEB_MA'] = df_valid['QEB'].rolling(window, center=True).mean()
df_valid['QEB_DEV'] = np.abs(df_valid['QEB'] - df_valid['QEB_MA'])
df_valid = df_valid[df_valid['QEB_DEV'] <= tolerance].copy()

# === Calculate weight concentration and head loss ===
df_valid['WEIGHT_CONCENTRATION'] = ((df_valid['DEB'] - 1.0) / df_valid['DEB']) * (rho_s / (rho_s - 1.0))

ELEB = 1180.376  # Elevation at pump station
EL45 = 1120.444  # Elevation at km 55
D45 = 55960      # Distance [m]

df_valid['DHDL'] = ((((df_valid['PEB'] - df_valid['P45']) * 10 - (ELEB - EL45)) / df_valid['DEB']) / D45) * 1000

# === Plot Head Loss Over Time ===
plt.figure(figsize=(14, 6))
plt.plot(df_valid['DATA'], df_valid['DHDL'], color='darkred', linewidth=1.5, label="Head Loss (DHDL)")
plt.title("Head Loss Over Time (Steady-State Filter via Moving Average)", fontsize=16, weight='bold')
plt.xlabel("Date & Time")
plt.ylabel("Head Loss (DHDL) [m/km]")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

