import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.ensemble import IsolationForest
from scipy.fft import fft, fftfreq
import pywt
import time
import OpenOPC
from IPython.display import display, clear_output
from plotly.offline import iplot, init_notebook_mode
import plotly.express as px

# Initialize global variables 
model = None 
npw_detections = [] 
ai_detections = [] 
fft_detections = [] 
time_steps = 50
sound_speed=1400
# Initialize notebook mode for Plotly 
init_notebook_mode(connected=True)

# Function to apply wavelet denoising

def apply_wavelet_denoising(signal, wavelet='db1', level=None):
    if level is None:
        level = min(1, pywt.dwt_max_level(len(signal), wavelet))
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    epsilon = 1e-8
    threshold = np.sqrt(2 * np.log(len(signal))) * np.median(np.abs(coeffs[-level]) + epsilon) / 0.6745
    new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(new_coeffs, wavelet)
    
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    elif len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), 'constant')
    
    return denoised_signal

# Train Isolation Forest model
def train_model(data):
    df = pd.DataFrame(data)
    features = df[["pressure_inlet", "pressure_mid", "pressure_outlet", "flow_inlet", "flow_mid", "flow_outlet", "density_inlet", "density_mid", "density_outlet"]]
    model = IsolationForest(contamination=0.05)
    model.fit(features)
    return model

# Function to detect leaks and calculate leak location using NPW method
def detect_leak_npw(df):
    # Compare pressure signals from the current time step and oldest time step
    pressure_diff1 = df["pressure_inlet"].iloc[-1] - df["pressure_inlet"].iloc[0]
    pressure_diff2 = df["pressure_mid"].iloc[-1] - df["pressure_mid"].iloc[0]
    flow_diff1 = df["flow_inlet"].iloc[-1] - df["flow_inlet"].iloc[0]
    flow_diff2 = df["flow_mid"].iloc[-1] - df["flow_mid"].iloc[0]

    # Thresholds for detecting leaks (example values)
    pressure_threshold = 10
    flow_threshold = 10

    leak_detected = (pressure_diff1 > pressure_threshold) & (pressure_diff2 > pressure_threshold) & \
                    (flow_diff1 > flow_threshold) & (flow_diff2 > flow_threshold)

    if leak_detected:
        # Calculate the time difference between pressure drops
        time_diff = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
        # Calculate the leak location
        leak_location = sound_speed * time_diff / 2

        # Volume balance check
        volume_in = df["flow_inlet"].sum()
        volume_out = df["flow_outlet"].sum()
        volume_balance = volume_in - volume_out

        if volume_balance > 0:  # Positive volume imbalance indicates a potential leak
            return True, leak_location
        else:
            return False, None
    else:
        return False, None

# Function to detect leaks using the trained Isolation Forest model
def detect_anomaly(model, datapoint):
    features = np.array([[datapoint["pressure_inlet"], datapoint["pressure_mid"], datapoint["pressure_outlet"], 
                          datapoint["flow_inlet"], datapoint["flow_mid"], datapoint["flow_outlet"], 
                          datapoint["density_inlet"], datapoint["density_mid"], datapoint["density_outlet"]]])
    prediction = model.predict(features)
    return prediction[0] == -1

# Function to perform FFT analysis on pressure signals
def perform_fft(signal, time_steps):
    fft_values = fft(signal)
    fft_freqs = fftfreq(time_steps, 1 / time_steps)
    return fft_freqs, np.abs(fft_values)

# Function to load pipeline profile from Excel file
def load_pipeline_profile_from_excel(file_path): 
    df_profile = pd.read_excel(file_path) 
    return df_profile 
# Load pipeline profile from user input 
file_path = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\List of Anomalies_Rosen_UTM.xlsx"
profile_df = load_pipeline_profile_from_excel(file_path) 
LR = profile_df['L'].values 
HR = profile_df['H'].values

# Function to load data from SCADA system using OPC
# def load_data_from_scada(opc_server, opc_tags):
#     opc = OpenOPC.client()
#     opc.connect(opc_server)
#     while True:
#         data = opc.read(opc_tags)
#         yield dict(zip(opc_tags, data))
#         time.sleep(1)

# Initialize sample data
data = {
    "timestamp": pd.date_range(start="2022-01-01", periods=50, freq="s"),
    "pressure_inlet": np.random.normal(loc=1.0, scale=0.1, size=50),
    "pressure_mid": np.random.normal(loc=1.0, scale=0.1, size=50),
    "pressure_outlet": np.random.normal(loc=1.0, scale=0.1, size=50)
}
df = pd.DataFrame(data)

# Function to update the plot
def update_real_time_plot(df):
    fig = px.line(df, x="timestamp", y=["pressure_inlet", "pressure_mid", "pressure_outlet"],
                  labels={"value": "Pressure", "variable": "Position"},
                  title="Real-Time Pressure Readings")
    clear_output(wait=True)
    display(fig)

# Simulating real-time data updates
for _ in range(10):
    new_data = {
        "timestamp": pd.Timestamp.now(),
        "pressure_inlet": np.random.normal(loc=1.0, scale=0.1),
        "pressure_mid": np.random.normal(loc=1.0, scale=0.1),
        "pressure_outlet": np.random.normal(loc=1.0, scale=0.1)
    }
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    update_real_time_plot(df)
    time.sleep(1)
