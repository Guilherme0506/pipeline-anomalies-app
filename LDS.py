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
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
import matplotlib.dates as mdates # Import the dates module


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

# Function to apply wavelet denoising
def apply_wavelet_denoising(series, wavelet='db1', level=1):
    data = np.asarray(series)
    max_level = pywt.dwt_max_level(len(data), wavelet)
    level = min(level, max_level) # Adjust the level to be within a valid range

    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    denoised_coeffs = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]
    denoised_data = pywt.waverec(denoised_coeffs, wavelet)

    # Ensure the length of denoised data matches the original data length
    denoised_data = denoised_data[:len(data)]

    return pd.Series(denoised_data, index=series.index)

def apply_moving_average(series, window_size=30):
    return series.rolling(window=window_size).mean()

# Constants
pipeline_length = 123000  # Length of the pipeline in meters


def detect_leak_npw(df, sound_speed, pipeline_length):
    # Sensor locations
    sensors = {"inlet": 0, "mid": 55000, "outlet": 123000}  # in meters

    # Thresholds
    pressure_threshold_percentage = 10
    volume_threshold_percentage = 5

    # Apply moving average to pressure and flow signals
    df["pressure_inlet"] = apply_moving_average(df["pressure_inlet"])
    df["pressure_mid"] = apply_moving_average(df["pressure_mid"])
    df["pressure_outlet"] = apply_moving_average(df["pressure_outlet"])
    df["flow_inlet"] = apply_moving_average(df["flow_inlet"])
    df["flow_outlet"] = apply_moving_average(df["flow_outlet"])

    # Initialize npw_detections list (moved outside the loop)
    npw_detections = []  

    while True:
        # Detect the first pressure drop at any sensor
        pressure_drops = {
            "inlet": (df["pressure_inlet"].pct_change(fill_method=None).abs() >= pressure_threshold_percentage).idxmax(),
            "mid": (df["pressure_mid"].pct_change(fill_method=None).abs() >= pressure_threshold_percentage).idxmax(),
            "outlet": (df["pressure_outlet"].pct_change(fill_method=None).abs() >= pressure_threshold_percentage).idxmax()
        }

        first_drop_sensor = min(pressure_drops, key=lambda k: df["timestamp"].iloc[pressure_drops[k]])
        first_drop_time = df["timestamp"].iloc[pressure_drops[first_drop_sensor]]

        # Check for subsequent pressure drops in the other sensors
        subsequent_drops = []
        for sensor, drop_index in pressure_drops.items():
            if sensor != first_drop_sensor:
                drop_time = df["timestamp"].iloc[drop_index]
                if drop_time > first_drop_time:
                    time_diff = (drop_time - first_drop_time).total_seconds()
                    leak_location = sound_speed * time_diff / 2
                    subsequent_drops.append((sensor, drop_time, leak_location))

        # Pre-classify based on subsequent pressure drops and leak location
        potential_leak = False
        for i in range(len(subsequent_drops)):
            for j in range(i + 1, len(subsequent_drops)):
                if sensors[subsequent_drops[i][0]] <= subsequent_drops[i][2] <= sensors[subsequent_drops[j][0]]:
                    potential_leak = True
                    break

        if potential_leak:
            # Confirm the potential leak with volume balance check
            volume_diff_in = df["flow_inlet"].pct_change().abs().iloc[-1]
            volume_diff_out = df["flow_outlet"].pct_change().abs().iloc[-1]
            volume_balance = volume_diff_in - volume_diff_out

            if -volume_threshold_percentage <= volume_balance <= volume_threshold_percentage:  # Volume imbalance check
                leak_location = (subsequent_drops[0][2] + subsequent_drops[1][2]) / 2
                return True, leak_location

        # Clean the previously processed data to avoid interference with the next iteration
        df = df.iloc[1:].reset_index(drop=True)

        # If no more data is available, break the loop
        if df.empty:
            break

        # Get the current time as start_time 
        start_time = datetime.now() 

        # Call detect_leak_npw and store the results (using the function itself)
        leak_detected_npw, leak_location_npw = detect_leak_npw(df, sound_speed, pipeline_length)

        # Process the results
        if leak_detected_npw:
            npw_detections.append((start_time, leak_location_npw))
            print(f"Leak detected using NPW at {leak_location_npw:.2f} meters from the inlet at {start_time}")

    return False, None
  
    start_time = datetime.now() 

    leak_detected_npw, leak_location_npw = detect_leak_npw(df, sound_speed, pipeline_length)

    if leak_detected_npw:
        npw_detections.append((start_time, leak_location_npw))
        print(f"Leak detected using NPW at {leak_location_npw:.2f} meters from the inlet at {start_time}")

# Initialize global variables
model = None
start_time = pd.Timestamp.now()  # Initialize start_time as a Timestamp
npw_detections = []
ai_detections = []
fft_detections = []
x_data = []
y_data_pressure_inlet = []
y_data_pressure_mid = []
y_data_pressure_outlet = []
y_data_flow_inlet = []
y_data_flow_outlet = []
y_data_batch_position = []
slurry_positions = []
slurry_heights = []
MAX_POINTS = 100  # Example max points
PIPELINE_LENGTH = 123000  # Example pipeline length in meters


# Define constants
leak_location_km = 70  # Example value, set according to your requirement
sound_speed = 1000  # Speed of sound in air in m/s (example value)
start_time = datetime.now()

def fetch_new_data(start_time):
    print(f"Function fetch_new_data called with start_time: {start_time}")  # Debug print statement

    time_to_inlet_sec = np.abs(123-leak_location_km) * 1000 / sound_speed
    time_to_mid_sec = np.abs((55 - leak_location_km) * 1000) / sound_speed
    time_to_outlet_sec = np.abs((123 - leak_location_km) * 1000) / sound_speed

    print(f"time_to_inlet_sec: {time_to_inlet_sec}")  # Debug print for time to inlet
    print(f"time_to_mid_sec: {time_to_mid_sec}")  # Debug print for time to mid
    print(f"time_to_outlet_sec: {time_to_outlet_sec}")  # Debug print for time to outlet

    non_leak_duration = timedelta(seconds=70)
    leak_duration = timedelta(seconds=70)

    # Initialize data lists
    timestamps = []
    flow_inlet = []
    flow_outlet = []
    pressure_inlet = []
    pressure_mid = []
    pressure_outlet = []
    density = []

    # Non-leak event
    non_leak_end_time = start_time + non_leak_duration
    current_time = start_time

    print(f"Non-leak event start: {start_time}")

    while current_time <= non_leak_end_time:
        timestamps.append(current_time)
        flow_inlet.append(random.uniform(230, 245))
        flow_outlet.append(random.uniform(230, 245))
        pressure_inlet.append(random.uniform(130, 145))
        pressure_mid.append(random.uniform(65, 75))
        pressure_outlet.append(random.uniform(3, 5))
        density.append(random.uniform(1400, 1700))
        
        current_time += timedelta(seconds=0.1)
        print(f"Non-leak event current_time: {current_time}")  # Debug print statement inside the loop

    print(f"Non-leak event end: {current_time}")
    leak_start_time = current_time  
    print(f"Leak event start: {leak_start_time}")  # Debug print statement for leak start time
    leak_end_time = leak_start_time + leak_duration

    while current_time <= leak_end_time:
        timestamps.append(current_time)
        
        # Generate initial pressure and flow values
        flow_inlet_val = random.uniform(230, 245)
        flow_outlet_val = random.uniform(230, 245)
        pressure_inlet_val = random.uniform(130, 145)
        pressure_mid_val = random.uniform(65, 75)
        pressure_outlet_val = random.uniform(3, 5)

        elapsed_time_leak = (current_time - leak_start_time).total_seconds()
        print(f"elapsed_time_leak: {elapsed_time_leak}, time_to_inlet_sec: {time_to_inlet_sec}, time_to_mid_sec: {time_to_mid_sec}, time_to_outlet_sec: {time_to_outlet_sec}")  # Debug print for elapsed time and thresholds

        # Apply pressure and flow reductions before randomization
        if elapsed_time_leak >= time_to_inlet_sec:
            pressure_inlet_val -= 50  # Increased reduction
            flow_inlet_val -= 20  # Increased reduction
            print(f"Reduction applied at inlet: pressure_inlet_val={pressure_inlet_val}, flow_inlet_val={flow_inlet_val}")  # Debug print for reduction
        if elapsed_time_leak >= time_to_mid_sec:
            pressure_mid_val -= 50  # Increased reduction
            print(f"Reduction applied at mid: pressure_mid_val={pressure_mid_val}")  # Debug print for reduction
        if elapsed_time_leak >= time_to_outlet_sec:
            pressure_outlet_val -= 50  # Increased reduction
            flow_outlet_val -= 20  # Increased reduction
            print(f"Reduction applied at outlet: pressure_outlet_val={pressure_outlet_val}, flow_outlet_val={flow_outlet_val}")  # Debug print for reduction

        # Append values to lists
        flow_inlet.append(flow_inlet_val)
        flow_outlet.append(flow_outlet_val)
        pressure_inlet.append(pressure_inlet_val)
        pressure_mid.append(pressure_mid_val)
        pressure_outlet.append(pressure_outlet_val)
        density.append(random.uniform(1400, 1700))

        current_time += timedelta(seconds=0.1)
        print(f"Leak event current_time: {current_time}")  # Debug print statement inside the loop

    print(f"Leak event end: {current_time}")  # Debug print statement for leak end time

    return pd.DataFrame({
        "timestamp": timestamps,
        "pressure_inlet": pressure_inlet,
        "pressure_mid": pressure_mid,
        "pressure_outlet": pressure_outlet,
        "flow_inlet": flow_inlet,
        "flow_outlet": flow_outlet,
        "flow_velocity": [1.5 for _ in range(len(timestamps))],
        "density": density
    })


# Example usage
data_df = fetch_new_data(start_time)


# Example usage
current_cycle = 0
print(f"Initial start time: {start_time}")  # Debug print
print(f"Length of data_df: {len(data_df)}")  # Debug print to display the length of the data frame

def linear_interpolate(lr_values, hr_values, x_values):
    y_values = [np.interp(x, lr_values, hr_values) for x in x_values]
    return y_values

def init_pipeline_profile():
    global fig, axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plt.ion()  # Enable interactive mode

def animate(i):
    global model, x_data, y_data_pressure_inlet, y_data_pressure_mid, y_data_pressure_outlet, y_data_flow_inlet, y_data_flow_outlet, data_df, current_cycle

    print(f"Animating frame: {i}, current_cycle: {current_cycle}")  # Debug print for `i`

    # Ensure the animation index does not exceed the length of the data
    if i == len(data_df):
        current_cycle += 1
        print("Fetching new data...")  # Debug print
        # Use the last timestamp in the current data as the start time for the next cycle
        current_time = data_df['timestamp'].iloc[-1] + timedelta(seconds=0.1)
        data_df = fetch_new_data(current_time)
        print(f"New data fetched, length of data_df: {len(data_df)}, resetting i to 0")  # Debug print
        i = 0  # Reset index to start from the beginning of the new data set
        return  # Exit the function to prevent further plotting in this iteration

    df = data_df.iloc[i % len(data_df)]  # Use modulo to wrap around the index

    # Update data for plotting
    x_data.append(df['timestamp'])
    y_data_pressure_inlet.append(df['pressure_inlet'])
    y_data_pressure_mid.append(df['pressure_mid'])
    y_data_pressure_outlet.append(df['pressure_outlet'])
    y_data_flow_inlet.append(df['flow_inlet'])
    y_data_flow_outlet.append(df['flow_outlet'])

  
    print(f"Updated data for frame: {i}")  # Debug print for `i`

    # Update batch position based on flow velocity
    flow_velocity = df["flow_velocity"]
    delta_time_seconds = 0.1  # Time interval in seconds

    # Ensure positive update of flow velocity
    batch_position_update = flow_velocity * delta_time_seconds
    new_batch_position = y_data_batch_position[-1] + batch_position_update if y_data_batch_position else batch_position_update
    new_batch_position = max(0, new_batch_position)  # Ensure non-negative position
    new_batch_position = min(PIPELINE_LENGTH, new_batch_position)  # Ensure it doesn't exceed the pipeline length
    y_data_batch_position.append(new_batch_position)

    # Use batch positions as slurry positions and interpolate heights
    slurry_positions.append(new_batch_position)
    slurry_heights = linear_interpolate(LR, HR, slurry_positions)

    # Clear previous plot
    for ax in axes:
        ax.clear()

    # Plot updated data for the first subplot (Pressure)
    axes[0].plot(x_data, y_data_pressure_inlet, label="Pressão na Entrada")
    axes[0].plot(x_data, y_data_pressure_mid, label="Pressão no Meio")
    axes[0].plot(x_data, y_data_pressure_outlet, label="Pressão na Saída")

    # Adjust x-axis labels frequency to improve readability
    label_interval = max(1, len(x_data) // 10)  # Show 10 labels at most
    axes[0].set_xticks(x_data[::label_interval])  # Adjust the frequency of labels
    axes[0].set_xticklabels([dt.strftime('%H:%M:%S.%f')[:-3] for dt in x_data[::label_interval]], rotation=45)

    axes[0].legend(loc='upper left')
    axes[0].set_title("Dados de Pressão ao Longo do Tempo")  # Subplot title
    axes[0].set_ylabel('Pressão (Pa)')
    axes[0].set_xlabel('Data e Hora')

    # Plot updated data for the second subplot (Flow)
    axes[1].plot(x_data, y_data_flow_inlet, label="Fluxo na Entrada")
    axes[1].plot(x_data, y_data_flow_outlet, label="Fluxo na Saída")

    # Adjust x-axis labels frequency to improve readability
    axes[1].set_xticks(x_data[::label_interval])  # Adjust the frequency of labels
    axes[1].set_xticklabels([dt.strftime('%H:%M:%S.%f')[:-3] for dt in x_data[::label_interval]], rotation=45)

    axes[1].legend(loc='upper left')
    axes[1].set_title("Dados de Fluxo ao Longo do Tempo")  # Subplot title
    axes[1].set_ylabel('Fluxo (m³/s)')
    axes[1].set_xlabel('Data e Hora')

    # Plot pipeline profile for the third subplot
    axes[2].plot(LR, HR, label="Perfil do Pipeline", color='blue')

    # Plot slurry positions (batch positions)
    axes[2].scatter(slurry_positions, slurry_heights, color='red', zorder=5, label="Posições da Interface do Slurry")

    # Highlight updated batch position in green
    axes[2].scatter([new_batch_position], [linear_interpolate(LR, HR, [new_batch_position])[0]], color='green', zorder=5, label="Posição Atual do Lote")

    # Plot leak detections
    if npw_detections:
        npw_times, npw_locations = zip(*npw_detections)
        axes[2].scatter(npw_locations, linear_interpolate(LR, HR, npw_locations), color='orange', zorder=5, label="Detecções de Vazamento NPW")

 
    if fft_detections:
        fft_times = fft_detections
        fft_locations = [random.uniform(0, PIPELINE_LENGTH) for _ in fft_detections]
        axes[2].scatter(fft_locations, linear_interpolate(LR, HR, fft_locations), color='purple', zorder=5, label="Detecções de Vazamento FFT")

    if ai_detections:
        ai_locations = [new_batch_position for _ in ai_detections]  # Assuming current batch position for AI detections
        axes[2].scatter(ai_locations, linear_interpolate(LR, HR, ai_locations), color='yellow', zorder=5, label="Detecções de Anomalia IA")

    # Set x-axis limits for the third subplot to the pipeline length
    axes[2].set_xlim(0, PIPELINE_LENGTH)
    axes[2].set_title("Perfil do Pipeline com Detecções de Vazamento e Anomalia")  # Subplot title
    axes[2].set_xlabel('Distância (m)')
    axes[2].set_ylabel('Altura (m)')
    axes[2].legend(loc='upper left')

    plt.tight_layout(pad=2)
    plt.subplots_adjust(bottom=0.2)  # Adjust space for x-axis labels

    fig.canvas.draw()  # Force canvas update


# Initialize and run the animation
init_pipeline_profile()
fig, axes = plt.subplots(3, 1, figsize=(15, 15))  # Adjusted to create three subplots
animation = FuncAnimation(fig, animate, frames=range(len(data_df) * 2), interval=100, repeat=False, blit=False)
plt.show(block=True)


