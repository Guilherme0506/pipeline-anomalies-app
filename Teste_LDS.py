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
import cProfile


# Function to load pipeline profile from Excel file

def load_pipeline_profile_from_excel(file_path): 
    df_profile = pd.read_excel(file_path) 
    return df_profile 
# Load pipeline profile from user input 
file_path = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\List of Anomalies_Rosen_UTM.xlsx"
profile_df = load_pipeline_profile_from_excel(file_path) 
LR = profile_df['L'].values 
HR = profile_df['H'].values

n=1
# Function to load data from SCADA system using OPC
# def load_data_from_scada(opc_server, opc_tags):
#     opc = OpenOPC.client()
#     opc.connect(opc_server)
#     while True:
#         data = opc.read(opc_tags)
#         yield dict(zip(opc_tags, data))
#         time.sleep(1)

# Constants
pipeline_length = 123000  # Length of the pipeline in meters


# Initialize global variables
window_size=123
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
leak_location_km = 20  # Example value, set according to your requirement
sound_speed = 1000  # Speed of sound in air in m/s (example value)
start_time = datetime.now()
current_time = start_time
print(start_time)
print(current_time)




# def apply_moving_average(series, window_size):
#     if len(series) >= window_size:
#         return series.rolling(window=window_size).mean()
#     else:
#         return series  # Return original series if not enough data

""" def update_accumulated_data(start_time,max_length=30):
    global accumulated_data

    # Fetch new data
    start_time, new_data = fetch_new_data(start_time)
    print(f"New data fetched: {new_data}")  # Debug print

    # Append new data to the accumulated DataFrame
    accumulated_data = pd.concat([accumulated_data, new_data], ignore_index=True)
    print(f"Accumulated data updated: {accumulated_data}")  # Debug print

    # Keep only the last `max_length` rows
    if len(accumulated_data) > max_length:
        accumulated_data = accumulated_data.iloc[-max_length:].reset_index(drop=True)
        print(f"Accumulated data trimmed to last {max_length} rows")  # Debug print

    # Modified: Apply moving average only if there are at least 30 points, otherwise keep original values
    for column in ["pressure_inlet", "pressure_mid", "pressure_outlet", "flow_inlet", "flow_outlet"]:
        if accumulated_data[column].count() >= 30:
            # Apply moving average to the entire column if there are enough points
            accumulated_data[column] = apply_moving_average(accumulated_data[column], window_size=30)  
        # if less than 30 points, keep original values for that column      

    print(f"Moving average applied to accumulated data: {accumulated_data}")  # Debug print

    # Return the entire accumulated data
    return start_time, accumulated_data """

# def apply_wavelet_denoising(series, wavelet='db1', level=1):
#     data = np.asarray(series)
#     max_level = pywt.dwt_max_level(len(data), wavelet)
#     level = min(level, max_level) # Adjust the level to be within a valid range

#     coeffs = pywt.wavedec(data, wavelet, level=level)
#     sigma = np.median(np.abs(coeffs[-1])) / 0.6745
#     threshold = sigma * np.sqrt(2 * np.log(len(data)))
#     denoised_coeffs = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]
#     denoised_data = pywt.waverec(denoised_coeffs, wavelet)

#     # Ensure the length of denoised data matches the original data length
#     denoised_data = denoised_data[:len(data)]

#     return pd.Series(denoised_data, index=series.index)

def linear_interpolate(LR, HR, x_values):
    y_values = [np.interp(x, LR, HR) for x in x_values]
    return y_values

def init_pipeline_profile():
    global fig, axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plt.ion()  # Enable interactive mode


def fetch_new_data(start_time, leak_location_km, sound_speed, non_leak_start_time, non_leak_end_time, leak_end_time, leak_start_time):  # Added arguments
    global current_time
    #print(f"Function fetch_new_data called with current time: {current_time}")

    # Calculate time thresholds based on leak location and sound speed
    time_to_mid_sec = np.abs((55 - leak_location_km) * 1000) / sound_speed
    time_to_outlet_sec = np.abs((123 - leak_location_km) * 1000) / sound_speed
    time_to_inlet_sec = np.abs((leak_location_km-0) * 1000) / sound_speed

    # Initialize lists to store data
    timestamps = []
    flow_inlet = []
    flow_outlet = []
    pressure_inlet = []
    pressure_mid = []
    pressure_outlet = []
    density = []

    # Check event conditions and generate data
    if start_time <= current_time <= non_leak_end_time: # Non-leak period based on start_time
        data_point = {
            'timestamp': current_time,
            'flow_inlet': random.uniform(240, 245),
            'flow_outlet': random.uniform(240, 245),
            'pressure_inlet': random.uniform(140, 145),
            'pressure_mid': random.uniform(70, 75),
            'pressure_outlet': random.uniform(5, 5),
            'density': random.uniform(1400, 1700)
        }

        timestamps.append(data_point['timestamp'])
        flow_inlet.append(data_point['flow_inlet'])
        flow_outlet.append(data_point['flow_outlet'])
        pressure_inlet.append(data_point['pressure_inlet'])
        pressure_mid.append(data_point['pressure_mid'])
        pressure_outlet.append(data_point['pressure_outlet'])
        density.append(data_point['density'])

    elif non_leak_end_time < current_time <= leak_end_time:  # Leak period based on start_time
        data_point = {
            'timestamp': current_time,
            'flow_inlet': random.uniform(240, 245),
            'flow_outlet': random.uniform(240, 245),
            'pressure_inlet': random.uniform(140, 145),
            'pressure_mid': random.uniform(70, 75),
            'pressure_outlet': random.uniform(5, 5),
            'density': random.uniform(1400, 1700)
        }

        elapsed_time_leak = (current_time - leak_start_time).total_seconds()
        #print(f"elapsed_time_leak: {elapsed_time_leak}, time_to_inlet_sec: {time_to_inlet_sec}, time_to_mid_sec: {time_to_mid_sec}, time_to_outlet_sec: {time_to_outlet_sec}")

        pressure_inlet_val = data_point['pressure_inlet']
        pressure_mid_val = data_point['pressure_mid']
        pressure_outlet_val = data_point['pressure_outlet']
        flow_inlet_val = data_point['flow_inlet']
        flow_outlet_val = data_point['flow_outlet']

        if elapsed_time_leak >= time_to_inlet_sec:
            pressure_inlet_val -= 20
            flow_outlet_val -= 0
            #print(f"Reduction applied at inlet: pressure_inlet_val={pressure_inlet_val}, flow_inlet_val={flow_inlet_val}")

        if elapsed_time_leak >= time_to_mid_sec:
            pressure_mid_val -= 20
            flow_outlet_val -= 10
            #print(f"Reduction applied at mid: pressure_mid_val={pressure_mid_val}")

        if elapsed_time_leak >= time_to_outlet_sec:
            pressure_outlet_val -= 20
            flow_outlet_val -= 0
            #print(f"Reduction applied at outlet: pressure_outlet_val={pressure_outlet_val}, flow_outlet_val={flow_outlet_val}")

        timestamps.append(current_time)
        flow_inlet.append(flow_inlet_val)
        flow_outlet.append(flow_outlet_val)
        pressure_inlet.append(pressure_inlet_val)
        pressure_mid.append(pressure_mid_val)
        pressure_outlet.append(pressure_outlet_val)
        density.append(data_point['density'])

    else:  # Transition to a new cycle
        non_leak_start_time = start_time
        non_leak_end_time = non_leak_start_time + timedelta(seconds=123)
        leak_start_time = non_leak_end_time + timedelta(seconds=2.5)
        leak_end_time = leak_start_time + timedelta(seconds=123)

        #print(f"Transitioning to new cycle. Updated event times:")
        #print(f"  non_leak_start_time: {non_leak_start_time}")
        #print(f"  non_leak_end_time: {non_leak_end_time}")
        #print(f"  leak_start_time: {leak_start_time}")
        #print(f"  leak_end_time: {leak_end_time}")

        data_point = {
            'timestamp': current_time,
            'flow_inlet': random.uniform(240, 245),
            'flow_outlet': random.uniform(240, 245),
            'pressure_inlet': random.uniform(140, 145),
            'pressure_mid': random.uniform(70, 75),
            'pressure_outlet': random.uniform(4, 5),
            'density': random.uniform(1400, 1700)
        }

        timestamps.append(data_point['timestamp'])
        flow_inlet.append(data_point['flow_inlet'])
        flow_outlet.append(data_point['flow_outlet'])
        pressure_inlet.append(data_point['pressure_inlet'])
        pressure_mid.append(data_point['pressure_mid'])
        pressure_outlet.append(data_point['pressure_outlet'])
        density.append(data_point['density'])

    data_df = pd.DataFrame({
        "timestamp": timestamps,
        "pressure_inlet": pressure_inlet,
        "pressure_mid": pressure_mid,
        "pressure_outlet": pressure_outlet,
        "flow_inlet": flow_inlet,
        "flow_outlet": flow_outlet,
        "flow_velocity": [1.5 for _ in range(len(timestamps))],
        "density": density,
        "pressure_inlet_time": timestamps,  # New column
        "pressure_mid_time": timestamps,    # New column
        "pressure_outlet_time": timestamps   # New column
    })

    if data_df.empty:
        print("No new data fetched. DataFrame is empty.")
    
    return data_df

from datetime import datetime, timedelta

# Define sensor positions and thresholds
sensors = {"inlet": 0, "mid": 55000, "outlet": 123000}
pressure_threshold_percentage = 0.1
volume_threshold_percentage = 0.02
previous_data = {}
pressure_drop_times = {sensor: None for sensor in sensors}
first_drop_time = None
window_end_time = None
window_active = False
pressure_drop_count = 0

def detect_leak_npw(start_time, sound_speed, pipeline_length, non_leak_start_time, non_leak_end_time, leak_end_time, leak_start_time, data_df):
    global previous_data, pressure_drop_times, first_drop_time, window_end_time, window_active, pressure_drop_count,time_to_inlet, time_to_outlet, time_to_mid

    window_time = timedelta(seconds=pipeline_length / sound_speed)

    for i, row in data_df.iterrows():
        print(f"Processing row {i}: timestamp = {row['timestamp']}")

        for sensor in sensors:
            current_pressure = row[f"pressure_{sensor}"]
            previous_pressure = previous_data.get(sensor, current_pressure)
            print(f"**DEBUG: {sensor} - Current Pressure: {current_pressure}, Previous Pressure: {previous_pressure}**")

            # Detect significant pressure drop compared to previous pressure
            if current_pressure <= previous_pressure * (1 - pressure_threshold_percentage):
                if pressure_drop_times[sensor] is None:
                    pressure_drop_times[sensor] = row["timestamp"]
                    pressure_drop_count += 1
                    print(f"**DEBUG: Pressure drop detected at {sensor} at {row['timestamp']} (Pressure: {current_pressure})**")
                    print(f"**DEBUG: Pressure drop count: {pressure_drop_count}**")

                    if not window_active:
                        first_drop_time = row["timestamp"]
                        window_end_time = first_drop_time + window_time
                        window_active = True
                        print(f"**DEBUG: Window active from {first_drop_time} to {window_end_time}**")

        # Update previous data for the next timestamp
        previous_data = {
            "inlet": row["pressure_inlet"],
            "mid": row["pressure_mid"],
            "outlet": row["pressure_outlet"]
        }
        print("**DEBUG: Previous data updated for the next timestamp**")

        # Check if pressure drop detected in all sensors within the window
        if window_active and row["timestamp"] >= window_end_time:
            print("**DEBUG: Window time exceeded without detecting required drops. Resetting.**")
            window_active = False
            pressure_drop_times = {sensor: None for sensor in sensors}
            pressure_drop_count = 0
            first_drop_time = None

        if pressure_drop_count >= 3:
            print("**DEBUG: Pressure drops detected in all sensors within the window. Evaluating leak location and volume imbalance.**")
            break

    # Calculate leak location if detected in all sensors within the window
    if pressure_drop_count >= 3:
        detected_sensors = [sensor for sensor in sensors if pressure_drop_times[sensor] is not None]
        leak_locations = []

    # Iterate through the detected sensors in pairs
        for j in range(len(detected_sensors) - 1):
            sensor1, sensor2 = detected_sensors[j], detected_sensors[j + 1]
            time_diff = ((pressure_drop_times[sensor1] - pressure_drop_times[sensor2]).total_seconds())
            print(f"DT Detecção {time_diff}")
            sum_of_sensor_positions = sensors[sensor1] + sensors[sensor2]
            print(f"Soma Posição Sensores{sum_of_sensor_positions} m")
            leak_location = (sound_speed * time_diff + sum_of_sensor_positions) / 2
            print(f"Leak location{leak_location} m")

        # Check if the calculated leak location is between the sensor locations
        if (sensors[sensor1]) <= leak_location <= (sensors[sensor2]):
            leak_locations.append(leak_location)
            print(f"Leak location between {sensor1} and {sensor2}: {leak_location} meters")

    # Print all detected leak locations
        if leak_locations:
            for location in leak_locations:
                print(f"Determined leak location: {location} meters")
                # Perform imbalance calculation using accumulated flow
            accumulated_flow_inlet = data_df[(data_df["timestamp"] >= first_drop_time) & (data_df["timestamp"] <= window_end_time)]["flow_inlet"].sum()
            accumulated_flow_outlet = data_df[(data_df["timestamp"] >= first_drop_time) & (data_df["timestamp"] <= window_end_time)]["flow_outlet"].sum()
            volume_diff = abs(accumulated_flow_inlet - accumulated_flow_outlet)

            print(f"**DEBUG: Accumulated Flow Inlet: {accumulated_flow_inlet}**")
            print(f"**DEBUG: Accumulated Flow Outlet: {accumulated_flow_outlet}**")
            print(f"**DEBUG: Volume Difference: {volume_diff}**")

            if volume_diff >= volume_threshold_percentage:
                print("Leak confirmed!")
                return True, leak_location
            else:
                print("Volume balance within threshold. Leak not confirmed.")
                # Reset variables if leak not confirmed
                print("**DEBUG: Cleaning data after non-confirmation of leak.**")
                pressure_drop_times = {sensor: None for sensor in sensors}
                first_drop_time = None
                window_end_time = None
                window_active = False
                pressure_drop_count = 0
                return False, None

        else:
            print("No leak location found within the sensor positions. Leak not confirmed")
            print("**DEBUG: Cleaning data after non-confirmation of leak.**")
            pressure_drop_times = {sensor: None for sensor in sensors}
            first_drop_time = None
            window_end_time = None
            window_active = False
            pressure_drop_count = 0
            return False, None

    print("**DEBUG: No leak confirmed in the given data.**")
    return False, None


n=0

def animate(i, fig, axes):
    global start_time, x_data, y_data_pressure_inlet, y_data_pressure_mid, y_data_pressure_outlet
    global y_data_flow_inlet, y_data_flow_outlet, y_data_batch_position, slurry_positions, current_time, non_leak_start_time, n,frame_start_time

    frame_start_time=datetime.now()
    
    current_time += timedelta(seconds=2.5)
    #print(f"Animating frame {i} at time {current_time}")

    non_leak_start_time = start_time
    non_leak_duration = timedelta(seconds=123)
    non_leak_end_time = non_leak_start_time + non_leak_duration
    leak_start_time = non_leak_start_time + non_leak_duration
    leak_duration = timedelta(seconds=123)
    leak_end_time = leak_start_time + leak_duration

    reset_occurred = False 
    if current_time >= leak_end_time and not reset_occurred: 
        start_time = current_time + timedelta(seconds=2.5) 
        non_leak_start_time = start_time
        non_leak_duration = timedelta(seconds=123)
        non_leak_end_time = non_leak_start_time + non_leak_duration
        leak_start_time = non_leak_start_time + non_leak_duration
        leak_duration = timedelta(seconds=123)
        leak_end_time = leak_start_time + leak_duration
        #print(f"Resetting for next cycle: start_time={start_time}") 
        n += 1 
        reset_occurred = True

    # print(f"Non Leak event start: {non_leak_start_time}")
    # print(f"Non Leak event duration: {non_leak_duration}")
    # print(f"Non Leak event end time: {non_leak_end_time}")
    # print(f"Leak event start: {leak_start_time}")
    # print(f"Leak event duration: {leak_duration}")
    # print(f"Leak event end time: {leak_end_time}")
    # print(f"Current_time: {current_time}")

    new_data = fetch_new_data(start_time, leak_location_km, sound_speed, non_leak_start_time, non_leak_end_time, leak_end_time, leak_start_time)
    #print("New data fetched:", new_data)

    leak_detected, leak_location = detect_leak_npw(start_time, sound_speed, pipeline_length, non_leak_start_time, non_leak_end_time, leak_end_time, leak_start_time,new_data) 

    if new_data.empty:
       # print("DataFrame is empty. Skipping this frame.")
        return

    x_data.append(new_data['timestamp'].iloc[0])
    y_data_pressure_inlet.append(new_data['pressure_inlet'].iloc[0])
    y_data_pressure_mid.append(new_data['pressure_mid'].iloc[0])
    y_data_pressure_outlet.append(new_data['pressure_outlet'].iloc[0])
    y_data_flow_inlet.append(new_data['flow_inlet'].iloc[0])
    y_data_flow_outlet.append(new_data['flow_outlet'].iloc[0])
    flow_velocity = new_data['flow_velocity'].iloc[0]

    # Clean data older than 140 seconds
    while (x_data[-1] - x_data[0]).total_seconds() >= 150:
        x_data.pop(0)
        y_data_pressure_inlet.pop(0)
        y_data_pressure_mid.pop(0)
        y_data_pressure_outlet.pop(0)
        y_data_flow_inlet.pop(0)
        y_data_flow_outlet.pop(0)


    delta_time_seconds = 2.5
    batch_position_update = flow_velocity * delta_time_seconds
    new_batch_position = (y_data_batch_position[-1] + batch_position_update if y_data_batch_position else batch_position_update)
    new_batch_position = max(0, new_batch_position)
    new_batch_position = min(PIPELINE_LENGTH, new_batch_position)
    y_data_batch_position.append(new_batch_position)

    slurry_positions.append(new_batch_position)
    slurry_heights = linear_interpolate(LR, HR, slurry_positions)

    # Adjust delta_time based on actual frame duration 
    frame_end_time = datetime.now()
    frame_duration = (frame_end_time - frame_start_time).total_seconds() 
    delta_time_seconds = frame_duration
    #print("Passo de tempo computational:", delta_time_seconds)

    for ax in axes:
        ax.cla()

    axes[0].plot(x_data, y_data_pressure_inlet, label="Pressão na Entrada")
    axes[0].plot(x_data, y_data_pressure_mid, label="Pressão no Meio")
    axes[0].plot(x_data, y_data_pressure_outlet, label="Pressão na Saída")
    axes[0].legend(loc='upper left')
    axes[0].set_title("Dados de Pressão ao Longo do Tempo")
    axes[0].set_ylabel('Pressão (Pa)')
    axes[0].set_xlabel('Data e Hora')

    axes[1].plot(x_data, y_data_flow_inlet, label="Fluxo na Entrada")
    axes[1].plot(x_data, y_data_flow_outlet, label="Fluxo na Saída")
    axes[1].legend(loc='upper left')
    axes[1].set_title("Dados de Fluxo ao Longo do Tempo")
    axes[1].set_ylabel('Fluxo (m³/s)')
    axes[1].set_xlabel('Data e Hora')

    axes[2].plot(LR, HR, label="Perfil do Pipeline", color='blue')
    axes[2].scatter(slurry_positions, slurry_heights, color='red', zorder=5, label="Posições da Interface do Slurry")
    axes[2].scatter([new_batch_position], [linear_interpolate(LR, HR, [new_batch_position])[0]], color='green', zorder=5, label="Posição Atual do Lote")

    axes[2].set_xlim(0, PIPELINE_LENGTH)
    axes[2].set_xlabel('Distância (m)')
    axes[2].set_ylabel('Altura (m)')
    axes[2].set_title("Perfil do Pipeline com Detecções de Vazamento e Anomalia")
    axes[2].legend(loc='upper left')

    for ax in axes[:2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout(pad=2)
    plt.subplots_adjust(bottom=0.2)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

    #print("End animate:", time.time())
    return axes[0].lines + axes[1].lines

def infinite_frames():
    i = 0
    while True:
        yield i
        i += 1

# Animation setup
fig, axes = plt.subplots(3, 1, figsize=(15, 15))
animation = FuncAnimation(fig, animate, fargs=(fig, axes), frames=infinite_frames(), interval=2000, repeat=False, blit=True)  # Use infinite_frames() for frames
plt.show()