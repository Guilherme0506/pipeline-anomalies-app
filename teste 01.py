import pandas as pd
from datetime import datetime, timedelta

# Lower the pressure threshold for testing
pressure_threshold_percentage = 0.0001  # 0.01% for testing purposes

# Simulate more significant pressure drops in the data
data = pd.DataFrame({
    "timestamp": [datetime.now() - timedelta(seconds=i) for i in range(10)],
    "pressure_inlet": [140 + i * 0.1 if i < 5 else 135 for i in range(10)],
    "pressure_mid": [70 + i * 0.05 if i < 5 else 65 for i in range(10)],
    "pressure_outlet": [5 + i * 0.02 if i < 5 else 4 for i in range(10)],
    "flow_inlet": [230 + i for i in range(10)],
    "flow_outlet": [240 + i for i in range(10)],
    "flow_velocity": [1.5 for i in range(10)],
    "density": [1540 + i for i in range(10)]
})

def detect_leak_npw(start_time, sound_speed, pipeline_length, non_leak_start_time, non_leak_end_time, leak_end_time, leak_start_time):
    print(f"Function detect_leak_npw called with start_time: {start_time}, sound_speed: {sound_speed}, pipeline_length: {pipeline_length}")

    # Ensure start_time is a datetime object
    if not isinstance(start_time, datetime):
        raise ValueError("start_time must be a datetime object")
    print(f"start_time is a valid datetime object: {start_time}")

    # Simulated new data
    print(f"New data fetched in detect_leak_npw:\n{data}")

    # Sensor locations (in meters)
    sensors = {"inlet": 0, "mid": 55000, "outlet": 123000}

    # Time for the wave to travel the entire pipeline length
    max_travel_time_sec = pipeline_length / sound_speed
    print(f"Max travel time for wave: {max_travel_time_sec} seconds")

    # Thresholds
    volume_threshold_percentage = 0.05
    print(f"Pressure threshold percentage: {pressure_threshold_percentage}")
    print(f"Volume threshold percentage: {volume_threshold_percentage}")

    # Detect the first pressure drop at any sensor
    pressure_changes = {
        "inlet": data["pressure_inlet"].pct_change().abs().fillna(0),
        "mid": data["pressure_mid"].pct_change().abs().fillna(0),
        "outlet": data["pressure_outlet"].pct_change().abs().fillna(0)
    }
    print(f"Pressure changes:\n{pressure_changes}")

    pressure_drops = {
        "inlet": (pressure_changes["inlet"] >= pressure_threshold_percentage).idxmax(),
        "mid": (pressure_changes["mid"] >= pressure_threshold_percentage).idxmax(),
        "outlet": (pressure_changes["outlet"] >= pressure_threshold_percentage).idxmax()
    }
    print(f"Pressure drops detected: {pressure_drops}")

    # Ensure pressure drops are detected
    if all(v != 0 for v in pressure_drops.values()):
        first_drop_time = min(data["timestamp"].iloc[pressure_drops[k]] for k in pressure_drops)
        print(f"First pressure drop detected at time: {first_drop_time}")

        # Wait for the maximum travel time and check for subsequent drops
        leak_detected = False
        subsequent_drops = []

        for sensor, drop_index in pressure_drops.items():
            drop_time = data["timestamp"].iloc[drop_index]
            time_diff = (drop_time - first_drop_time).total_seconds()
            print(f"Time difference for sensor {sensor}: {time_diff} seconds")

            if time_diff <= max_travel_time_sec:
                leak_location = sensors[sensor] + sound_speed * time_diff
                subsequent_drops.append((sensor, drop_time, leak_location))
                leak_detected = True
                print(f"Pressure drop detected at sensor: {sensor}, time: {drop_time}, calculated leak location: {leak_location}")

        # Confirm the potential leak with volume balance check if drops are detected
        if leak_detected:
            volume_diff_in = data["flow_inlet"].pct_change(fill_method=None).abs().iloc[-1]
            volume_diff_out = data["flow_outlet"].pct_change(fill_method=None).abs().iloc[-1]
            volume_balance = volume_diff_in - volume_diff_out
            print(f"Volume balance calculated: {volume_balance}")

            if -volume_threshold_percentage <= volume_balance <= volume_threshold_percentage:  # Volume imbalance check
                final_leak_location = sum(d[2] for d in subsequent_drops) / len(subsequent_drops)  # Average location
                print(f"Final leak location calculated: {final_leak_location}")
                return True, final_leak_location
            else:
                print(f"Volume balance is out of the threshold range: {volume_balance}")

    print("No significant pressure drop detected or volume imbalance detected.")
    return False, None

# Example usage
start_time = datetime.now()
sound_speed = 1000  # Example sound speed in meters per second
pipeline_length = 123000  # Example pipeline length in meters
non_leak_start_time = start_time  # Example values
non_leak_end_time = start_time
leak_end_time = start_time
leak_start_time = start_time

detect_leak_npw(start_time, sound_speed, pipeline_length, non_leak_start_time, non_leak_end_time, leak_end_time, leak_start_time)


