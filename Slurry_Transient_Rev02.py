import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FFMpegWriter

# Parameters
L = 123000               # Tube length
Nx = 4433*3              # Number of nodes in the tube
Nt = 40000*3             # Number of time steps
dx = L / (Nx-1)
tf = 1200                # Total simulation time
dt = tf / (Nt - 1)
tav = 30                 # Valve opening time
g = 9.81                 # Gravity acceleration
E = 200e9                # Pipe material elastic modulus
K = 1.75e9               # Fluid elastic modulus
D = (9.625*25.4)/1000    # Pipe External Diameter
esp = 0.384*25.4 / 1000  # Wall thickness
Dint = D - 2 * esp
A = np.pi * 0.25 * Dint**2   # Cross-sectional area
f = 0.0175               # Darcy-Weisbach resistance factor
rho = 1700               # Liquid density

# Read pipeline profile data
RP = input("Enter the file path for the pipeline profile: ")
df = pd.read_excel(RP)
LR, HR = df['L'].values, df['H'].values

# Determine the maximum elevation in the pipeline profile (Static Head)
static_head = np.max(HR)
print(f"Static Head (Maximum Elevation): {static_head} m")

# Pressure wave celerity
a = np.sqrt((K / rho) / (1 + K * D / (E * esp)))
Courant = dt * a / dx
print(f"Wave speed: {a} m/s")
print(f"Courant number: {Courant}")

# Stability enforcement
if Courant > 1:
    print("Stability criterion violated! Adjusting dt...")
    dt = dx / a

# **User selection for inlet boundary condition**
print("\nSelect the inlet boundary condition:")
print("1 - Reservoir")
print("2 - Centrifugal Pump")
print("3 - Reciprocating Pump")
inlet_choice = int(input("Enter choice (1, 2, or 3): "))

pump_mode = None
if inlet_choice in [2, 3]:  # If user selects a pump
    print("\nSelect pump operation mode:")
    print("1 - Valve Closure Accidentally")
    print("2 - Start-up")
    print("3 - Shutdown")
    print("4 - Trip Condition")
    pump_mode = int(input("Enter choice (1, 2, 3, or 4): "))

# **User selection for outlet boundary condition**
print("\nSelect the outlet boundary condition:")
print("1 - Normal Operation (Constant Flow)")
print("2 - Valve Closure")
print("3 - Valve Opening")
outlet_choice = int(input("Enter choice (1, 2, or 3): "))

# Initialize variables
Q = np.zeros((Nx, 2))
H = np.zeros((Nx, 2))
Cmais = np.zeros(Nx)
Cmenos = np.zeros(Nx)

# Initialize pressure envelope tracking
MaxH = np.full(Nx, -np.inf)
MinH = np.full(Nx, np.inf)

# **Set Initial Condition Based on User Selection**
if inlet_choice in [2, 3] and pump_mode == 2:  # Pump Start-Up Mode
    H[:, 0] = static_head  # Set pipeline to static head
    Q[:, 0] = 0  # No initial flow (outlet closed)
    print("Pump Start-Up Mode Selected: Initial Condition Set to Static Head.")
else:
    # Regular operation (steady-state flow)
    zr = 1954  # Water level elevation
    Q0 = 245 / 3600  # Initial flow rate
    V0 = Q0 / A
    H[:, 0] = zr - (f * np.linspace(0, L, Nx) / Dint) * (V0**2 / (2 * g))
    Q[:, 0] = Q0
    print("Regular Operation Selected: Initial Condition Set to Steady-State Flow.")

# Pump parameters
pump_ramp_time = 60  # Time for pump to reach full speed
pump_max_flow = 240 / 3600  # Maximum flow rate (m³/s)

# Simulation parameters
B = a / (g * A)
R = f * dx / (2 * g * Dint * A**2)

# Transient Simulation
for n in range(1, Nt):
    prev, curr = (n - 1) % 2, n % 2

    # Compute C+ and C-
    Cmenos[1:Nx-1] = H[2:Nx, prev] - B * Q[2:Nx, prev] + R * Q[2:Nx, prev] * abs(Q[2:Nx, prev])
    Cmais[1:Nx-1] = H[0:Nx-2, prev] + B * Q[0:Nx-2, prev] - R * Q[0:Nx-2, prev] * abs(Q[0:Nx-2, prev])

    # Compute interior nodes
    H[1:Nx-1, curr] = 0.5 * (Cmais[1:Nx-1] + Cmenos[1:Nx-1])
    Q[1:Nx-1, curr] = (H[1:Nx-1, curr] - Cmenos[1:Nx-1]) / B

    # Track max & min pressures
    MaxH = np.maximum(MaxH, H[:, curr])
    MinH = np.minimum(MinH, H[:, curr])

    # **Pump Start-Up Sequence**
    if inlet_choice in [2, 3] and pump_mode == 2:  # Only applies for start-up mode
        if n * dt <= pump_ramp_time:
            Q[0, curr] = pump_max_flow * (n * dt / pump_ramp_time)  # Ramp up flow
        else:
            Q[0, curr] = pump_max_flow  # Pump reaches full operation
        H[0, curr] = Q[0, curr] * B + Cmenos[1]

    # **Outlet Valve Gradual Opening**
    if outlet_choice == 3 and n * dt > pump_ramp_time:  # Valve Opening
        CdA = 0.009 * ((n * dt - pump_ramp_time) / tav)**1.5 if (n * dt - pump_ramp_time) < tav else 0.009
        Q[Nx-1, curr] = np.sqrt((B**2) * (CdA**2) + 2 * CdA * Cmais[Nx-2]) - CdA * B

    # **Valve Closure Condition**
    elif outlet_choice == 2:
        CdA = 0.009 * (1 - (n * dt / tav))**1.5 if n * dt < tav else 0
        Q[Nx-1, curr] = np.sqrt((B**2) * (CdA**2) + 2 * CdA * Cmais[Nx-2]) - CdA * B

# Final Plots
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, L, Nx), MaxH, label='Maximum Pressure', color='red')
plt.plot(np.linspace(0, L, Nx), MinH, label='Minimum Pressure', color='blue')
plt.xlabel('Position (m)')
plt.ylabel('Pressure Head (m)')
plt.title('Pump Start-Up Transient')
plt.legend()
plt.grid(True)
plt.show()
