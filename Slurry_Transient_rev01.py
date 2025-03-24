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
tav = 30                 # Valve closing/opening time
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
    print("1 - Regular Operation")
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

# Initial steady-state conditions
zr = 1954  # Water level elevation
Q0 = 245 / 3600  # Initial flow rate
V0 = Q0 / A
x = np.linspace(0, L, Nx)

# Compute initial steady-state H profile
H[:, 0] = zr - (f * x / Dint) * (V0**2 / (2 * g))
Q[:, 0] = Q0

# Pump parameters
pump_frequency = 400 * 2 * np.pi / 60
omega = 2 * np.pi * pump_frequency
Q_mean = 240 / 3600
Amplitude = 20

# Centrifugal pump characteristic curve (quadratic fit)
Qp = np.array([0, 250/3600, 300/3600])  # Flow (m³/s)
Hp = np.array([2500, 1900, 1700])  # Head (m)
PC_coef = np.polyfit(Qp, Hp, 2)
PC = np.poly1d(PC_coef)

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

    # **Inlet Boundary Conditions**
    if inlet_choice == 1:  # Reservoir
        H[0, curr] = 1950
        Q[0, curr] = (H[0, curr] - Cmenos[1]) / B
    elif inlet_choice == 2:  # Centrifugal Pump
        Q[0, curr] = (1 / (2 * PC_coef[0])) * (
            B - PC_coef[1] - np.sqrt((B - PC_coef[1]) ** 2 + 4 * PC_coef[0] * (Cmenos[1] - PC_coef[2]))
        )
        H[0, curr] = Q[0, curr] * B + Cmenos[1]
    elif inlet_choice == 3:  # Reciprocating Pump
        theta = omega * (n * dt)
        Q[0, curr] = Q_mean * (1 + (Amplitude / 3) * (np.sin(theta) + np.sin(theta + 2 * np.pi / 3) + np.sin(theta + 4 * np.pi / 3)))
        H[0, curr] = Q[0, curr] * B + Cmenos[1]

    # **Outlet Boundary Conditions**
    if outlet_choice == 1:  # Normal Operation
        Q[Nx-1, curr] = 240 / 3600
    elif outlet_choice == 2:  # Valve Closure
        CdA = 0.009 * (1 - (n * dt / tav))**1.5 if n * dt < tav else 0
        Q[Nx-1, curr] = np.sqrt((B**2) * (CdA**2) + 2 * CdA * Cmais[Nx-2]) - CdA * B
    elif outlet_choice == 3:  # Valve Opening
        CdA = 0.009 * (n * dt / tav)**1.5 if n * dt < tav else 0.009
        Q[Nx-1, curr] = np.sqrt((B**2) * (CdA**2) + 2 * CdA * Cmais[Nx-2]) - CdA * B

# Now **both inlet and outlet BCs work independently!** 🚀


# Final Plots
plt.figure(figsize=(12, 6))
plt.plot(x, MaxH, label='Maximum Pressure', color='red')
plt.plot(x, MinH, label='Minimum Pressure', color='blue')
plt.plot(x, H[:, 0], label='Initial Pressure', linestyle='--', color='black')
plt.plot(LR, HR, label='Pipeline Profile', linestyle='dotted', color='green')

plt.xlabel('Position (m)')
plt.ylabel('Pressure Head (m)')
plt.title('Pressure Envelope')
plt.legend()
plt.grid(True)
plt.show()
