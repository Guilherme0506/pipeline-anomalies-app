import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# Parameters
L = 123000             # Tube length
Nx = 6001            # Number of nodes in the tube
Nt = 601*50 # Number of time steps
dx = L / (Nx - 1)
tf = 60 * 10        # Total simulation time
dt = tf / (Nt - 1)
tav = 30             # Valve opening time
Io = 0.0            # Slope of the tube
g = 9.81            # Gravity acceleration
E = 200e9           # Elastic modulus of the tube material
K = 1.75e9          # Fluid elastic modulus
D = 0.5             # Tube diameter
esp = 20 / 1000     # Wall thickness
A = np.pi * 0.25 * D**2  # Cross-sectional area
f = 0.018           # Darcy-Weisbach resistance factor
rho = 1700          # Liquid density
k = 1               # Coefficient depending on the relationship

# Set the path for FFMpeg
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

# Pressure wave celerity calculation
a = np.sqrt((K / rho) / (1 + K * D * k / (E * esp)))
dxa = dx / a
Courant = dt * a / dx

print(Courant)

# Stability criterion check
if dt > dxa:
    print('Critério de estabilidade violado')

# Initialize arrays for pressure and flow
x = np.linspace(0, L, Nx)
Q = np.zeros((Nx, Nt))
H = np.zeros((Nx, Nt))
Cmais = np.zeros((Nx, Nt))
Cmenos = np.zeros((Nx, Nt))

# Initial conditions - Steady state
zr = 150  # Water level elevation
CdA0 = 0.009  # Product of discharge coefficient and area at t=0
Q0 = np.sqrt(zr * 2 * g / (f * L / (D * A**2) + 1 / (CdA0**2)))  # Flow rate at t=0

# Calculate steady-state piezometric line
for i in range(Nx):
    x[i] = i * dx
    Q[i, 0] = Q0
    H[i, 0] = zr - (Q0**2 / (2 * g * A**2)) * (f * x[i] / D)

# Plot initial steady-state pressure head
plt.figure()
plt.plot(x, H[:, 0])
plt.xlabel('Position (m)')
plt.ylabel('Pressure Head (m)')
plt.title('Initial Pressure Head Distribution')
plt.grid(True)
plt.show()

T = 0
H[0, :] = zr  # H at x=1 for all instances
n = 0
B = a / (g * A)
R = f * dx / (2 * g * D * A**2)

# Hydraulic transient calculation
for t in np.arange(dt, tf + dt, dt):
    n += 1
    T += dt

    for i in range(1, Nx-1):  # Changed to Nx-1 to avoid boundary issues
        Cmenos[i, n] = H[i + 1, n-1] - B * Q[i + 1, n-1] + R * Q[i + 1, n-1] * abs(Q[i + 1, n-1])
        Cmais[i, n] = H[i - 1, n-1] + B * Q[i - 1, n-1] - R * Q[i - 1, n-1] * abs(Q[i - 1, n-1])
        H[i, n] = 0.5 * (Cmais[i, n] + Cmenos[i, n])
        Q[i, n] = (H[i, n] - Cmenos[i, n]) / B

    # Valve boundary condition
        
        if T <= tav:
           CdA = CdA0 * (1 - T / tav)**1.5
        
        if T> tav and T<=60:
           CdA = 0

        if T > 60 and T<= 60+tav:
            CdA=0.009*(1-(1-T/30))**1.5

        Q[Nx-1, n] = -CdA * B + np.sqrt((B**2) * (CdA**2) + 2 * CdA * Cmais[Nx-2, n])
        H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

    # Reservoir boundary condition
    H[0, n] = zr
    Q[0, n] = (H[0, n] - Cmenos[1, n]) / B


plt.figure(figsize=(12, 10))

# Pressure head at the pipe outlet over time
plt.subplot(2, 2, 1)
plt.plot(np.linspace(0, tf, Nt), H[Nx-1, :])
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('H(L,t) [m] - at x = L')

# Pressure head and flow rate normalized
for n in range(0, Nt, 500):  # Adjusted interval for better visualization
    plt.subplot(2, 2, 2)
    plt.plot(x, H[:, n] / np.max(H), label=f'H/Hmax (t={n*dt:.2f}s)')
    plt.plot(x, Q[:, n] / np.max(Q), label=f'Q/Qmax (t={n*dt:.2f}s)')
    plt.xlabel('Position (x)')
    plt.ylabel('Normalized H and Q')
    plt.xlim([0, L])
    plt.ylim([-1, 1])
    plt.legend(loc='upper right')

# Envelope with maximum H values
MaxH = np.zeros(Nx)
MinH = np.zeros(Nx)

for i in range(Nx):
    MaxH[i] = np.max(H[i, :])
    MinH[i] = np.min(H[i, :])

plt.subplot(2, 2, 3)
plt.plot(x, MaxH)
plt.xlabel('Position (x)')
plt.ylabel('max(H(t))')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, MinH)
plt.xlabel('Position (x)')
plt.ylabel('min(H(t))')
plt.grid(True)

plt.show()

time_steps_to_plot = list(range(0, Nt, 100))  # Adjusted interval for better visualization

plt.figure(figsize=(12, 6))
for t in time_steps_to_plot:
    plt.plot(x, H[:, t], label=f'Time step {t}')

plt.xlabel('Position along the pipeline (m)')
plt.ylabel('Pressure head (m)')
plt.title('Pressure Head Distribution along the Pipeline')
plt.legend()
plt.grid(True)
plt.show()

# Animation
fig = plt.figure()


# Animation
fig = plt.figure()

# Create video writer object
metadata = dict(title='Movie', artist='codinglikemad')
writer = FFMpegWriter(fps=30, metadata=metadata)

# Open video file
with writer.saving(fig, 'Transiente_Hidraulico.mp4', 100):
    for n in range(0, Nt, 10):  # Adjusted interval for better visualization
        plt.clf()  # Clear the current figure
        
        # Plot the data
        plt.plot(x, H[:, n])
        plt.xlabel('x [m]')
        plt.ylabel('H [mca]')
        plt.grid(True)
        plt.xlim([0, L])
        plt.ylim([np.nanmin(H), np.nanmax(H)])  # Use nanmin and nanmax to avoid NaN/Inf issues
        
        # Capture and write frame
        writer.grab_frame()

plt.close()

