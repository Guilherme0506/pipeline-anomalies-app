import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# Parameters
L = 123000             # Tube length
Nx = 151*10            # Number of nodes in the tube
Nt = 61*50*9           # Number of time steps
dx = L / (Nx-1)
tf = 60 * 50        # Total simulation time
dt = tf / (Nt - 1)
tav = 30             # Valve opening time
Io = 0.0            # Slope of the tube
g = 9.81            # Gravity acceleration
E = 200e9           # Elastic modulus of the tube material
K = 1.75e9          # Fluid elastic modulus
D = 9.625*25.4/1000             # Tube diameter
esp = 0.412*25.4 / 1000     # Wall thickness
A = np.pi * 0.25 * D**2  # Cross-sectional area
f = 0.020           # Darcy-Weisbach resistance factor
rho = 1700          # Liquid density
k = 1               # Coefficient depending on the relationship

# Set the path for FFMpeg
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

# Pressure wave celerity calculation
a = np.sqrt((K / rho) / (1 + K * D * k / (E * esp)))
dxa = dx / a
Courant = dt * a / dx
print(a)
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
zr = 1950  # Water level elevation
CdA0 = 0.009  # Product of discharge coefficient and area at t=0
Q0 = 240 / 3600

H[0, :] = zr  # H at x=1 for all instances
# Calculate steady-state piezometric line
for i in range(Nx):
    x[i] = i * dx
    Q[i, :] = Q0
    H[i, :] = zr - (f * x[i] / D) * (1.66**2) / (2 * g)

# Plot initial steady-state pressure head
plt.figure()
plt.plot(x, H[:, 0])
plt.xlabel('Position (m)')
plt.ylabel('Pressure Head (m)')
plt.title('Initial Pressure Head Distribution')
plt.grid(True)
plt.show()

T = 0
n = 0
B = a / (g * A)
R = f * dx / (2 * g * D * A**2)

# Hydraulic transient calculation using Lax-Wendroff scheme
for t in np.arange(dt, tf, dt):
    n += 1
    T += dt

    for i in range(1, Nx-1):
        # Predictor step (half step in time)
        H_half = 0.5 * (H[i+1, n-1] + H[i-1, n-1]) - 0.5 * dt/dx * (Q[i+1, n-1] - Q[i-1, n-1])
        Q_half = 0.5 * (Q[i+1, n-1] + Q[i-1, n-1]) - 0.5 * dt/dx * (H[i+1, n-1] - H[i-1, n-1])

        # Corrector step (full step in time)
        H[i, n] = H[i, n-1] - dt/dx * (Q_half - Q[i-1, n-1])
        Q[i, n] = Q[i, n-1] - dt/dx * (H_half - H[i-1, n-1])

    # Valve boundary condition
    if T < 30:
        Q[Nx-1, n] = 240 / 3600
        H[Nx-1, n] = 536.75

    if T >= 30 and T <= 60:
        Cv = (0.5 * 240 / 3600)**2 / (2 * 536.75)
        CdA = Cv * (1 - (T - 30) / tav)**1.5
        Q[Nx-1, n] = -CdA * B + np.sqrt((B**2) * (CdA**2) + 2 * CdA * Cmais[Nx-2, n])
        H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]
    else:
        CdA = 0
        Q[Nx-1, n] = -CdA * B + np.sqrt((B**2) * (CdA**2) + 2 * CdA * Cmais[Nx-2, n])
        H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

    # Reservoir boundary condition
    H[0, n] = 1950
    Q[0, n] = (H[0, n] - Cmenos[1, n]) / B

plt.figure(figsize=(12, 10))

# Pressure head at the pipe outlet over time
u = np.linspace(0, tf, Nt-1)
plt.subplot(2, 2, 1)
plt.plot(u*dt, H[Nx-1, 0:Nt-1])
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('H(L,t) [m] - at x = L')

# Pressure head and flow rate normalized
for n in range(0, Nt, 500):
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

time_steps_to_plot = list(range(0, Nt, 100))

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

# Create video writer object
metadata = dict(title='Movie', artist='codinglikemad')
writer = FFMpegWriter(fps=30, metadata=metadata)

# Open video file
with writer.saving(fig, 'Transiente_Hidraulico.mp4', 100):
    for n in range(0, Nt - 1, 1):
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
