import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import cv2

# Set the path for FFMpeg
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

# Parameters
L = 123000            # Tube length
Nx = 60001           # Number of nodes in the tube
Nt = 28001          # Number of time steps
dx = L / (Nx - 1)
tf = 60             # Total simulation time
dt = tf / (Nt - 1)
tav = 30            # Valve opening time
Io = 0.0            # Slope of the tube
g = 9.81            # Gravity acceleration
E = 200e9           # Elastic modulus of the tube material
K = 1.75e9          # Fluid elastic modulus
D = 9.625 * 25.4 / 1000          # Tube diameter
esp = 0.344 * 25.4 / 1000        # Wall thickness
A = np.pi * 0.25 * D**2  # Cross-sectional area
f = 0.02            # Darcy-Weisbach resistance factor
rho = 1700          # Liquid density
k = 1               # Coefficient depending on the relationship

# Pressure wave celerity calculation
a = np.sqrt((K / rho) / (1 + K * D * k / (E * esp)))
dxa = dx / a
Courant = dt * a / dx

# Check stability criterion
if dt > dxa:
    print('Stability criterion violated')

# Initialize arrays for pressure and flow
x = np.linspace(0, L, Nx)
Q = np.zeros((Nx, Nt))
H = np.zeros((Nx, Nt))

# Initial conditions - Steady state
zr = 2000            # Water level elevation
CdA0 = 0.009         # Product of discharge coefficient and area at t=0
Q0 = 240/3600 #np.sqrt(zr * 2 * g / (f * L / (D * A**2) + 1 / (CdA0**2)))  # Flow rate at t=0
#H0 = Q0**2 / ((CdA0**2) * 2 * g)  # H(x=L,t=0)

# Calculate steady-state piezometric line
for i in range(Nx):
    Q[i, 0] = Q0
    H[i, 0] = zr - 0.012 * x[i]

# Plot initial steady-state pressure head
plt.figure()
plt.plot(x, H[:, 0])
plt.xlabel('Position (m)')
plt.ylabel('Pressure Head (m)')
plt.title('Initial Pressure Head Distribution')
plt.grid(True)
plt.show()

n = 0
T = 0
H[0, :] = zr  # H at x=1 for all instances

# Hydraulic transient calculation
for t in np.arange(dt, tf + dt, dt):
    n += 1
    T += dt

    # Boundary conditions:
    # Reservoir output
    Cmenos = Q[1, n-1] / A + g * (Io - f * Q[1, n-1] * abs(Q[1, n-1]) / (2 * g * D * A**2)) * dt - (g / a) * H[1, n-1]
    Q[0, n] = A * (Cmenos + g * zr / a)

    # Pipe output
    if T <= 20:
        CdA = CdA0 * (1 - T / tav)**1.5
    elif T > 20 and T <= tf:
        CdA = CdA0 * (1 - ((T - 20) / 10))**1.5

    Cmais = Q[Nx-2, n-1] / A + g * (Io - f * Q[Nx-2, n-1] * abs(Q[Nx-2, n-1]) / (2 * g * D * A**2)) * dt + (g / a) * H[Nx-2, n-1]
    Q[Nx-1, n] = CdA * (np.sqrt((a**2) * (CdA**2) / (A**2) + 2 * a * Cmais) - a * CdA / A)
    H[Nx-1, n] = (a / g) * (Cmais - Q[Nx-1, n] / A)

    # Internal nodes calculation
    for i in range(1, Nx-1):
        Q[i, n] = 0.5 * (Q[i-1, n-1] + Q[i+1, n-1]) + 0.5 * dt * g * A * (
            Io - f * Q[i-1, n-1] * abs(Q[i-1, n-1]) / (2 * g * D * A**2) + Io - f * Q[i+1, n-1] * abs(Q[i+1, n-1]) / (2 * g * D * A**2)
        ) + (g * A / (2 * a)) * (H[i-1, n-1] - H[i+1, n-1])
        
        H[i, n] = 0.5 * (H[i-1, n-1] + H[i+1, n-1]) + 0.5 * a * dt * (
            Io - f * Q[i-1, n-1] * abs(Q[i-1, n-1]) / (2 * g * D * A) - Io + f * Q[i+1, n-1] * abs(Q[i+1, n-1]) / (2 * g * D * A)
        ) + (a / (2 * g * A)) * (Q[i-1, n-1] - Q[i+1, n-1])

# Results visualization
plt.figure(figsize=(12, 10))

# Pressure head at the pipe outlet over time
plt.subplot(2, 2, 1)
plt.plot(np.linspace(0, tf, Nt), H[Nx-1, :])
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

time_steps_to_plot = list(range(0, Nt, 500))

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
    for n in range(0, Nt, 50):
        plt.clf()  # Clear the current figure
        
        # Plot the data
        plt.plot(x, H[:, n])
        plt.xlabel('x [m]')
        plt.ylabel('H [mca]')
        plt.grid(True)
        plt.xlim([0, L])

        # Capture and write frame
        writer.grab_frame()

plt.close()


