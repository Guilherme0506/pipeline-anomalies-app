import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import cv2

# Set the path for FFMpeg
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

# Parameters
L = 123000            # Tube length
Nx = 601           # Number of nodes in the tube
Nt = 1081          # Number of time steps
dx = L / (Nx - 1)
tf = 120             # Total simulation time
dt = tf / (Nt - 1)
tav = 30            # Valve opening time
Io = 0.0           # Slope of the tube
g = 9.81           # Gravity acceleration
E = 200e9          # Elastic modulus of the tube material
K = 1.75e9         # Fluid elastic modulus
D = 9.625*25.4/1000           # Tube diameter
esp = 0.344*25.4/1000         # Wall thickness
A = np.pi * 0.25 * D**2  # Cross-sectional area
f = 0.2          # Darcy-Weisbach resistance factor
rho = 1700         # Liquid density
k = 1              # Coefficient depending on the relationship

# Pressure wave celerity calculation
a = np.sqrt((K / rho) / (1 + K * D * k / (E * esp)))
dxa = dx / a
Courant = dt * a / dx

# Check stability criterion
if dt > dxa:
    print('Stability criterion violated')

# Initialize arrays for pressure and flow
x = np.zeros(Nx)
Q = np.zeros((Nx, Nt+1))
H = np.zeros((Nx, Nt+1))

# Initial conditions - Steady state
zr = 2000                                      # Water level elevation
CdA0 = 0.009                                   # Product of discharge coefficient and area at t=0
Q[0,:] = 240 /3600                                  # Flow rate at t=0
H[0,0]=zr

# Calculate steady-state piezometric line
for i in range(1,Nx):
    x[i] = (i-1) * dx
    Q[i, 0] = 240/3600
    H[i, 0] = zr - 0.012*x[i-1]

# Plot initial steady-state pressure head
plt.figure()
plt.plot(x, H[:, 0])
plt.xlabel('Position (m)')
plt.ylabel('Pressure Head (m)')
plt.title('Initial Pressure Head Distribution')
plt.grid(True)
plt.show()

n=1
T = 0
H[0, :] = zr  # H at x=1 for all instances

# Hydraulic transient calculation
for t in np.arange(0, tf, dt):
    T += dt
    n += 1

    # Boundary conditions:
    # Reservoir output
    Cmenos = Q[1, n-2] / A + g * (Io - f * Q[1, n-2] * abs(Q[1, n-2]) / (2 * g * D * A**2)) * dt - (g / a) * H[1, n-2]
    Q[0, n-1] = A * (Cmenos + g * H[0,n-1] / a)

    # Pipe output
    if T <= 2:
        #CdA = CdA0 * (1 - T / 2)**1.5
        CdA=0
    else:
        CdA=0
    #if T > 30 and T<=tf:
    #    CdA = CdA0 * (1- (( 30-20 ) / 20))**1.5
    

    Cmais = Q[Nx-2, n-2] / A + g * (Io - f * Q[Nx-2, n-2] * abs(Q[Nx-2, n-2]) / (2 * g * D * A**2)) * dt + (g / a) * H[Nx-2, n-2]
    Q[Nx-1, n-1] = CdA * (np.sqrt((a**2) * (CdA**2) / (A**2) + 2 * a * Cmais) - a * CdA / A)
    H[Nx-1, n-1] = (a / g) * (Cmais - Q[Nx-1, n-1] / A)

    # Internal nodes calculation
    for i in range(1, Nx-2):
        Q[i, n] = 0.5 * (Q[i-1, n-1] + Q[i+1, n-1]) + 0.5 * dt * g * A * (Io - f * Q[i-1, n-1] * abs(Q[i-1, n-1]) / (2 * g * D * A**2) + Io - f * Q[i+1, n-1] * abs(Q[i+1, n-1]) / (2 * g * D * A**2)) + (g * A / (2 * a)) * (H[i-1, n-1] - H[i+1, n-1])
        
        H[i, n] = 0.5 * (H[i-1, n-1] + H[i+1, n-1]) + 0.5 * a * dt * (Io - f * Q[i-1, n-1] * abs(Q[i-1, n-1]) / (2 * g * D * A)) - Io + f * Q[i+1, n-1] * abs(Q[i+1, n-1]) / (2 * g * D * A) + (a / (2 * g * A)) * (Q[i-1, n-1] - Q[i+1, n-1])

  
# Results visualization

b=len(H[Nx-1,:])
print(b)
c=np.linspace(0,tf+2*dt,int((tf+2*dt)/dt))
k=len(c)
print(k)
plt.figure(figsize=(12, 10))

# Pressure head at the pipe outlet over time
plt.subplot(2, 2, 1)
plt.plot(c, H[Nx-1, :])
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('H(L,t) [m] - at x = L')

# Pressure head and flow rate normalized
for n in range(0, Nt, 500):
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(0, L+dx, dx), H[:, n] / np.max(H), label=f'H/Hmax (t={n*dt:.2f}s)')
    plt.plot(np.arange(0, L+dx,dx), Q[:, n] / np.max(Q), label=f'Q/Qmax (t={n*dt:.2f}s)')
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
    for n in range(0, Nt, 5):
        plt.clf()  # Clear the current figure
        
        # Create x-axis points
        x = np.arange(0, L+dx, dx)
        
        # Plot the data
        plt.plot(x, H[:, n-1])
        plt.xlabel('x [m]')
        plt.ylabel('H [mca]')
        plt.grid(True)
        plt.xlim([0, L])
        plt.ylim([np.min(H), np.max(H)])
        
        # Capture and write frame
        writer.grab_frame()

plt.close()