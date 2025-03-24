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
Io = 0.0                 # Slope of the tube
g = 9.81                 # Gravity acceleration
E = 200e9                # Elastic modulus of the tube material
K = 1.75e9               # Fluid elastic modulus
D = (9.625*25.4)/1000    # Pipe External Diameter
esp = 0.384*25.4 / 1000  # Wall thickness
Dint=D-2*esp
A = np.pi * 0.25 * Dint**2   # Cross-sectional area
f = 0.0175               # Darcy-Weisbach resistance factor
rho = 1700               # Liquid density
k = 1                    # Coefficient depending on the relationship

# Set the path for FFMpeg
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

RP=input("Informe o Caminho do Arquivo Contendo o Perfil do Rejeitoduto / Mineroduto:")
df=pd.read_excel(RP)

LR=df['L'].values
HR=df['H'].values

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
zr = 1954       # Water level elevation
CdA0 = 0.009    # Product of discharge coefficient and area at t=0
#Q0 = np.sqrt(zr * 2 * g / (f * L / (D * A**2) + 1 / (CdA0**2)))  # Flow rate at t=0
Q0=245/3600
V0=Q0/A
print(V0)

H[0, :] = zr    # H at x=1 for all instances
# Calculate steady-state piezometric line
for i in range(Nx):
    x[i] = i * dx
    Q[i, :] = Q0
    H[i, :] = zr - (f * x[i] / (Dint)) * (V0**2 / (2 * g))


# Plot initial steady-state pressure head
plt.figure()
plt.plot(x, H[:, 0],label=f'Perfil do Terreno')
plt.plot(LR, HR, label=f'Perfil do Terreno')  
plt.xlabel('Position (m)')
plt.ylabel('Pressure Head (m)')
plt.title('Initial Pressure Head Distribution')
plt.grid(True)
plt.show()

T = 0
n = 0
B = a / (g * A)
R = f * dx / (2 * g * Dint * A**2)

rupture_disk_pressure = 1268
Ta=0
rupture_activated = False
pump_frequency=400*2*np.pi/60
omega=2*np.pi*pump_frequency
Q_mean=240/3600
Amplitude=20

# Hydraulic transient calculation
for t in np.arange(dt, tf, dt):
    n += 1
    T += dt
    
    for i in range(1, Nx-1):  # Changed to Nx-1 to avoid boundary issues
        Cmenos[i, n] = H[i + 1, n-1] - B * Q[i + 1, n-1] + R * Q[i + 1, n-1] * abs(Q[i + 1, n-1])
        Cmais[i, n] = H[i - 1, n-1] + B * Q[i - 1, n-1] - R * Q[i - 1, n-1] * abs(Q[i - 1, n-1])
        H[i, n] = 0.5 * (Cmais[i, n] + Cmenos[i, n])
        Q[i, n] = (H[i, n] - Cmenos[i, n]) / B

    # Outlet boundary condition
    if H[Nx-2,n]>=rupture_disk_pressure:
        rupture_activated = True
        
    if not rupture_activated:
        if T < 30 and H[Nx-2, n] < rupture_disk_pressure: 
            Q[Nx-1, n] = 240 / 3600
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

        elif 30 <= T <= 60 and H[Nx-2, n] < rupture_disk_pressure: 
        
            Cv = (0.5 * 240 / 3600) ** 2 / (2 * 536.75)
            CdA = Cv * (1 - (T - 30) / tav) ** 1.5
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

        elif T > 60 and H[Nx-2, n] < rupture_disk_pressure: 
       
            Cv = 0
            CdA = 0
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

    else:
        
        H[Nx-1, n] = 536.75
        Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B
        #Q[Nx-1, n] = 240 / 3600
        #H[Nx-1, n] = 700

    #Reciprocating Pump Boundary Condition  
    #theta=omega*T
    #Q[0, n] = 240/3600 # Constant Flow
    #Q[0,n]=Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3))) # Reciprocating Pump
    #H[0, n] = Q[0, n]*B + Cmenos[1,n] 

    # Reservor Boundary Condition
    #H[0, n] = 1950
    #Q[0, n] = (H[0, n] - Cmenos[1, n]) / B

    # Centrifugian Pump Boundary Condition (Regular Condition)
    Qp = np.array([0, 250/3600, 300/3600]) 
    Hp = np.array([2500, 1900, 1700]) 
    PC_coeficients = np.polyfit(Qp, Hp, 2)
    PC = np.poly1d(PC_coeficients)
    Q[0,n] = (1/(2*PC_coeficients[0]))*(B - PC_coeficients[1] - np.sqrt((B-PC_coeficients[1])**2 + 4*PC_coeficients[0]*(Cmenos[1,n] - PC_coeficients[2])))
    H[0, n] = Q[0, n]*B + Cmenos[1,n] 

    # Centrifugal Pump Boundaray Conition (Start-up)


    


    # Centrifugal Pump Boundaray Conidtion (Stoppage)






plt.figure(figsize=(12, 10))

# Pressure head at the pipe outlet over time
u=np.linspace(0, tf, Nt-1)

plt.subplot(2, 2, 1)
plt.plot(u*dt, H[Nx-1, 0:Nt-1])
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
plt.plot(x, MaxH,label=f'Máxima Pressão de Transiente' )
plt.plot(x,H[:,0], label=f'Linha Piezométrica (Pressão de Operação)')
plt.plot(LR, HR, label=f'Perfil do Terreno') 
plt.xlabel('Position (x)')
plt.ylabel('max(H(t))')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, MinH)
plt.plot(LR, HR, label=f'Perfil do Terreno') 
plt.xlabel('Position (x)')
plt.ylabel('min(H(t))')
plt.grid(True)

plt.show()

time_steps_to_plot = list(range(0, Nt, 100))  # Adjusted interval for better visualization

plt.figure(figsize=(12, 6))
for t in time_steps_to_plot:
    plt.plot(x, H[:, t], label=f'Time step {t*dt}')

plt.plot(LR, HR, label=f'Perfil do Terreno')     
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
    for n in range(0, Nt-1, 100):  # Adjusted interval for better visualization
        plt.clf()  # Clear the current figure
        
        # Plot the data
        plt.plot(x, H[0:Nx, n], label=f'Pressão em Regime Transiente (Fechamento Indevido Válvula)')
        plt.plot(LR, HR, label=f'Perfil do Terreno') 
        plt.plot(x,H[:,0], label=f'Linha Piezométrica (Pressão de Operação)')
        plt.xlabel('x [m]')
        plt.ylabel('H [mca]')
        plt.grid(True)
        plt.xlim([0, L])
        plt.ylim([0, np.nanmax(H)])  # Use nanmin and nanmax to avoid NaN/Inf issues
        
        # Capture and write frame
        writer.grab_frame()

plt.close()