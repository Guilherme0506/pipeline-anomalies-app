import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FFMpegWriter

# Parameters
L = 123000              # Tube length
Nx = 2000               # Number of nodes in the tube
Nt = 16000               # Number of time steps
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

# **User selection for inlet boundary condition**
print("\nSelect the inlet boundary condition:")
print("1  - Reservoir at Inlet")
print("2  - Centrifugal Pump at Inlet with Regular Operation")
print("3  - Centrifugal Pump Startup with Outlet Valve Openning")
print("4  - Centrifugal Pump Regular Shutdown with Outlet Valve Closure")
print("5  - Reciprocating Pump at inlet with Regular Operation")
print("6  - Reciprocating Pump Startup with Outlet Valve Openning")
print("7  - Reciprocating Pump Shutdown [Frequency Inversor] with Outlet Valve Closure")
print("8  - Centrifugal Pump Trip [Energy off] with Outlet Valve Closure")
print("9  - Reciprocating Pump Trip [Energy off] with Outlet Valve Closure")
print("10 - Centrifugal Pump with Outlet Valve Closure [Accidentaly]")
print("11 - Reciprocating with Outlet Valve Closure [Accidentaly]")
inlet_choice = int(input("Enter choice (1, 2, 3, 4, 5, 6, 7, 8, 9, 10 or 11): "))

# Initialize arrays for pressure and flow
x = np.linspace(0, L, Nx)
Q = np.zeros((Nx, Nt))
H = np.zeros((Nx, Nt))
Cmais = np.zeros((Nx, Nt))
Cmenos = np.zeros((Nx, Nt))

# Initial conditions - Steady state
CdA0 = 0.009    # Product of discharge coefficient and area at t=0
#Q0 = np.sqrt(zr * 2 * g / (f * L / (D * A**2) + 1 / (CdA0**2)))  # Flow rate at t=0

static_head = np.max(HR)
print(f"Static Head (Maximum Elevation): {static_head} m")


# **Set Initial Condition Based on User Selection**
if inlet_choice in [3, 6]: 
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

#Dados da Bomba Centrifuga
Qp = np.array([0, 250/3600, 300/3600]) 
Hp = np.array([2500, 1900, 1700]) 
PC_coeficients = np.polyfit(Qp, Hp, 2)
PC = np.poly1d(PC_coeficients)

Hs=PC[0]
a1=PC[1]
a2=PC[2]

  # Inlet Boundary Condition
TR=None

## Dados para Calculo do Trip

alfa00 = 1
alfa0  = 1
v0  = 1
v00 = 1
v=2*v0-v00
alfa=2*alfa0-alfa00
beta0=1
h=2000
HR=h
QR=240
HPM=50
BSQ=100
Delta_H=10
tau=1
BS=10
C31=500


WH=([ 0.634,  0.643,  0.646,  0.640,  0.629,  0.613,  0.595,  0.575, 0.552, 0.533, 0.516, 0.505, 0.504, 0.510, 0.512, 0.522, 0.539, 0.559, 0.580, 0.601, 0.630, 0.662, 0.692, 0.722, 0.753, 0.782, 0.808, 0.832, 0.857, 0.879, 0.904, 0.930, 0.959, 0.996, 1.027, 1.060, 1.090, 1.124, 1.165, 1.204, 1.238, 1.258, 1.271, 1.282, 1.288, 1.281, 1.260, 1.225, 1.121, 1.107, 1.031, 0.942, 0.842, 0.733, 0.617, 0.500, 0.368, 0.240, 0.125, 0.011, -0.102, -0.168, -0.255, -0.342, -0.423, -0.434, -0.556, -0.620, -0.655, -0.670, -0.670, -0.660, -0.655, -0.640, -0.600, -0.570, -0.520, -0.470, -0.430, -0.360, -0.275, -0.160, -0.040,  0.130,  0.295,  0.430,  0.550,  0.620,  0.634])
WB=([-0.684, -0.547, -0.414, -0.292, -0.187, -0.105, -0.053, -0.012, 0.042, 0.097, 0.156, 0.227, 0.300, 0.371, 0.444, 0.522, 0.596, 0.672, 0.738, 0.763, 0.797, 0.837, 0.865, 0.883, 0.886, 0.877, 0.859, 0.838, 0.804, 0.756, 0.703, 0.645, 0.583, 0.520, 0.454, 0.408, 0.370, 0.343, 0.331, 0.329, 0.338, 0.354, 0.372, 0.405, 0.450, 0.486, 0.520, 0.552, 0.579, 0.603, 0.616, 0.617, 0.606, 0.582, 0.546, 0.500, 0.432, 0.360, 0.288, 0.214,  0.123,  0.037, -0.053, -0.161, -0.248, -0.314, -0.372, -0.580, -0.740, -0.880, -1.000, -1.120, -1.250, -1.370, -1.490, -1.590, -1.660, -1.690, -1.770, -1.650, -1.590, -1.520, -1.420, -1.320, -1.230, -1.100, -0.980, -0.820, -0.584])
x1 = np.linspace(0, 2 * np.pi, 89)

coefficients_WH= np.polyfit(x1, WH, 5)
coefficients_WB= np.polyfit(x1, WB, 5)
p_WH = np.poly1d(coefficients_WH)
p_WB = np.poly1d(coefficients_WB)
WH_fit = p_WH(x1) 
WB_fit = p_WB(x1)

dxw=2*np.pi/len(WH)

# Define the functions for F1 and F2
def F1(BSQ, HPM, HR, alfa, v, A0, A1, Delta_H, tau, BS):
    return HPM - BSQ*v + HR*(alfa**2 + v**2) * (A0 + A1 * (np.pi + np.arctan(v / alfa))) - Delta_H * (v * abs(v)) / tau**2

def F2(Q, H, alfa, v, B0, B1, C31, alfa00, alfa0):
    return (alfa**2 + v**2) * (B0 + B1 * (np.pi + np.arctan(v / alfa))) + C31 * (alfa00 - alfa0)

# Define the partial derivatives
def dF1_dv(BSQ, alfa, v, A0, A1, Delta_H, tau, BS, HR):
    return -BSQ + HR * (2 * v * (A0 + A1 * (np.pi + np.arctan(v / alfa))) + A1 * alfa) - 2 * Delta_H * (abs(v) / tau**2)

def dF1_dalfa(alfa, v, A0, A1, HR):
    return HR * (2 * alfa * (A0 + A1 * (np.pi + np.arctan(v / alfa))) - A1 * v)

def dF2_dv(alfa, v, B0, B1):
    return 2 * v * (B0 + B1 * (np.pi + np.arctan(v / alfa))) - B1 * alfa

def dF2_dalfa(alfa, v, B0, B1, C31):
    return 2 * alfa * (B0 + B1 * (np.pi + np.arctan(v / alfa))) - B1 * v + C31

iterations=[]
alfa_values =[]
v_values=[]

# Hydraulic transient calculation
for t in np.arange(dt, tf, dt):
    n += 1
    T += dt
    
    for i in range(1, Nx-1):  # Changed to Nx-1 to avoid boundary issues
        Cmenos[i, n] = H[i + 1, n-1] - B * Q[i + 1, n-1] + R * Q[i + 1, n-1] * abs(Q[i + 1, n-1])
        Cmais[i, n] = H[i - 1, n-1] + B * Q[i - 1, n-1] - R * Q[i - 1, n-1] * abs(Q[i - 1, n-1])
        H[i, n] = 0.5 * (Cmais[i, n] + Cmenos[i, n])
        Q[i, n] = (H[i, n] - Cmenos[i, n]) / B

    if inlet_choice == 8:
       
        # 1. Regular Operation (T ≤ 30s) 
       
        if T <= 30:
            eta = 1
            H[Nx-1, n] = 536.75
            Q[Nx-1, n] = (Cmais[Nx-2, n] - H[Nx-1, n]) / B  # Compute outlet flow

            # Inlet Boundary Condition - Regular Operation
            C = 4 * a2 * ((eta ** 2) * Hs - Cmenos[1, n]) / (B - eta * a1) ** 2
            C=min(C,1)
            Q[0, n] = ((B - eta * a1) / (2 * a2)) * (1 - (1 - C) ** 0.5)  # Prevents zero flow
            H[0, n] = Q[0, n] * B + Cmenos[1, n]

            print(f"T={T:.2f}: Regular Operation - H[Nx-1]={H[Nx-1, n]:.2f}, Q[Nx-1]={Q[Nx-1, n]:.4f}")

        # 2. Pump Trip Phase (30s < T ≤ 60s) 
     
        else:
            # Outlet Boundary
            H[Nx-1, n] = 536.75
            Q[Nx-1, n] = (Cmais[Nx-2, n] - H[Nx-1, n]) / B  # Compute outlet flow

            # Inlet Boundary Condition - Pump Shutdown
            print(f"Pump tripped at time step {t}, time = {T:.2f}s")

            for _ in range(20):
                tol = 1e-4  # Tolerance for convergence

                pump_tripped = True
                x2=np.pi+np.arctan2(v,alfa)

                I = min(max(int(x2 / dxw), 0), len(WH) - 2) 
                A1=(WH[I+1] - WH[I])/dxw
                B1=(WB[I+1] - WB[I])/dxw
                A0= WH[I+1] -   I*A1*dxw
                B0= WB[I+1] -   I*B1*dxw

                F1_val=F1(BSQ, HPM, HR, alfa, v, A0, A1, Delta_H, tau, BS)
                F2_val=F2(Q, H, alfa, v, B0, B1, C31, alfa00, alfa0)

                if abs(F1_val) < tol and abs(F2_val) < tol:
                    break
                    

                dF1_dv_val = dF1_dv(BSQ, alfa, v, A0, A1, Delta_H, tau, BS, HR)
                dF1_dalfa_val = dF1_dalfa(alfa, v, A0, A1, HR)
                dF2_dv_val = dF2_dv(alfa, v, B0, B1)
                dF2_dalfa_val = dF2_dalfa(alfa, v, B0, B1, C31)

                delta_alfa = (F2_val * dF1_dv_val - F1_val * dF2_dv_val) / (dF1_dalfa_val * dF2_dv_val - dF1_dv_val * dF2_dalfa_val)
                delta_v = -F1_val / dF1_dv_val - delta_alfa * (dF1_dalfa_val / dF1_dv_val)

                v += delta_v
                alfa +=delta_alfa
            print(v)
            print(alfa)

            v00=v0
            alfa00=alfa0
            v0=v
            alfa0=alfa
            Q[0,n]=v*Q[0,n-1]
            H[0,n]=H[0,n-1]*(alfa**2+v**2)


  
   

   
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
    for n in range(0, Nt-1, 10):  # Adjusted interval for better visualization
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