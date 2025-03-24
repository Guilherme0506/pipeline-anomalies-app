import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.ticker as ticker
import geopandas as gpd
from fastkml import kml
from shapely.geometry import LineString, Point
from xml.etree import ElementTree as ET
import contextily as ctx  # For adding a basemap
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import Normalize, ListedColormap
import re
from pyproj import Transformer


# Parameters
L = 123000              # Tube length
Nx = 500               # Number of nodes in the tube
Nt = 14500               # Number of time steps
dx = L / (Nx-1)
tf = 4000                # Total simulation time
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
print("10 - Centrifugal Pump with Outlet Valve Closure [Accidentaly] with Rupture Disk Activated")
print("11 - Reciprocating with Outlet Valve Closure [Accidentaly]")
print("12 - Centrifugal Pump with Outlet Valve Closure [Accidentaly]")
print("13 - Reservoir at Inlet with valve closure")
inlet_choice = int(input("Enter choice (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 or 13): "))

# Initialize arrays for pressure and flow
x = np.linspace(0, L, Nx)
Q = np.zeros((Nx, Nt))
H = np.zeros((Nx, Nt))
HB = np.zeros((1, Nt))
Cmais = np.zeros((Nx, Nt))
Cmenos = np.zeros((Nx, Nt))

# Initial conditions - Steady state
CdA0 = 0.009    # Product of discharge coefficient and area at t=0
#Q0 = np.sqrt(zr * 2 * g / (f * L / (D * A**2) + 1 / (CdA0**2)))  # Flow rate at t=0

static_head = np.nanmax(HR)
print(f"Static Head (Maximum Elevation): {static_head} m")


# **Set Initial Condition Based on User Selection**
if inlet_choice in [3, 6]: 
    H[:, 0] = static_head  # Set pipeline to static head
    Q[:, 0] = 0  # No initial flow (outlet closed)
    print("Pump Start-Up Mode Selected: Initial Condition Set to Static Head.")
else:
    # Regular operation (steady-state flow)
    zr = 1961.26 # Water level elevation
    Q0 = 245 / 3600  # Initial flow rate
    V0 = Q0 / A
    H[0:Nx, 0] = zr - (f * np.linspace(0, L, Nx) / Dint) * (V0**2 / (2 * g))
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
Q_mean=245/3600
Amplitude=20

#Dados da Bomba Centrifuga
Qp = np.array([0, 245/3600, 300/3600]) 
Hp = np.array([2100, 1961, 1900]) 
PC_coeficients = np.polyfit(Qp, Hp, 2)
PC = np.poly1d(PC_coeficients)

Hs=PC[0]
a1=PC[1]
a2=PC[2]

  # Inlet Boundary Condition
TR=None
Q0cv=None

# Hydraulic transient calculation
for t in np.arange(dt, tf, dt):
    n += 1
    T += dt
    
    for i in range(1, Nx-1):  # Changed to Nx-1 to avoid boundary issues
        Cmenos[i, n] = H[i + 1, n-1] - B * Q[i + 1, n-1] + R * Q[i + 1, n-1] * abs(Q[i + 1, n-1])
        Cmais[i, n] = H[i - 1, n-1] + B * Q[i - 1, n-1] - R * Q[i - 1, n-1] * abs(Q[i - 1, n-1])
        H[i, n] = 0.5 * (Cmais[i, n] + Cmenos[i, n])
        Q[i, n] = (H[i, n] - Cmenos[i, n]) / B

    # Reservor Boundary Condition
    if inlet_choice==1:
        # Inlet Boundary Condition
        H[0, n] = 1961.259
        Q[0, n] = (H[0, n] - Cmenos[1, n]) / B

        #Outlet Boundary Condition
        H[Nx-1, n] = 531.747
        Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B
   
    # Centrifugian Pump Boundary Condition (Regular Condition)
    if inlet_choice==2:

        # Inlet Boundary Condition
        Q[0,n] = (1/(2*a2))*(B - a1 - np.sqrt((B-a1)**2 + 4*a2*(Cmenos[1,n] - Hs)))
        H[0, n] = Q[0, n]*B + Cmenos[1,n] 

        #Outlet Boundary Conditoin
        H[Nx-1, n] = 531.747
        Q[Nx-1, n] = (Cmais[Nx-2, n] - H[Nx-1, n]) / B

      
    # Centrifugal Pump Boundary Condition (Start-up) with Valve Openning
    if inlet_choice == 3:

        # ---------------------- #
        # 1. Valve Closed Phase  #
        # ---------------------- #
        if T <= 60:
            # Outlet Boundary Condition - Valve Closed
            Q[Nx-1, n] = 0  
            H[Nx-1, n] = Cmais[Nx-2, n]  # No outflow, pressure buildup

            # Inlet Boundary Condition: Pump Start-Up with Gradual Ramp-Up
            eta = T / 60  # Gradual increase from 0 to 1
            ramp_head = Hs* (eta ** 2)  # Ensuring smooth transition

            if ramp_head <= static_head:
                Q[0, n] = 0
                H[0, n] = Q[0, n] * B + Cmenos[1, n]
            else:
                C = 4 * a2 * (ramp_head - Cmenos[1, n]) / (B - eta * a1) ** 2
                Q[0, n] = (((B - eta * a1) / (2 * a2)) * (1 - (1 - C) ** 0.5))  # Ensures nonzero flow
                H[0, n] = Q[0, n] * B + Cmenos[1, n]

            print(f"T={T:.2f}: Pump Start-Up - H[0]={H[0, n]:.2f}, Q[0]={Q[0, n]:.4f}")

        # ---------------------- #
        # 2. Valve Opening Phase #
        # ---------------------- #
        elif 60 < T <= 120:
            # Outlet Boundary Condition Gradual Valve Opening
            Cv = ( 245 / 3600) ** 2 / (2 * 531.747)  # Base coefficient
            CdA = (Cv * ((T - 60) / 60) ** 1.5)  # Smooth valve opening
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            # Inlet Boundary Condition - Pump in Normal Operation Mode
            eta = 1  # Fully operational
            C = 4 * a2 * ((eta ** 2) * Hs - Cmenos[1, n]) / (B - eta * a1) ** 2
            Q[0, n] = ((B - eta * a1) / (2 * a2)) * (1 - (1 - C) ** 0.5)  # Prevents zero flow
            H[0, n] = Q[0, n] * B + Cmenos[1, n]

            print(f"T={T:.2f}: Valve Opening - CdA={CdA:.6f}, H[Nx-1]={H[Nx-1, n]:.2f}, Q[Nx-1]={Q[Nx-1, n]:.4f}")

        # ---------------------- #
        # 3. Valve Fully Opened
        # ---------------------- #
        else:
            # Outlet Boundary Condition
            H[Nx-1, n] = 531.747
            Q[Nx-1, n]  = (Cmais[Nx-2, n] - H[Nx-1, n] ) / B

            # Inlet Boundary Condition
            eta = 1  
            C = 4 * a2 * ((eta ** 2) * Hs - Cmenos[1, n]) / (B - eta * a1) ** 2
            Q[0, n] = ((B - eta * a1) / (2 * a2)) * (1 - (1 - C) ** 0.5) # Ensures smooth transition
            H[0, n] = Q[0, n] * B + Cmenos[1, n]

            print(f"T={T:.2f}: Valve Fully Opened - H[Nx-1]={H[Nx-1, n]:.2f} (Atmospheric), Q[Nx-1]={Q[Nx-1, n]:.4f}")

    # Centrifugian Shutdown with outlet Valve Closure
    if inlet_choice == 4:
       
        # 1. Regular Operation (T ≤ 30s) 
        if T <= 30:
            
            H[Nx-1, n] = 531.747
            Q[Nx-1, n] = (Cmais[Nx-2, n] - H[Nx-1, n]) / B  # Compute outlet flow

            # Inlet Boundary Condition - Regular Operation
            eta=1
            ramp_head = Hs* (eta ** 2)  # Ensuring smooth transition
            C = (4 * a2 * (ramp_head - Cmenos[1, n])) / ((B - eta * a1) ** 2)
            Q[0, n] = max(0,((B - eta * a1) / (2 * a2)) * (1 - np.sqrt((1 - C))))  # Ensures smooth shutdown
            H[0, n] = Q[0, n]*B + Cmenos[1,n] 
            print(f"T={T:.2f}: Regular Operation - H[Nx-1]={H[Nx-1, n]:.2f}, Q[Nx-1]={Q[Nx-1, n]:.4f}")

        # 2. Pump Shutdown Phase (30s < T ≤ 60s) 
        elif 30 < T <= 60:
            # Outlet Boundary
            Cv = (245 /3600) ** 2 / (2 * 531.747)
            time_factor = (60 - T) / 30  # Ensure non-negative base
            CdA = Cv * (time_factor ** 1.5)  # Gradual closure
            Q[Nx-1, n] =max(0,-CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n]))
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]


            # Inlet Boundary Condition - Regular Operation
            eta=1
            ramp_head = Hs* (eta ** 2)  # Ensuring smooth transition
            C = (4 * a2 * (ramp_head - Cmenos[1, n])) / ((B - eta * a1) ** 2)
            Q[0, n] = max(0,((B - eta * a1) / (2 * a2)) * (1 - np.sqrt((1 - C))))  # Ensures smooth shutdown
            H[0, n] = Q[0, n]*B + Cmenos[1,n] 
            print(f"T={T:.2f}: Regular Operation - H[Nx-1]={H[Nx-1, n]:.2f}, Q[Nx-1]={Q[Nx-1, n]:.4f}")


        # 2. Pump Shutdown Phase (30s < T ≤ 60s) 
        elif 60 < T <= 100:
            # Outlet Boundary
            Cv = (245 /3600) ** 2 / (2 * 531.747)
            time_factor = 0  # Ensure non-negative base
            CdA = Cv * (time_factor ** 1.5)  # Gradual closure
            Q[Nx-1, n] =max(0,-CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n]))
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]


            # Inlet Boundary Condition - Pump Shutdown
            eta = 1  # Smooth pump ramp-down
            ramp_head = Hs* (eta ** 2)  # Ensuring smooth transition


            C = (4 * a2 * (ramp_head - Cmenos[1, n])) / ((B - eta * a1) ** 2)
            Q[0, n] = max(0,((B - eta * a1) / (2 * a2)) * (1 - np.sqrt((1 - C))))  # Ensures smooth shutdown
            H[0, n] = Q[0, n] * B + Cmenos[1, n]

            print(f"T={T:.2f}: PumpShutting Down, H[Nx-1]={H[Nx-1, n]:.2f}, Q[Nx-1]={Q[Nx-1, n]:.4f}")


        elif 100 < T <= 490:
            # Outlet Boundary
            Cv = 0
            time_factor = (490 - T) / 390  # Ensure non-negative base
            CdA = Cv * (time_factor ** 1.5)  # Gradual closure
            Q[Nx-1, n] =max(0,-CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n]))
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]


            # Inlet Boundary Condition - Pump Shutdown
            eta = (490 - T) / 390  # Smooth pump ramp-down
            ramp_head = Hs* (eta ** 2)  # Ensuring smooth transition
            C = (4 * a2 * (ramp_head - Cmenos[1, n])) / ((B - eta * a1) ** 2)
            Q[0, n] = max(0,((B - eta * a1) / (2 * a2)) * (1 - np.sqrt((1 - C))))  # Ensures smooth shutdown
            H[0, n] = Q[0, n] * B + Cmenos[1, n]

            print(f"T={T:.2f}: PumpShutting Down, H[Nx-1]={H[Nx-1, n]:.2f}, Q[Nx-1]={Q[Nx-1, n]:.4f}")

        #3. Set TR (First Time T > 60) 
        
        else:
            # Valve is completely closed
            CdA = 0
            Q[Nx-1, n] = 0
            H[Nx-1, n] = Cmais[Nx-2, n]  # Maintain last calculated head

            # Enforce Zero Flow at Inlet (Pump Off)
            Q[0, n] = 0
            #H[0, n] = static_head
            H[0, n]=Q[0, n]*B + Cmenos [1,n]
            #Q[0, n] =0

            print(f"T={T:.2f}: Valve Fully Closed - H[Nx-1]={H[Nx-1, n]:.2f}, Q[Nx-1]={Q[Nx-1, n]:.4f}")


    # Reciprocating Pump at inlet with Regular Operation
    if inlet_choice==5:
        # Inlet Boundary Condition
        theta=omega*T
        Q[0,n]=Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
        H[0, n] = Q[0, n] * B + Cmenos[1, n]

        #Outlet Boundary Condition
        H[Nx-1, n] = 531.747
        Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B

        
    # Reciprocating Pump Boundary Condition (Start-up) with Valve Openning
    if inlet_choice==6:  
        theta=omega*T
        if T <= 30:
            #Outlet Boundary Condition
            Cv = 0
            CdA = Cv * ((60-30)/30) ** 1.5
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]
            
            eta =0
            theta=omega*T
            Q[0,n]=eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
            H[0, n] = Q[0, n]*B + Cmenos[1,n] 

        elif 30 < T <= 60:
            # Outlet Boundary Condition
            Cv = (245/ 3600) ** 2 / (2 * 531.747)
            CdA = Cv * ((T-30)/30) ** 1.5
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]


            #Inlet Boundary Conditon
            eta =0
            theta=omega*T
            Q[0,n]=eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
            H[0, n] = Q[0, n]*B + Cmenos[1,n] 


        elif 60 < T <= 90:

            #Outlet
            Cv = ( 245/ 3600) ** 2 / (2 * 531.747)
            CdA = Cv * 1
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            # Inlet Boundary Condition
            eta =(T-60)/30
            theta=omega*T
            Q[0,n]=eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
            H[0, n] = Q[0, n]*B + Cmenos[1,n] 

        else:
            Cv = ( 245/ 3600) ** 2 / (2 * 531.747)
            CdA = Cv * 1
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]
            
            eta =1
            theta=omega*T
            Q[0,n]=eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
            H[0, n] = Q[0, n]*B + Cmenos[1,n] 

          
    # Reciprocating Pump Boundary Condition (Shutdown) with Valve Closure
    if inlet_choice==7:
        theta=omega*T
        if T <= 30:
            #Outlet Boundary Condition
            H[Nx-1, n] = 531.747
            Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B

            #Inlet Boundary Conditon
            Q[0,n] = Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
            H[0, n] = Q[0, n]*B + Cmenos[1,n]   

        elif 30 < T <= 60:
            # Outlet Boundary Condition - Valve Closure
            Cv = ( 245 / 3600) ** 2 / (2 * 531.747)
            CdA = Cv * (1)
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            # Inlet Boundary Condition
            eta =(60-T)/30
            Q[0,n]=max(0,eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3))))
            H[0, n] = max(Q[0, n]*B + Cmenos[1,n],static_head) 

        elif 60 < T <= 90:
            # Outlet Boundary Condition - Valve Closure
            Cv = ( 245 / 3600) ** 2 / (2 * 531.747)
            CdA = Cv * (1)
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            # Inlet Boundary Condition
            eta =0
            Q[0,n]=max(0,eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3))))
            H[0, n] = max(Q[0, n]*B + Cmenos[1,n],static_head) 

        elif 90 < T <= 120:
            # Outlet Boundary Condition - Valve Closure
            Cv = ( 245 / 3600) ** 2 / (2 * 531.747)
            CdA = Cv * (1)
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            # Inlet Boundary Condition
            eta =0
            Q[0,n]=max(0,eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3))))
            H[0, n] = max(Q[0, n]*B + Cmenos[1,n],static_head) 


        elif 120 < T <= 150:
            # Outlet Boundary Condition - Valve Closure
            Cv = ( 245 / 3600) ** 2 / (2 * 531.747)
            CdA = Cv * (1 - (T - 120) / tav) ** 1.5
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            # Inlet Boundary Condition
            eta =0
            Q[0,n]=max(0,eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3))))
            H[0, n] = max(Q[0, n]*B + Cmenos[1,n],static_head) 

        else:
            #Outlet Boundary Condition:
            Cv = 0
            CdA=Cv
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            #Inlet Boundary Condition
            Q[0,n] = 0
            H[0, n] = max(Q[0, n]*B + Cmenos[1,n],static_head)


# Reciprocating Pump Boundary Condition (Shutdown) with Valve Closure
    if inlet_choice==9:

        index = np.where(HR == static_head)[0]
        Ls=12000/123000
        Ns=int(Nx*Ls)
        theta=omega*T
        if T <= 30:
            #Outlet Boundary Condition
            H[Nx-1, n] = 531.747
            Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B

            #Inlet Boundary Conditon
            Q[0,n] = Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
            H[0, n] = Q[0, n]*B + Cmenos[1,n]   

        elif 30 < T <= 150:
            # Outlet Boundary Condition - Valve Closure
            Cv = ( 245 / 3600) ** 2 / (2 * 531.747)
            CdA =  Cv * ( 1 ) ** 1.5
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            # Inlet Boundary Condition
            if 30<T<=40:
                eta=(40-T)/10
                Q[0,n] = eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
                H[0, n] = Q[0, n]*B + Cmenos[1,n] 

            else:
                eta=0
                Q[0,n] = eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
                H[0, n] = Q[0, n]*B + Cmenos[1,n]


        elif 150 < T <= 180:
            # Outlet Boundary Condition - Valve Closure
            Cv = ( 245 / 3600) ** 2 / (2 * 531.747)
            CdA =  Cv * (( 180 -T )/30) ** 1.5
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            # Inlet Boundary Condition
            eta=0
            Q[0,n] = eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
            H[0, n] = Q[0, n]*B + Cmenos[1,n]  


        else:
            #Outlet Boundary Condition:
            Cv = 0
            CdA=Cv
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            #Inlet Boundary Condition
            eta=0
            Q[0,n] = eta*Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
            H[0, n] = Q[0, n]*B + Cmenos[1,n]

        H[0:Ns, :][H[0:Ns, :] <= static_head] = static_head
          
    
    # Centrifugal Pump with Outlet Valve Closure [Accidentaly]
    if inlet_choice==10:
        # Outlet Boundary Condition
        if H[Nx-2,n]>=rupture_disk_pressure:
            rupture_activated = True
        
        if not rupture_activated:
            
            if T < 30: 
                #Outlet Boundary Condition
                H[Nx-1, n] = 536.75
                Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B

                #Inlet Boundary Condition
                eta = 1  # Fully operational
                C = 4 * a2 * ((eta ** 2) * Hs - Cmenos[1, n]) / (B - eta * a1) ** 2
                Q[0, n] = (((B - eta * a1) / (2 * a2)) * (1 - (1 - C) ** 0.5))  # Prevents zero flow
                H[0, n] = Q[0, n] * B + Cmenos[1, n]


            elif 30 <= T <= 60:
                #Outlet Boundary Condition
                Cv = ( 240 / 3600) ** 2 / (2 * 536.75)
                CdA = Cv * (1 - (T - 30) / 30) ** 1.5
                Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
                H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

                #Inlet Boundary Condition
                eta = 1  # Fully operational
                C = 4 * a2 * ((eta ** 2) * Hs - Cmenos[1, n]) / (B - eta * a1) ** 2
                Q[0, n] = (((B - eta * a1) / (2 * a2)) * (1 - (1 - C) ** 0.5))  # Prevents zero flow
                H[0, n] = Q[0, n] * B + Cmenos[1, n]

            elif T > 60:
       
                Cv = 0
                CdA = 0
                Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
                H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

                #Inlet Boundary Condition
                eta = 1  # Fully operational
                C = 4 * a2 * ((eta ** 2) * Hs - Cmenos[1, n]) / (B - eta * a1) ** 2
              
                Q[0, n] = ((B - eta * a1) / (2 * a2)) * (1 - (1 - C) ** 0.5)  # Prevents zero flow
                H[0, n] = Q[0, n] * B + Cmenos[1, n]

        else:
            #Outlet Boundary Condition
            H[Nx-1, n] = 536.75
            Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B

            #Inlet Boundary Condition
            eta = 1  # Fully operational
            C = 4 * a2 * ((eta ** 2) * Hs - Cmenos[1, n]) / (B - eta * a1) ** 2
            Q[0, n] = ((B - eta * a1) / (2 * a2)) * (1 - (1 - C) ** 0.5) # Prevents zero flow
            H[0, n] = Q[0, n] * B + Cmenos[1, n]
      
    # Reciprocating Pump with Outlet Valve Closure [Accidentaly] with Rupture Disk
    if inlet_choice==11:
        # Outlet Boundary Condition
        if H[Nx-2,n]>=rupture_disk_pressure:
            rupture_activated = True
        
        if not rupture_activated:
            
            if T < 30: 
                #Outlet Boundary Condition
                H[Nx-1, n] = 531.747
                Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B

                #Inlet Boundary Condition
                theta=omega*T
                Q[0, n] = Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
                H[0, n] = Q[0, n] * B + Cmenos[1, n]


            elif 30 <= T <= 60:
                #Outlet Boundary Condition
                Cv = (245 / 3600) ** 2 / (2 * 531.747)
                CdA = Cv * (1 - (T - 30) / 30) ** 1.5
                Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
                H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

                #Inlet Boundary Condition
                theta=omega*T

                if H[1,n]>=2100:
                    Q[0,n]=0
                    H[0, n] = Q[0, n] * B + Cmenos[1, n]

                else: 
                    
                    Q[0, n] = Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
                    H[0, n] = Q[0, n] * B + Cmenos[1, n]

            elif T > 60:
       
                Cv = 0
                CdA = 0
                Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
                H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

                #Inlet Boundary Condition
                theta=omega*T

                if H[1,n]>=2100:
                    Q[0,n]=0
                    H[0, n] = Q[0, n] * B + Cmenos[1, n]

                else: 
                    
                    Q[0, n] = Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
                    H[0, n] = Q[0, n] * B + Cmenos[1, n]

        else:
            #Outlet Boundary Condition
            H[Nx-1, n] = 531.747
            Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B

                #Inlet Boundary Condition
            theta=omega*T

            if H[1,n]>=2100:
                Q[0,n]=0
                H[0, n] = Q[0, n] * B + Cmenos[1, n]

            else: 
                    
                Q[0, n] = Q_mean*(1 + (Amplitude/3)*(np.sin(theta) + np.sin(theta + 2*np.pi/3) + np.sin(theta + 4*np.pi/3)))
                H[0, n] = Q[0, n] * B + Cmenos[1, n]

    # Centrifugal Pump with Outlet Valve Closure [Accidentaly] without Rupture Disk
    if inlet_choice==12:
        # Outlet Boundary Condition
                 
            if T < 30: 
                #Outlet Boundary Condition
                H[Nx-1, n] = 531.747
                Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B

                #Inlet Boundary Condition
                Q[0,n] = (1/(2*a2))*(B - a1 - np.sqrt((B-a1)**2 + 4*a2*(Cmenos[1,n] - Hs)))
                H[0, n] = Q[0, n] * B + Cmenos[1, n]


            elif 30 <= T <= 60:
                #Outlet Boundary Condition
                Cv = ( 245 / 3600) ** 2 / (2 * 531.747)
                CdA = Cv * (1 - (T - 30) / 30) ** 1.5
                Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
                H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

                #Inlet Boundary Condition
                D=(B-a1)**2 + 4*a2*(Cmenos[1,n] - Hs)

                if D<0:
                    D=0    
                    Q[0,n] = (1/(2*a2))*(B - a1 - np.sqrt(D))
                    H[0, n] = Q[0, n] * B + Cmenos[1, n]
                else: 
                    Q[0,n] = (1/(2*a2))*(B - a1 - np.sqrt(D))
                    H[0, n] = Q[0, n] * B + Cmenos[1, n]

            elif T > 60:
       
                Cv = 0
                CdA = 0
                Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
                H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

                #Inlet Boundary Condition
                D=(B-a1)**2 + 4*a2*(Cmenos[1,n] - Hs)

                if D<0:
                    D=0    
                    Q[0,n] = (1/(2*a2))*(B - a1 - np.sqrt(D))
                    H[0, n] = Q[0, n] * B + Cmenos[1, n]
                else: 
                    Q[0,n] = (1/(2*a2))*(B - a1 - np.sqrt(D))
                    H[0, n] = Q[0, n] * B + Cmenos[1, n]

    print(f"T={T:.2f}: H[0,n]={H[0, n]:.2f}, Q[0,n]={Q[0, n]:.4f}")

    # Reservor Boundary Condition
    if inlet_choice==13:

        if T < 30: 
            # Inlet Boundary Condition
            H[0, n] = 1950
            Q[0, n] = (H[0, n] - Cmenos[1, n]) / B

            #Outlet Boundary Condition
            H[Nx-1, n] = 536.75
            Q[Nx-1, n] = (Cmais[Nx-2, n]-H[Nx-1, n]) / B

        elif 30 <= T <= 60:
            #Outlet Boundary Condition
            Cv = ( 240 / 3600) ** 2 / (2 * 536.75)
            CdA = Cv * (1 - (T - 30) / 30) ** 1.5
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]

            # Inlet Boundary Condition
            H[0, n] = 1950
            Q[0, n] = (H[0, n] - Cmenos[1, n]) / B

        elif T > 60:
       
            Cv = 0
            CdA = 0
            Q[Nx-1, n] = -CdA * B + np.sqrt((B ** 2) * (CdA ** 2) + 2 * CdA * Cmais[Nx-2, n])
            H[Nx-1, n] = Cmais[Nx-2, n] - B * Q[Nx-1, n]


            # Inlet Boundary Condition
            H[0, n] = 1950
            Q[0, n] = (H[0, n] - Cmenos[1, n]) / B


# === User Input for Output Folder ===
output_folder = input("Enter the output folder name (Press Enter for default 'Results'): ").strip()
if output_folder == "":
    output_folder = "Results"  # Default folder name

# Ensure the folder exists or create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"Saving results in: {output_folder}/")

# Create figure with 3 stacked subplots (3x1 layout)
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=False)
ax1 = axes[0]  # Animated: Transient pressure profile
ax2 = axes[1]  # Animated: Pressure at key locations over time
ax3 = axes[2]  # Animated: Flow in and out over time

# Time vector (ensuring correct length)
u = np.linspace(0, tf, Nt)  

# Initialize empty lines for animation
line1, = ax1.plot([], [], label="Transiente Hidraulico", color='blue')
line2, = ax2.plot([], [], label="Altura Manométrica [123 km]", color='red')
line3, = ax2.plot([], [], label="Altura Manométrica [0 km]", color='green')
line4, = ax2.plot([], [], label="Altura Manométrica [55 km]", color='purple')
line5, = ax3.plot([], [], label="Vazão [0 Km]", color='blue')
line6, = ax3.plot([], [], label="Vazão [123 Km]", color='orange')

# 🔹🔹 CHANGED: Include Terrain Profile & Steady-State Pressure in the Transient Pressure Profile Video 🔹🔹
terrain_plot, = ax1.plot(LR, HR, label="Perfil do Terreno", linestyle='-', color='brown')
steady_state_plot, = ax1.plot(x, H[:, 0], label="Steady-State Pressure", linestyle='dotted', color='black')

# # Initialize the fill areas as PolyCollection objects
#fill_terrain = ax1.fill_between(LR, HR, y2=min(HR), color='sandybrown', alpha=0.6)

# Fix the x-axis for the second and third subplots so data grows over time
ax2.set_xlim([0, tf * dt])
ax3.set_xlim([0, tf * dt])

# Calculate min and max values for the Y-axis from your data
y_min = min(H[1, :].min(), H[250, :].min(), H[Nx-1, :].min())
y_max = max(H[1, :].max(), H[250, :].max(), H[Nx-1, :].max())

# Add a margin for better visualization
margin = (y_max - y_min) * 0.1
y_min -= margin
y_max += margin

# Set Y-axis limits for the second subplot
ax2.set_ylim([y_min, y_max])

# Labels and grid for subplot 1
ax1.set_xlabel('Comprimento [m]')
ax1.set_ylabel('Altura Manométrica H [m]')
ax1.legend()
ax1.set_title('Transiente Hidraulico')
ax1.grid(True)

# Labels and grid for subplot 2 (Pressure at key locations)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Pressure Head (H) [m]')
ax2.set_title('Altura Manométrica [0, 65 e 123 Km]')
ax2.legend()
ax2.grid(True)

# Labels and grid for subplot 3 (Flow in and out)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Flow Rate (Q) [m³/s]')
ax3.legend()
ax3.grid(True)

# --- Animation Function ---
def update(n):
    # Update subplot 1: Transient pressure profile along x
    line1.set_data(x, H[:, n])

    # # Fill below the Terrain Profile
    #ax1.fill_between(LR, HR, y2=min(HR), color='sandybrown', alpha=0.6)

    # 🔹🔹 CHANGED: Ensure Pressure Over Time Includes All Three Locations 🔹🔹
    line2.set_data(u[:n+1] * dt, H[Nx-1, :n+1])  # Outlet
    line3.set_data(u[:n+1] * dt, H[1, :n+1])  # Inlet
    line4.set_data(u[:n+1] * dt, H[250, :n+1])  # Middle

    # 🔹🔹 CHANGED: Ensure Flow In and Out Includes Both Data Points 🔹🔹
    line5.set_data(u[:n+1] * dt, Q[1, :n+1])  # Flow at inlet
    line6.set_data(u[:n+1] * dt, Q[Nx-1, :n+1])  # Flow at outlet

    return line1, line2, line3, line4, line5, line6

# --- Save Combined Animation ---
metadata = dict(title='Hydraulic Transient Animation', artist='Pipeline Simulation')
writer = FFMpegWriter(fps=30, metadata=metadata)

with writer.saving(fig, os.path.join(output_folder, 'Transiente_Hidraulico.mp4'), 100):
    for n in range(0, Nt, 10):  
        update(n)
        writer.grab_frame()

# --- Save Separate Videos for Each Subplot ---
def save_subplot_video(ax, filename, lines, update_functions, additional_static_plots=None):
    fig_single, ax_single = plt.subplots(figsize=(12, 5))

    # 🔹🔹 CHANGED: Ensure Terrain Profile & Steady-State Line are in Transient Pressure Video 🔹🔹
    if additional_static_plots:
        for plot in additional_static_plots:
            ax_single.plot(plot.get_xdata(), plot.get_ydata(), linestyle=plot.get_linestyle(), color=plot.get_color(), label=plot.get_label())

    line_objs = [ax_single.plot([], [], label=line.get_label(), color=line.get_color())[0] for line in lines]

    # Copy axis labels and grid
    ax_single.set_xlabel(ax.get_xlabel())
    ax_single.set_ylabel(ax.get_ylabel())
    ax_single.grid(True)
    ax_single.set_xlim(ax.get_xlim())
    ax_single.set_ylim(ax.get_ylim())

    # Animation update function
    def update_subplot(n):
        for line, update_func in zip(line_objs, update_functions):
            update_func(n, line)
        return line_objs

    # Save animation
    writer_single = FFMpegWriter(fps=30, metadata=metadata)
    with writer_single.saving(fig_single, os.path.join(output_folder, filename), 100):
        for n in range(0, Nt, 10):
            update_subplot(n)
            writer_single.grab_frame()

# Subplot 1: Transient Pressure Profile (including terrain and steady-state line)
save_subplot_video(
    ax1,
    'Transient_Pressure_Profile.mp4',
    [line1],
    [lambda n, line: line.set_data(x, H[:, n])],
    additional_static_plots=[terrain_plot, steady_state_plot]
)

# Subplot 2: Pressure Over Time (inlet, middle, and outlet)
save_subplot_video(
    ax2,
    'Pressure_Over_Time.mp4',
    [line2, line3, line4],
    [lambda n, line: line.set_data(u[:n+1] * dt, H[Nx-1, :n+1]),
     lambda n, line: line.set_data(u[:n+1] * dt, H[1, :n+1]),
     lambda n, line: line.set_data(u[:n+1] * dt, H[250, :n+1])]
)

# Subplot 3: Flow In and Out
save_subplot_video(
    ax3,
    'Flow_In_Out.mp4',
    [line5, line6],
    [lambda n, line: line.set_data(u[:n+1] * dt, Q[1, :n+1]),
     lambda n, line: line.set_data(u[:n+1] * dt, Q[Nx-1, :n+1])]
)

# --- 🔹🔹 CHANGED: Save Final Frame from the Full Animation 🔹🔹 ---
plt.savefig(os.path.join(output_folder, 'Final_Frame.png'), dpi=300)
plt.show()

# # Envelope with maximum H values
MaxH = np.zeros(Nx)
MinH = np.zeros(Nx)

for i in range(Nx):
    MaxH[i] = np.max(H[i, :])
    MinH[i] = np.min(H[i, :])

# # --- Create Figure ---
fig_profile, ax_profile = plt.subplots(figsize=(12, 6))

# Plot MaxH, MinH, and Pipeline Profile
ax_profile.plot(x, MaxH, label="Máxima Altura Manométrica Transiente",
                color='crimson', linestyle='-', linewidth=1.8)
ax_profile.plot(x, MinH, label="Mínima Altura Manométrica Transiente",
                color='royalblue', linestyle='-', linewidth=1.8)
ax_profile.plot(LR, HR, label="Perfil do Terreno",
                color='saddlebrown', linestyle='-', linewidth=1.5)
ax_profile.plot(x, H[:, 0], label="Altura Manométrica Permanente",
                color='black', linestyle=':', linewidth=1.5)

# Fill the terrain below
ax_profile.fill_between(LR, HR, y2=min(HR),
                        color='sandybrown', alpha=0.5)

# Beautification (matches first plot style)
ax_profile.set_xlabel("Comprimento (L) [m]", fontsize=12)
ax_profile.set_ylabel("Altura Manométrica (H) [m]", fontsize=12)
ax_profile.set_title("Máximas e Mínimas Pressões Transiente", fontsize=14, weight='bold')
ax_profile.set_xlim(min(x), max(x))
ax_profile.set_ylim(min(MinH.min(), HR.min()) - 10, max(MaxH.max(), HR.max()) + 10)
ax_profile.grid(True, linestyle='--', alpha=0.4)
ax_profile.legend(loc='upper left', fontsize=10, frameon=True)

# Save and show
plt.tight_layout()
output_path = os.path.join(output_folder, "Pressão em Transiente.png")
fig_profile.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved at: {output_path}")


#####################################################################################################################################################

# Coletar Dados do Excel e Plotando as Míminas Espessuras

EM=input("Informe o Caminho do Arquivo com as Espessuras Mínimas:")
Tmin=pd.read_excel(EM)

LEM=Tmin['LEM'].values                  # Posição da espessura Mínima ao Longo da Tubulação
HEM=Tmin['heighting [m]']               # Cota da Espessura Minima
EM3H=Tmin['ESP03'].values               # Espessira Minima à 03:00
EM6H=Tmin['ESP06'].values               # Espessura Mínima à 06:00
EM9H=Tmin['ESP09'].values               # Espessura Mínima à 09:00
EM12H=Tmin['ESP12'].values              # Espessura Mínima à 12:00
TD=Tmin['Taxa de Desgaste'].values      # Taxa de Desgaste Calculada
EM2024=Tmin['Espessura 2024'].values    # Projeção espessuras 2024
EM2030=Tmin['Espessura 2030'].values    # Projeção Espessuras 2030
EM2035=Tmin['Espessura 2035'].values    # Projeção Espessuras 2035
LRT2024=Tmin['Resistencia 2024'].values # Projeção MAOP Tubulação 2024
LRT2030=Tmin['Resistencia 2030'].values # Projeção MAOP Tubulação 2030
LRT2035=Tmin['Resistencia 2035'].values # Projeção MAOP Tubulação 2035
ESP_min=Tmin['ESP MIN'].values          # Projeção MAOP Tubulação 2035



###################################################################################################################################
# Espessuras Originais
esppp = np.array([10.31, 8.74, 7.92, 8.74, 10.31])                                # Espessuras de parede em mm
Lprojeto = np.array([9381, 11314, 54093, 46631, 1581.59])                         # Comprimentos de cada segmento em mm
S = 360                                                                           # Tensão admissível (kgf/cm²)
dext = 9.625 * 25.4                                                               # Diâmetro externo em mm

# Cálculo da Pressão Máxima Admissível de Operação (MAOP) em kgf/cm²
MAOPP = (2 * S * esppp * 0.8 / dext) * 10.1972                                   # Conversão adequada


# Calculando o comprimento acumulado da tubulação
pipeline_length = np.cumsum(Lprojeto)  # Soma cumulativa dos comprimentos
pipeline_length_steps = np.insert(np.repeat(pipeline_length, 2), 0, 0)  # Adiciona 0 inicial
MAOPP_steps = np.insert(np.repeat(MAOPP, 2), 0, MAOPP[0])  # Adiciona o primeiro valor de MAOP no início



##################################################################################################################################
#Análise de Resultados da Inspeção

# Espessuras Medidas em cada trecho de tubulação
ranges = [
    (None, 9381),
    (9381, 20695),
    (20695, 48031),
    (48031, 48440),
    (48440, 67680),
    (67680, 67805),
    (67805, 75052),
    (75052, 77144),
    (77144, 77320),
    (77320, 82320),
    (82320, 82525),
    (82525, 89937),
    (89937, 90148),
    (90148, 100344),
    (100344, 104812),
    (104812, 118710),
    (118710, 121000),
    (121000, 123000)
]

segment_values = []
range_midpoints = []

# # Calculate the segment values for each LEM range
# for r in ranges:
#     if r[0] is None:
#         mask = LEM <= r[1]
#         midpoint = r[1]  # Using upper bound as the midpoint for the first open range
#     else:
#         mask = (LEM > r[0]) & (LEM <= r[1])
#         midpoint = (r[0] + r[1]) / 2
    
#     if np.any(mask):  # Only calculate if there are valid values in the range
#         valid_values = ESP_min[mask]
#         valid_values = valid_values[~np.isnan(valid_values)]  # Remove NaN values
        
#         if len(valid_values) > 0:  # Check if there are still valid values
#             mean_value = np.mean(valid_values)
#             std_dev = np.std(valid_values)
#             segment_value = mean_value - 3 * std_dev
#         else:
#             segment_value = np.nan
        
#         segment_values.append(segment_value)
#         range_midpoints.append(midpoint)
#     else:
#         segment_values.append(np.nan)  # Maintain list length consistency
#         range_midpoints.append(midpoint)

# Manually set segment values
segment_values = [8.32, 6.75, 6.17, 6.17, 6.17, 6.17, 6.17, 6.65, 6.65, 6.65, 6.65, 6.65, 6.65, 6.65, 6.65, 6.65, 6.65, 6.65]
range_midpoints = [(r[1] if r[0] is None else (r[0] + r[1]) / 2) for r in ranges]

# Prepare the data for visualization and analysis
results_df = pd.DataFrame({
    'Range Midpoint': range_midpoints,
    'Segment Value (Mean - 2*SD)': segment_values
})

#####################################################################################################################################################################

# Cálculo da MAOP em Kg/cm² ( Espessuras Medidas - Resultado da Inspeção)

esppm = np.array(segment_values, dtype=float)
Lm = np.array([9381, 11314, 27336, 409, 19240, 125, 7247, 2092, 176, 
               5000, 205, 7412, 211, 10196, 4468, 13898, 2290, 2000])        # Comprimentos em mm
S = 360                                                                      # Tensão admissível (kgf/cm²)
dext = 9.625 * 25.4                                                          # Diâmetro externo em mm2 mpa to kg/cm2

# Cálculo da Pressão Máxima Admissível de Operação (MAOP) em kgf/cm²
MAOPm = (2 * S * esppm * 0.8 / dext) * 10.1972                               # Conversão adequada

#####################################################################################################################################################################
# Calculo da MAOP (Metros de Coluna de Fluido)

# 1. MAOP de Projeto Original

L=np.array([9381, 11314, 54093, 46631, 1581])

A=len(LR)
MAOPH=np.zeros((A))
MAOPHm=np.zeros((A))
# MAOP em metros de coluna de fluido:
for j in range(A):  # Assuming A is the length of MP or the number of iterations needed
    if LR[j] <= L[0]:
        MAOPH[j] = (10*MAOPP[0] / (rho / 1000)) + HR[j]
    elif L[0] < LR[j] <= (L[0] + L[1]):
        MAOPH[j] = (10*MAOPP[1] / (rho / 1000)) + HR[j]
    elif (L[0]+L[1]) < LR[j] <= (L[0] + L[1]+L[2]):
        MAOPH[j] = (10*MAOPP[2] / (rho / 1000)) + HR[j]
    elif (L[0]+L[1]+L[2]) < LR[j] <= (L[0] + L[1]+L[2]+L[3]):
        MAOPH[j] = (10*MAOPP[3] / (rho / 1000)) + HR[j]
    elif (L[0]+L[1]+L[2]+L[3]) < LR[j] <= (L[0] + L[1]+L[2]+L[3]+L[4]):
        MAOPH[j] = (10*MAOPP[4] / (rho / 1000)) + HR[j]

# 2. MAOP atual (após realocações) de Acordo com as Espessuras Medidas

for j in range(A):                                              
    if LR[j] <= 9381:
        MAOPHm[j] = (10*MAOPm[0] / (rho / 1000)) + HR[j]
    elif 9381 < LR[j] <= 20695:
        MAOPHm[j] = (10*MAOPm[1] / (rho / 1000)) + HR[j]
    elif 20695 < LR[j] <= 48031:
        MAOPHm[j] = (10*MAOPm[2] / (rho / 1000)) + HR[j]
    elif 48031 < LR[j] <= 48440:
        MAOPHm[j] = (10*MAOPm[3] / (rho / 1000)) + HR[j]
    elif 48440 < LR[j] <= 67680:
        MAOPHm[j] = (10*MAOPm[4] / (rho / 1000)) + HR[j]
    elif 67680 < LR[j] <= 67805:
        MAOPHm[j] = (10*MAOPm[5] / (rho / 1000)) + HR[j]
    elif 67805 < LR[j] <= 75052:
        MAOPHm[j] = (10*MAOPm[6] / (rho / 1000)) + HR[j]
    elif 75052 < LR[j] <= 77144:
        MAOPHm[j] = (10*MAOPm[7] / (rho / 1000)) + HR[j]
    elif 77144 < LR[j] <= 77320:
        MAOPHm[j] = (10*MAOPm[8] / (rho / 1000)) + HR[j]
    elif 77320 < LR[j] <= 82320:
        MAOPHm[j] = (10*MAOPm[9] / (rho / 1000)) + HR[j]
    elif 82320 < LR[j] <= 82525:
        MAOPHm[j] = (10*MAOPm[10] / (rho / 1000)) + HR[j]
    elif 82525 < LR[j] <= 89937:
        MAOPHm[j] = (10*MAOPm[11] / (rho / 1000)) + HR[j]
    elif 89937 < LR[j] <= 90148:
        MAOPHm[j] = (10*MAOPm[12] / (rho / 1000)) + HR[j]
    elif 90148 < LR[j] <= 100344:
        MAOPHm[j] = (10*MAOPm[13] / (rho / 1000)) + HR[j]
    elif 100344 < LR[j] <= 104812:
        MAOPHm[j] = (10*MAOPm[14] / (rho / 1000)) + HR[j]
    elif 104812 < LR[j] <= 118710:
        MAOPHm[j] = (10*MAOPm[15] / (rho / 1000)) + HR[j]
    elif 118710 < LR[j] <= 121000:
        MAOPHm[j] = (10*MAOPm[16] / (rho / 1000)) + HR[j]
    elif 121000 < LR[j] <= 123000:
        MAOPHm[j] = (10*MAOPm[17] / (rho / 1000)) + HR[j]



####################################################################################################################################################################
# Interpolando Dados para Obter a Máxima Pressão de Operação:

MaxH = np.array(MaxH)
MaxH_interpolator=interp1d(x,MaxH,kind="cubic")
MinH_interpolator=interp1d(x,MinH,kind="cubic")

MaxHi=MaxH_interpolator(LR)
MinHi=MinH_interpolator(LR)

Max_Pressure = ((MaxHi - HR)*(rho/1000))/10
Min_Pressure = ((MinHi - HR)*(rho/1000))/10


# Cálculo da Vida Útil (Espessuras Mínimas)
MaxHvu=MaxH_interpolator(LEM)     #Interpolando para calcular a vida útil da tubulação
Max_Pressure_vu = ((MaxHvu - HEM)*(rho/1000))/10
espminvu=(Max_Pressure_vu * 9.625 * 25.4)/ (360 * 20 * 0.8)

vu = ((ESP_min - espminvu) / TD) + 2019

#########################################################################################################################################################################

segment_starts = [r[0] if r[0] is not None else 0 for r in ranges]  # First value should start at 0
segment_ends = [r[1] for r in ranges]

# Repeat segment values at each boundary to maintain step consistency
segment_values_adjusted = []
x_positions = []

for i in range(len(segment_starts)):
    x_positions.extend([segment_starts[i], segment_ends[i]])  # Add start and end of each segment
    segment_values_adjusted.extend([segment_values[i], segment_values[i]])  # Maintain same value


##############################################################################################################################################################################
 ##  Inserindo a Logo


# Load the logo using PIL
logo_path = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\Logo.jpg"
logo = Image.open(logo_path)

# Original logo size
original_width, original_height = 276, 120

# Set the scaling factor (e.g., 0.5 to reduce size by 50%)
logo_scale = 0.4

# Calculate the new size
new_size = (int(original_width * logo_scale), int(original_height * logo_scale))
print(f'Resized Logo Dimensions: {new_size}')

# Resize the logo
logo = logo.resize(new_size, Image.Resampling.LANCZOS)


################################################################################################################################################################################
# Gráficos do Sistema

# Define the plots with (x, y, label, color, title)
plots = [
    (LEM, Max_Pressure_vu, 'Máxima Pressão de Operação', '#003366', 'Máxima Pressão & MAOP', 
     'Comprimento da Tubulação [m]', 'Pressão [Kg/cm²]', (0, 123000), (30, 250),
     [
         (LEM, LRT2024, 'MAOP [20205]', '#8B0000', 'scatter'),
         (x_positions, (2*S*np.array(segment_values_adjusted)*0.8/dext)*10.1972, 'MAOP Espessura Mínima [2025]', '#555555', 'step')
     ], 'upper right'  # ✅ Legend position included
    ),

    (LEM, espminvu, 'Mínima Espessura', '#1f77b4', 'Espessura Mínima', 
     'Comprimento da Tubulação [m]', 'Espessura [mm]', (0, 123000), (0, 8), []),  # ✅ No legend position

    (LEM, vu, 'Final Vida Útil', '#1f77b4', 'Final de Vida Útil', 
     'Comprimento da Tubulação [m]', 'Final de Vida Útil [ano]', (0, 123000), (2018, 2050), [], 'upper right'),

    (x, MaxH, 'Máxima Pressão de Operação em Transiente', '#1f77b4', 
     'Máxima Pressão de Operação', 'Comprimento da Tubulação [m]', 'Altura Manométrica [mcf]', None, None, []),  # ✅ No legend position
    
    (LEM, EM3H, 'Espessura Minima 3h', '#003366', 'Espessura_Minima',
     'Comprimento da Tubulação [m]', 'Espessura Mínima [mm]', (0, 123000), (5, 12),
     [
         (LEM, EM6H, 'Espessura Mínima 6h', '#00274D'),
         (LEM, EM9H, 'Espessura Mínima 9h', '#8B0000'),
         (LEM, EM12H, 'Espessura Mínima 12h', '#555555')
     ], 'upper right')
]

# ✅ Corrected loop to handle variable tuple sizes
for plot in plots:
    if len(plot) == 11:  # ✅ If legend position is provided
        data_x, data_y, label, color, title, xlabel, ylabel, xlim, ylim, extra_plots, legend_pos = plot
    else:  # ✅ Default case when legend position is missing
        data_x, data_y, label, color, title, xlabel, ylabel, xlim, ylim, extra_plots = plot
        legend_pos = 'best'  # ✅ Default legend position if not specified

    fig, ax = plt.subplots(figsize=(8, 6))

    # Main scatter plot
    ax.scatter(data_x, data_y, color=color, label=label, s=10, marker='o', linewidth=0, zorder=3)

    # Overlay additional scatter and step plots
    for extra in extra_plots:
        if len(extra) == 4:
            extra_x, extra_y, extra_label, extra_color = extra
            plot_type = 'scatter'
        else:
            extra_x, extra_y, extra_label, extra_color, plot_type = extra

        if plot_type == 'scatter':
            ax.scatter(extra_x, extra_y, color=extra_color, label=extra_label, s=5, zorder=3)
        elif plot_type == 'step':
            ax.step(extra_x, extra_y, where='pre', linestyle='-', color=extra_color, label=extra_label)

    # Labels, title, and grid
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc=legend_pos, fontsize=10)  # ✅ Uses correct legend position
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    # Reduce minor grid density
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

    # Set axis limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Add logo
    logo_ax = fig.add_axes([0.8, 0.88, 0.1, 0.1], anchor='NE', zorder=10)
    logo_ax.imshow(logo)
    logo_ax.axis('off')

    # Save the plot
    output_path = os.path.join(output_folder, f'{title.replace(" ", "_")}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f'Plot saved at: {output_path}')

    plt.show()
    plt.close()





########################################################################################################################

# Step plot with logo scaling
fig, ax = plt.subplots(figsize=(8, 6))

# Plot with formal color scheme
ax.plot(LR[:], MAOPHm[:], linestyle='-', color='#003366', label='MAOP Tubulação Nova')  # Dark navy blue
ax.plot(LR[:], MAOPH[:], linestyle='-', color='#8B0000', label='MAOP Tubulação 2025')  # Deep red

# Titles and labels
ax.set_title('Maximum Allowable Operation Pressure [MAOP]', fontsize=14, fontweight='bold')
ax.set_xlabel('Comprimento Tubulação [m]', fontsize=12)
ax.set_ylabel('Máxima Pressão [mcf]', fontsize=12)

# Set x-axis limits
ax.set_xlim(0, 123000)

# Grid and legend
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right', fontsize=12, frameon=True)

# Add the logo outside the plot area, top right
logo_ax = fig.add_axes([0.8, 0.87, 0.1, 0.1], anchor='NE', zorder=10)
logo_ax.imshow(logo)
logo_ax.axis('off')

# Save the step plot
output_path = os.path.join(output_folder, 'MAOP 2025.png')
plt.savefig(output_path, bbox_inches='tight')
print(f'Plot saved at: {output_path}')


plt.show()
plt.close()


#################################################################################################################################
# Análise dos Resultados da Inspeção de acordo com a ASME B31G Modificada
    
    # Calculate the failure pressure of a corroded pipeline using Modified ASME B31G.
    
    # Parameters:
    # SMYS : float - Specified Minimum Yield Strength of the pipe material (psi or MPa)
    # tm   : float - Measured remaining wall thickness (inches or mm)
    # d    : float - Maximum depth of the corrosion defect (inches or mm)
    # Ld    : float - Longitudinal length of the corrosion defect (inches or mm)
    # D    : float - Outside diameter of the pipe (inches or mm)
    
    # Returns:
    # Pf    : float - Estimated failure pressure (same unit as SMYS)

# Given parameters
SMYS = 52000  # API 5L X52, 52000 PSI
Sflow = 1.1 * SMYS  # Flow stress

# Load data (assuming df is already loaded)
df = df.fillna("")  # Replace NaN with empty strings

# Define corrosion types for filtering
corrosion_types = ["Anomaly  / Corrosion", "Anomaly  / Corrosion cluster"]

# Debug: Ensure 'anom. type/ident' column exists
if "anom. type/ident" not in df.columns:
    raise KeyError("Column 'anom. type/ident' not found in the dataset.")

# Normalize column values
print("Unique Anomaly Types Found Before Filtering:", df["anom. type/ident"].unique())
df["anom. type/ident"] = df["anom. type/ident"].astype(str).str.strip()

# Apply filtering
df_filtered = df[df["anom. type/ident"].isin(corrosion_types)]

# Debug: Print count of filtered data
print("Filtered Data Count:", len(df_filtered))
if len(df_filtered) == 0:
    raise ValueError("No corrosion anomalies found! Check data formatting.")

# Initialize lists for failure pressures and corresponding LR values
Pfd_kg_cm2_list = []
LR_valid = []
Surface_Location = []

# Iterate over all rows in DataFrame
for index, row in df_filtered.iterrows():
    LA, tr, d, LRd, surface_loc = (
        row.get("LA", ""),
        row.get("rem. t [mm]", ""),
        row.get("abs. depth [mm]", ""),
        row.get("L", ""),
        row.get("surf. loc.", "").strip().upper()
    )
    
    try:
        LA = float(LA) / 25.4 if str(LA).strip() else np.nan
        tr = float(tr) / 25.4 if str(tr).strip() else np.nan
        d = float(d) / 25.4 if str(d).strip() else np.nan
        LRd = float(LRd) if str(LRd).strip() else np.nan
    except ValueError:
        continue  # Skip invalid rows

    if any(np.isnan([LA, tr, d, LRd])) or tr <= 0:
        continue

    D = 9.625  # Pipe outer diameter in inches
    z = (LA**2) / (D * tr)
    M = np.sqrt(1 + 0.6275 * z - 0.003375 * z**2) if z <= 50 else 3.3 + 0.032 * z
    SF = Sflow * (1 - 0.85 * (d / tr)) / (1 - 0.85 * (d / tr) / M)
    Pfd_psi = (2 * SF * tr) / D
    Pfd_kg_cm2 = Pfd_psi * 0.070307

    Pfd_kg_cm2_list.append(Pfd_kg_cm2)
    LR_valid.append(LRd)
    Surface_Location.append(surface_loc)

Pfd_kg_cm2 = np.array(Pfd_kg_cm2_list)
LR_valid = np.array(LR_valid)
Surface_Location = np.array(Surface_Location)
valid_indices = ~np.isnan(Pfd_kg_cm2) & ~np.isnan(LR_valid)

# Define colors for anomaly classification
surface_colors = {"EXT": "red", "INT": "blue"}
def get_color(surface):
    return surface_colors.get(surface, "gray")


###############################################################################################################################################################################
# Plot MAOPm against the range midpoints
###############################################################################################################################################################################
# Plot MAOPm as constant lines
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(len(ranges)):
    if i == 0:
        x_start = 0
    else:
        x_start = ranges[i][0]
    x_end = ranges[i][1]
    if not np.isnan(MAOPm[i]):
        ax.hlines(y=MAOPm[i], xmin=x_start, xmax=x_end, colors='gray', linestyles='-', linewidth=3.5, label='Máxima Pressão de Operação Admissível [MAOP]' if i == 0 else "")

ax.set_xlabel('Comprimento da Tubulação [m]')
ax.set_ylabel('Pressão [kg/cm²]')
ax.set_title('Máxima Pressão Admissível e Pressão de Falha Anomalias de Corrosão')
ax.grid(True)
ax.legend()

# Scatter plot of Pfd_Kg_cm²
for surface_class in np.unique(Surface_Location[valid_indices]):
    mask = (Surface_Location == surface_class) & valid_indices
    ax.scatter(
        LR_valid[mask], Pfd_kg_cm2[mask],
        label=f"{surface_class} Pressão de Falha", color=get_color(surface_class), s=60, edgecolors='cyan'
    )


# Set axis limits
ax.set_xlim(0, 123000)
ax.set_ylim(0, 300)

# Add grid
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)

# Add legend at the upper right
ax.legend(loc='upper right', fontsize=10, frameon=True)

# Add the logo outside the plot area, top right
logo_ax = fig.add_axes([0.8, 0.88, 0.1, 0.1], anchor='NE', zorder=10)
logo_ax.imshow(logo)
logo_ax.axis('off')

# Save the step plot
output_path = os.path.join(output_folder, 'Pressão de Falha & MAOP.png')
plt.savefig(output_path, bbox_inches='tight')
print(f'Plot saved at: {output_path}')


ax.legend()
plt.show()



################################################################################################################################################################################
## Plot External and Internal Corrosion
##########################################################################################################################################################



fig, ax = plt.subplots(figsize=(8, 6))

# Plot data with color coding based on surface location
for surface_class in np.unique(Surface_Location[valid_indices]):
    mask = (Surface_Location == surface_class) & valid_indices
    ax.scatter(
        LR_valid[mask], Pfd_kg_cm2[mask],
        label=f"{surface_class} Failure Pressure", color=get_color(surface_class), s=16, edgecolors='cyan'
    )

# Plot max pressure with a formal deep red
ax.plot(LR, Max_Pressure, label="Maximum Operating Pressure", color="#8B0000", linewidth=1.8)  # Deep red

# Plot min pressure with a formal navy blue
ax.plot(LR, Min_Pressure, label="Minimum Operating Pressure", color="#003366", linewidth=1.8)  # Dark navy blue

# Add title
ax.set_title("Failure Pressure & Operating Pressure", fontsize=14, fontweight="bold")

# Labels
ax.set_xlabel("Comprimento da Tubulação [m]", fontsize=12)
ax.set_ylabel("Pressure [kg/cm²]", fontsize=12)

# Set axis limits
ax.set_xlim(0, 123000)
ax.set_ylim(-10, 300)

# Add grid
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)

# Add legend at the upper right
ax.legend(loc='upper right', fontsize=10, frameon=True)

# Add the logo outside the plot area, top right
logo_ax = fig.add_axes([0.8, 0.88, 0.1, 0.1], anchor='NE', zorder=10)
logo_ax.imshow(logo)
logo_ax.axis('off')

# Save the step plot
output_path = os.path.join(output_folder, 'Internal and External Corrosion.png')
plt.savefig(output_path, bbox_inches='tight')
print(f'Plot saved at: {output_path}')

# Show plot
plt.show()



#################################################################################################################################################
# Estimated Repair Factor
#################################################################################################################################################

valid_indices = np.isin(df["L"], df_filtered["L"])

# Filter Max_Pressure using valid_indices to match Pfd_kg_cm2's shape
Max_Pressure_filtered = Max_Pressure[valid_indices]

safe_max_pressure = np.where(Max_Pressure_filtered == 0, np.nan, Max_Pressure_filtered)
ERF = safe_max_pressure / Pfd_kg_cm2
ERF = np.nan_to_num(ERF, nan=0)

# Now, safely plot ERF
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(LR_valid, ERF, label="Estimated Repair Factor (ERF)", color="blue", s=20, edgecolors='cyan')
ax.set_xlabel("Comprimento da Tubulação [m]")
ax.set_ylabel("ERF")
ax.legend()
ax.set_title('Estimated Repair Factor', fontsize=12, fontweight='bold')
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
ax.set_xlim(0, 123000)
ax.set_ylim(0, 1.5)

# Add the logo outside the plot area, top right
logo_ax = fig.add_axes([0.8, 0.88, 0.1, 0.1], anchor='NE', zorder=10)
logo_ax.imshow(logo)
logo_ax.axis('off')

# Save the step plot
output_path = os.path.join(output_folder, 'ERF.png')
plt.savefig(output_path, bbox_inches='tight')
print(f'Plot saved at: {output_path}')

plt.show()
plt.close(fig)

################################################################################################################################################
# Plotar rota do Mineroduto e anomalias Filtradas por ERF
#####################################################################################################################################


# Function to load the KML ee
def load_kml_route(kml_file):
    with open(kml_file, 'r', encoding='utf-8') as f:
        kml_content = f.read()

    # Parse KML using ElementTree
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    root = ET.fromstring(kml_content)
    
    # Search for all LineString elements in the KML file
    for placemark in root.findall(".//kml:Placemark", namespace):
        line_string = placemark.find(".//kml:LineString", namespace)
        if line_string is not None:
            coordinates = line_string.find(".//kml:coordinates", namespace).text.strip()
            coord_list = []
            for coord in coordinates.split():
                lon, lat, *_ = map(float, coord.split(','))
                coord_list.append((lon, lat))
            return LineString(coord_list)  # Convert to Shapely LineString
    
    raise ValueError("❌ No LineString pipeline route found in the KML file.")

# Filepath to KML
kml_filepath = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\Rota Mineroduto.kml"

# Load pipeline route
pipeline_route = load_kml_route(kml_filepath)

# Convert route to GeoDataFrame
gdf_route = gpd.GeoDataFrame(geometry=[pipeline_route], crs="EPSG:4326")

# Define ERF threshold
ERF_THRESHOLD = 0.75  # Adjust as needed

# Ensure 'ERF' exists in the DataFrame
df["ERF"] = np.nan  # Initialize with NaN

# Assign computed ERF values back to the original DataFrame
valid_indices = np.isin(df["L"], df_filtered["L"])
df.loc[valid_indices, "ERF"] = ERF

# Drop invalid coordinates
df = df[(df["long"].astype(str).str.strip() != '') & (df["lat"].astype(str).str.strip() != '')]
df["long"] = pd.to_numeric(df["long"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df = df.dropna(subset=["long", "lat"])

# Save updated DataFrame with ERF values
output_filepath = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\Updated_Anomalies.xlsx"
df.to_excel(output_filepath, index=False)
print(f"✅ ERF values written to {output_filepath}")

# Filter corrosion anomalies with ERF threshold
df_valid = df[(df["anom. type/ident"].isin(["Anomaly  / Corrosion cluster", "Anomaly  / Corrosion"])) &
              (df["ERF"].notna()) &  # Ensure ERF is not NaN
              (df["ERF"] > ERF_THRESHOLD)].copy()

# Ensure valid coordinates
df_valid = df_valid[(df_valid["long"].astype(str).str.strip() != '') & (df_valid["lat"].astype(str).str.strip() != '')]
df_valid["long"] = pd.to_numeric(df_valid["long"], errors="coerce")
df_valid["lat"] = pd.to_numeric(df_valid["lat"], errors="coerce")
df_valid = df_valid.dropna(subset=["long", "lat"])

# Convert to GeoDataFrame
gdf_corrosion = gpd.GeoDataFrame(
    df_valid, geometry=gpd.points_from_xy(df_valid["long"], df_valid["lat"]), crs="EPSG:4326"
)

# Create figure with increased height
fig_map = plt.figure(figsize=(12, 16))  # Taller figure
ax_map = fig_map.add_subplot(111)

# Plot Pipeline Route with Corrosion Anomalies
gdf_route_3857 = gdf_route.to_crs(epsg=3857)
gdf_route_3857.plot(ax=ax_map, color='cyan', linewidth=1.5, label="Pipeline Route")
gdf_corrosion_3857 = gdf_corrosion.to_crs(epsg=3857)
gdf_corrosion_3857.plot(ax=ax_map, color='red', marker='o', markersize=80, alpha=1, edgecolor='yellow', label=f"Corrosion Anomalies")

# Get bounds of the pipeline route (BEFORE modifying them)
bounds = gdf_route_3857.geometry.total_bounds

# Define separate expansion factors for X and Y
expand_y = (bounds[3] - bounds[1]) * 2  # Expand Y by 50% of total height

# Expand map area BEFORE adding basemap
ax_map.set_xlim(bounds[0], bounds[2])  # Expand horizontally
ax_map.set_ylim(bounds[1] - expand_y, bounds[3] + expand_y)  # Expand vertically

# Add basemap AFTER modifying limits
ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, crs='EPSG:3857')

ax_map.set_title(f"Pipeline Route with Corrosion Anomalies (ERF > {ERF_THRESHOLD})")
ax_map.set_xlabel("Longitude")
ax_map.set_ylabel("Latitude")
ax_map.legend()
ax_map.grid(True, linestyle="--", alpha=0.5)

# Save and Show
plt.savefig(os.path.join(output_folder, "Pipeline_Map_ERF.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# Plot Pipeline Length vs Elevation with Corrosion Anomalies
fig_profile = plt.figure(figsize=(12, 6))
ax_profile = fig_profile.add_subplot(111)

ax_profile.plot(df["L"], df["H"], label="Pipeline Profile", color='gray', linewidth=2)

# ✅ Fill the area below the pipeline (terrain)
ax_profile.fill_between(df["L"], df["H"], df["H"].min() - 10, color="lightgray", alpha=0.5, label="Terrain")

ax_profile.scatter(gdf_corrosion["L"], gdf_corrosion["H"], 
                   color='red', edgecolors='yellow', linewidth=1.2,
                   label=f"Corrosion Anomalies (ERF > {ERF_THRESHOLD})", s=120)
ax_profile.set_xlabel("Pipeline Length (m)")
ax_profile.set_ylabel("Elevation (m)")
ax_profile.set_title("Pipeline Profile with Corrosion Anomalies")
ax_profile.legend()

# ✅ Define major and minor ticks
ax_profile.set_xticks(np.linspace(0, 123000, 10))  # 25 major grid divisions
ax_profile.set_yticks(np.linspace(ax_profile.get_ylim()[0], ax_profile.get_ylim()[1], 15))  # 15 Y divisions

# ✅ Enable minor ticks
ax_profile.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))  # 4 minor ticks per major tick
ax_profile.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))  

# ✅ Enable both major and minor grid lines
ax_profile.grid(True, linestyle="--", linewidth=1, alpha=0.7)  # Bold major grid
ax_profile.grid(True, which="minor", linestyle=":", linewidth=0.7, alpha=0.7)  # More visible minor grid

# ✅ Ensure the grid is visible
ax_profile.set_facecolor("white")  # White background for contrast

# ✅ Axis Limits
ax_profile.set_xlim(0, 123000)  # Set limits for X-axis (Pipeline Length)


plt.savefig(os.path.join(output_folder, "Pipeline_ERF.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# --------------------------
# 3️⃣ 3D Histogram of Anomaly Depth Distribution
# --------------------------
fig_hist = plt.figure(figsize=(12, 8))
ax_hist = fig_hist.add_subplot(111, projection='3d')

hist_data = df_valid[["ERF", "L"]].dropna()
L_values = hist_data["L"].values
depth_values = hist_data["ERF"].values

# Define bins to fully cover the range
x_bins = np.linspace(0, 123000, 21)  # 20 equal bins covering full range
y_bins = np.linspace(0, 1, 11)  # 10 equal bins covering full range

hist, xedges, yedges = np.histogram2d(L_values, depth_values, bins=[x_bins, y_bins])

# ✅ Correct bar placement (align bars exactly with bin edges)
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")  # Use bin edges directly
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)  # Bars start at Z = 0

# ✅ Set dx, dy to match bin widths exactly
dx = np.diff(xedges)[:, None].repeat(len(yedges) - 1, axis=1).ravel()
dy = np.diff(yedges)[None, :].repeat(len(xedges) - 1, axis=0).ravel()
dz = hist.ravel()  # Bar heights from histogram counts

# ✅ Define colormap with white for zero values
cmap = cm.viridis
cmap_array = cmap(np.linspace(0, 1, 256))
cmap_array[0] = [1, 1, 1, 1]  # Set the lowest value (zero quantity) to white
custom_cmap = ListedColormap(cmap_array)

# Normalize color range
norm = Normalize(vmin=dz.min(), vmax=dz.max())
colors = custom_cmap(norm(dz))

# ✅ Create the 3D histogram bars
ax_hist.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=colors)

# ✅ Set axis limits and ticks to ensure full coverage
ax_hist.set_xlim([0, 123000])  # Ensure full pipeline length range
ax_hist.set_ylim([0, 1])  # Ensure full anomaly depth percentage range
ax_hist.set_xticks(np.linspace(0, 123000, 11))  # 10 intervals for better visualization
ax_hist.set_yticks(np.linspace(0, 1, 11))  # 10 intervals to match binning

# ✅ Adjust colorbar size and label
# sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=ax_hist, shrink=0.3, aspect=6, pad=0.1, label='Counts')

# ✅ Labels and title
ax_hist.set_xlabel("Pipeline Length (m)")
ax_hist.set_ylabel("Anomaly ERF")
ax_hist.set_zlabel("Quantity")
ax_hist.set_title(f"3D Histogram Corrosion Anomalies [ERF > {ERF_THRESHOLD}]")


# Add the logo outside the plot area, top right
# logo_ax = fig.add_axes([0.8, 0.85, 0.1, 0.1], anchor='NE', zorder=10)
# logo_ax.imshow(logo)
# logo_ax.axis('off')

# Save the step plot
#output_path = os.path.join(output_folder, 'Pipeline_Route_and_Anomalies.png')
#plt.savefig(output_path, bbox_inches='tight')
#print(f'Plot saved at: {output_path}')

# Adjust layout to prevent overlap
plt.savefig(os.path.join(output_folder, "Anomaly_Histogram_ERF.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#################################################################################################################
# Ploting Corrosion Anomalies Filtered By Depht
#################################################################################################################

# Define depth threshold (minimum depth)
DEPTH_THRESHOLD_MIN = 40  # Minimum depth

# Load dataset
df = pd.read_excel(r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\Updated_Anomalies.xlsx")

# Convert "wl [%]" to numeric
df["wl [%]"] = pd.to_numeric(df["wl [%]"], errors="coerce")

# Filter anomalies based on depth criteria
df_valid = df[(df["anom. type/ident"].isin(["Anomaly  / Corrosion cluster", "Anomaly  / Corrosion"])) &
              (df["wl [%]"].notna()) &
              (df["wl [%]"] >= DEPTH_THRESHOLD_MIN)].copy()

# Ensure valid coordinates
df_valid = df_valid[(df_valid["long"].astype(str).str.strip() != '') & (df_valid["lat"].astype(str).str.strip() != '')]
df_valid["long"] = pd.to_numeric(df_valid["long"], errors="coerce")
df_valid["lat"] = pd.to_numeric(df_valid["lat"], errors="coerce")
df_valid = df_valid.dropna(subset=["long", "lat"])

# Convert to GeoDataFrame
gdf_corrosion = gpd.GeoDataFrame(
    df_valid, geometry=gpd.points_from_xy(df_valid["long"], df_valid["lat"]), crs="EPSG:4326")


# Create figure with subplots
fig_map = plt.figure(figsize=(12, 16))  # Taller figure
ax_map = fig_map.add_subplot(111)

# Plot Pipeline Route with Corrosion Anomalies
gdf_route_3857 = gdf_route.to_crs(epsg=3857)
gdf_route_3857.plot(ax=ax_map, color='cyan', linewidth=1, label="Pipeline Route")
gdf_corrosion_3857 = gdf_corrosion.to_crs(epsg=3857)
gdf_corrosion_3857.plot(ax=ax_map, color='red', marker='o', markersize=100, alpha=1,edgecolor='yellow',
                         label=f"Corrosion Anomalies (Depth >= {DEPTH_THRESHOLD_MIN}%)")


# Define separate expansion factors for X and Y
expand_y = (bounds[3] - bounds[1]) * 2  # Expand Y by 50% of total height

# Expand map area BEFORE adding basemap
ax_map.set_xlim(bounds[0], bounds[2])  # Expand horizontally
ax_map.set_ylim(bounds[1] - expand_y, bounds[3] + expand_y)  # Expand vertically

# Expand map area BEFORE adding basemap
ax_map.set_xlim(bounds[0], bounds[2])  # Expand horizontally
ax_map.set_ylim(bounds[1] - expand_y, bounds[3] + expand_y)  # Expand vertically

# Add basemap
ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, crs='EPSG:3857')
ax_map.set_title("Pipeline Route with Corrosion Anomalies")
ax_map.legend()
ax_map.grid(True, linestyle="--", alpha=0.5)

# Save and Show
plt.savefig(os.path.join(output_folder, "Pipeline_Map_ERF.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#############################################################################################################################################


# Plot Pipeline Length vs Elevation with Corrosion Anomalies
fig_profile = plt.figure(figsize=(12, 6))
ax_profile = fig_profile.add_subplot(111)

ax_profile.plot(df["L"], df["H"], label="Pipeline Profile", color='gray', linewidth=1)

# ✅ Fill the area below the pipeline (terrain)
ax_profile.fill_between(df["L"], df["H"], df["H"].min() - 10, color="lightgray", alpha=0.5, label="Terrain")


ax_profile.scatter(gdf_corrosion["L"], gdf_corrosion["H"], color='red', edgecolors='yellow',
                    label=f"Corrosion Anomalies (Depth >= {DEPTH_THRESHOLD_MIN}%)", s=80)
ax_profile.set_xlabel("Pipeline Length (m)")
ax_profile.set_ylabel("Elevation (m)")
ax_profile.set_title("Pipeline Profile with Corrosion Anomalies")
ax_profile.legend()


# ✅ Set fixed X-axis limits (Pipeline Length)
ax_profile.set_xlim(0, 123000)

# ✅ Define major and minor ticks
ax_profile.set_xticks(np.linspace(0, 123000, 10))  # 25 major grid divisions
ax_profile.set_yticks(np.linspace(ax_profile.get_ylim()[0], ax_profile.get_ylim()[1], 15))  # 15 Y divisions

# ✅ Enable minor ticks
ax_profile.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))  # 4 minor ticks per major tick
ax_profile.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))  

# ✅ Enable both major and minor grid lines
ax_profile.grid(True, linestyle="--", linewidth=1, alpha=0.7)  # Bold major grid
ax_profile.grid(True, which="minor", linestyle=":", linewidth=0.7, alpha=0.7)  # More visible minor grid

# ✅ Ensure the grid is visible
ax_profile.set_facecolor("white")  # White background for contrast


plt.savefig(os.path.join(output_folder, "Pipeline_Profile_WL.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()






##### Poting the histogram

# 3D Histogram for Anomaly Depth Distribution
fig_hist = plt.figure(figsize=(12, 8))
ax_hist = fig_hist.add_subplot(111, projection='3d')

hist_data = df_valid[["wl [%]", "L"]].dropna()
L_values = hist_data["L"].values
depth_values = hist_data["wl [%]"].values

# Define bins
x_bins = np.linspace(0, 123000, 21)
y_bins = np.linspace(0, 100, 11)

hist, xedges, yedges = np.histogram2d(L_values, depth_values, bins=[x_bins, y_bins])

# Create 3D histogram bars
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
dx = np.diff(xedges)[:, None].repeat(len(yedges) - 1, axis=1).ravel()
dy = np.diff(yedges)[None, :].repeat(len(xedges) - 1, axis=0).ravel()
dz = hist.ravel()

# ✅ Define colormap with white for zero values
cmap = cm.viridis
cmap_array = cmap(np.linspace(0, 1, 256))
cmap_array[0] = [1, 1, 1, 1]  # Set the lowest value (zero quantity) to white
custom_cmap = ListedColormap(cmap_array)

# Normalize color range
norm = Normalize(vmin=dz.min(), vmax=dz.max())
colors = custom_cmap(norm(dz))


ax_hist.bar3d(xpos.ravel(), ypos.ravel(), np.zeros_like(dz), dx, dy, dz, shade=True, color=colors)
ax_hist.set_xlabel("Pipeline Length (m)")
ax_hist.set_ylabel("Anomaly Depth [%]")
ax_hist.set_zlabel("Quantity")
ax_hist.set_title(f"3D Histogram Corrosion Anomalies [WL > {DEPTH_THRESHOLD_MIN}]")

# Add the logo outside the plot area, top right
# logo_ax = fig.add_axes([0.8, 0.85, 0.1, 0.1], anchor='NE', zorder=10)
# logo_ax.imshow(logo)
# logo_ax.axis('off')

# # Save the step plot
# output_path = os.path.join(output_folder, 'Anomalies_Filtered_Depht.png')
# plt.savefig(output_path, bbox_inches='tight')
# print(f'Plot saved at: {output_path}')

#plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.07, hspace=0.15)

plt.savefig(os.path.join(output_folder, "Anomaly_Histogram_WL.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


#####################################################################################################################
# Plotting the anomalies in the circunferenctial position and also in the pipeline prfile profile
####################################################################################################################


# Function to convert o'clock format to degrees
def oclock_to_degrees(time_str):
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})", str(time_str))
    if match:
        hours, minutes, _ = map(int, match.groups())
        degrees = (hours % 12) * 30 + minutes * 0.5
        return degrees
    return np.nan  # Handle invalid formats

# --------------------------
# Apply conversion if column exists
# --------------------------
if "o'clock" in df.columns:
    df["Circumference Degrees"] = df["o'clock"].fillna("").apply(oclock_to_degrees)
    print("✅ 'Circumference Degrees' column computed successfully.")
else:
    print("⚠️ Warning: Column 'o'clock' is missing in df. 'Circumference Degrees' cannot be computed.")
    df["Circumference Degrees"] = np.nan

# --------------------------
# Save the updated DataFrame back to Excel
# --------------------------
try:
    df.to_excel(RP, index=False)
    print("✅ 'Circumference Degrees' column saved successfully to Excel.")
except Exception as e:
    print(f"❌ Error saving file: {e}")
    raise


# Filter only corrosion anomalies and create a copy to avoid SettingWithCopyWarning
df_corrosion = df[df["anom. type/ident"].isin(["Anomaly  / Corrosion", "Anomaly  / Corrosion cluster"])].copy()

def get_color(surf_loc):
    return 'red' if surf_loc == 'EXT' else 'blue'

# Assign colors based on surface location before filtering
if "surf. loc." in df_corrosion.columns:
    df_corrosion["Color"] = df_corrosion["surf. loc."].apply(get_color)
else:
    print("⚠️ Warning: 'surf. loc.' column missing. Assigning default color.")
    df_corrosion["Color"] = "gray"

# Define filtering thresholds
ERF_THRESHOLD = 0.75
WL_THRESHOLD = 40

# Assign ERF and wl [%] directly from df
if "ERF" not in df.columns:
    print("⚠️ Warning: ERF column not found in df. Assigning NaN.")
    df["ERF"] = np.nan
if "wl [%]" not in df.columns:
    print("⚠️ Warning: wl [%] column not found in df. Assigning NaN.")
    df["wl [%]"] = np.nan

# Filter only corrosion anomalies
columns_needed = ["L", "H", "ERF", "wl [%]", "anom. type/ident", "surf. loc.", "Circumference Degrees"]
df_corrosion = df.loc[df["anom. type/ident"].isin(["Anomaly  / Corrosion", "Anomaly  / Corrosion cluster"]), columns_needed].copy()

# Assign colors again to filtered datasets
df_corrosion["Color"] = df_corrosion["surf. loc."].apply(get_color)
df_internal = df_corrosion[df_corrosion["surf. loc."] == "INT"].copy()
df_external = df_corrosion[df_corrosion["surf. loc."] == "EXT"].copy()
df_erf_internal = df_internal[df_internal["ERF"] > ERF_THRESHOLD].copy()
df_wl_internal = df_internal[df_internal["wl [%]"] >= WL_THRESHOLD].copy()
df_erf_external = df_external[df_external["ERF"] > ERF_THRESHOLD].copy()
df_wl_external = df_external[df_external["wl [%]"] >= WL_THRESHOLD].copy()

for df_subset in [df_internal, df_external, df_erf_internal, df_wl_internal, df_erf_external, df_wl_external]:
    df_subset["Color"] = df_subset["surf. loc."].apply(get_color)

# Function to plot anomalies
def plot_anomalies(df, title1, title2):
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # Circumferential Location vs Pipeline Length
    axes[0].scatter(df["L"], df["Circumference Degrees"], c=df["Color"], alpha=0.75, s=50, edgecolors='black')
    axes[0].set_ylabel("Circumferential Location (Degrees)")
    axes[0].set_title(title1)
    axes[0].set_yticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
    axes[0].set_ylim(-10, 370)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Pipeline Profile with Anomalies
    axes[1].plot(df_corrosion["L"], df_corrosion["H"], color='gray', linewidth=1, linestyle='solid', label="Pipeline Profile")
    axes[1].scatter(df["L"], df["H"], c=df["Color"], alpha=0.75, s=50, edgecolors='black', label="Anomalies")
    axes[1].set_xlabel("Pipeline Length (m)")
    axes[1].set_ylabel("Pipeline Elevation (m)")
    axes[1].set_title(title2)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Add the logo outside the plot area, top right
    logo_ax = fig.add_axes([0.8, 0.88, 0.1, 0.1], anchor='NE', zorder=10)
    logo_ax.imshow(logo)
    logo_ax.axis('off')

    plt.show()


# Function to plot cumulative anomalies + histogram (Updated)
def plot_cumulative_anomalies(df, title, label, color, bin_width=3000):
    if df.empty or df["L"].isnull().all():
        print("⚠️ No data available to plot cumulative anomalies.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 3.5))  # ✅ Reduced figure size

    # Sort and compute cumulative count
    df_sorted = df[df["L"].notnull()].sort_values(by="L").copy()
    df_sorted["Cumulative Count"] = np.arange(1, len(df_sorted) + 1)

    # ✅ Plot cumulative count (primary y-axis)
    ax1.plot(df_sorted["L"], df_sorted["Cumulative Count"], label=label, 
             color=color, linewidth=2.5, zorder=2)

    ax1.set_xlabel("Pipeline Length (m)")
    ax1.set_ylabel("Cumulative Count", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.3)  # ✅ Lighter grid lines
    ax1.set_title(title, fontsize=12, pad=15)

    # Secondary y-axis: Histogram
    ax2 = ax1.twinx()

    # ✅ Adjusted bin width for cleaner histogram
    bins = np.arange(df_sorted["L"].min(), df_sorted["L"].max() + bin_width, bin_width)

    # ✅ Softer histogram color with gray tones
    counts, bin_edges, patches = ax2.hist(df_sorted["L"], bins=bins, 
                                          color='#555555', alpha=0.7, 
                                          edgecolor='black', linewidth=1.0, 
                                          label="Anomalies per Segment")

    # ✅ Add count labels inside bars in gray
    for rect, count in zip(patches, counts):
        if count > 0:  # Only label non-zero bars
            ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() - 1, 
                     f"{int(count)}", ha='center', va='bottom', fontsize=9, color="gray")

    # ✅ Labels and formatting
    ax2.set_ylabel("Anomalies per Segment", color='black', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=10)
    ax2.grid(True, linestyle="--", linewidth=0.3, alpha=0.3)  # ✅ Softer minor grid

    # ✅ Legends
    # ✅ Collect legend handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()  # Cumulative plot legend
    handles2, labels2 = ax2.get_legend_handles_labels()  # Histogram legend

    # ✅ Merge both legends and place in a single location
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=10, frameon=True)


    # ✅ Save the figure
    output_path = os.path.join(output_folder, 'Cumulative_Anomalies.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f'Plot saved at: {output_path}')

    plt.show()


# Generate updated plots with pipeline profile
plot_anomalies(df_corrosion, "All Anomalies - Circumferential Location", "All Anomalies - Pipeline Profile")
plot_anomalies(df_erf_external, f"External Anomalies (ERF > {ERF_THRESHOLD}) - Circumferential Location", f"External Anomalies (ERF > {ERF_THRESHOLD}) - Pipeline Profile")
plot_anomalies(df_wl_external, f"External Anomalies (WL [%] >= {WL_THRESHOLD}) - Circumferential Location", f"External Anomalies (WL [%] >= {WL_THRESHOLD}) - Pipeline Profile")
plot_anomalies(df_erf_internal, f"Internal Anomalies (ERF > {ERF_THRESHOLD}) - Circumferential Location", f"Internal Anomalies (ERF > {ERF_THRESHOLD}) - Pipeline Profile")
plot_anomalies(df_wl_internal, f"Internal Anomalies (WL [%] >= {WL_THRESHOLD}) - Circumferential Location", f"Internal Anomalies (WL [%] >= {WL_THRESHOLD}) - Pipeline Profile")

# -----------------------------
# GENERATE UPDATED PLOTS
# -----------------------------
plot_cumulative_anomalies(df_corrosion, "Cumulative Sum + Histogram of All Anomalies", "All Anomalies", "purple")
plot_cumulative_anomalies(df_external, "Cumulative Sum + Histogram of External Anomalies", "External Anomalies", "red")
plot_cumulative_anomalies(df_internal, "Cumulative Sum + Histogram of Internal Anomalies", "Internal Anomalies", "blue")
plot_cumulative_anomalies(df_erf_external, f"Cumulative + Histogram (ERF > {ERF_THRESHOLD})", "External (ERF)", "darkred")
plot_cumulative_anomalies(df_wl_internal, f"Cumulative + Histogram (WL [%] >= {WL_THRESHOLD})", "Internal (WL)", "darkblue")




#########################################################################################################################
## Ploting addition Anomaly Types
########################################################################################################################


# ✅ Define Anomaly Categories (Merging Corrosion Types)
ANOMALY_GROUPS = {
    "Corrosion": ["Anomaly  / Corrosion", "Anomaly  / Corrosion cluster"],
    "Dent": ["Anomaly  / Dent"],
    "Girth Weld Anomaly": ["Anomaly  / Girth weld anomaly"],
    "Grinding": ["Anomaly  / Grinding"],
    "Lamination": ["Anomaly  / Lamination"],
    "Milling": ["Anomaly  / Milling"],
}

# ✅ Compute Global X Limits for Pipeline Profile Alignment
global_x_min = df["L"].min()
global_x_max = df["L"].max()
global_xlim = [global_x_min, global_x_max]  # ✅ Set a fixed X-axis range for all plots


# -----------------------------
# FUNCTION TO PLOT MAP (GEOGRAPHIC VIEW)
# -----------------------------
def plot_anomaly_map(df, gdf_route, anomaly_label, anomaly_types):
    """ Plot the pipeline route map with grouped anomalies """
    # ✅ Filter anomalies that match the given types
    df_anomaly = df[df["anom. type/ident"].isin(anomaly_types)].copy()
    df_anomaly = df_anomaly.dropna(subset=["long", "lat"])
    df_anomaly["long"] = pd.to_numeric(df_anomaly["long"], errors="coerce")
    df_anomaly["lat"] = pd.to_numeric(df_anomaly["lat"], errors="coerce")
    df_anomaly = df_anomaly.dropna(subset=["long", "lat"])

    if df_anomaly.empty:
        print(f"⚠️ No valid anomalies found for: {anomaly_label}")
        return

    # ✅ Convert anomalies to GeoDataFrame
    gdf_anomaly = gpd.GeoDataFrame(df_anomaly, geometry=gpd.points_from_xy(df_anomaly["long"], df_anomaly["lat"]), crs="EPSG:4326")

    # ✅ Convert both route and anomalies to EPSG:3857 for plotting
    gdf_route_3857 = gdf_route.to_crs(epsg=3857)
    gdf_anomaly_3857 = gdf_anomaly.to_crs(epsg=3857)

    # ✅ Create a Separate Figure for the Map
    fig, ax_map = plt.subplots(figsize=(12, 12))  # ✅ Adjusted to square for better visualization

    # ✅ Plot the Pipeline Route and Anomalies
    gdf_route_3857.plot(ax=ax_map, color='cyan', linewidth=2, label='Pipeline Route')
    gdf_anomaly_3857.plot(ax=ax_map, color='red', markersize=150, alpha=0.9, 
                           edgecolor='yellow', marker='o', zorder=3, label='Anomalies')

    # ✅ Get Bounds and Expand Only the Y-Axis
    bounds = gdf_route_3857.total_bounds
    expand_y = (bounds[3] - bounds[1]) * 2  # ✅ Expanding Y by 200%

    # ✅ Apply Limits (Keep X Fixed, Expand Y)
    ax_map.set_xlim(bounds[0], bounds[2])  
    ax_map.set_ylim(bounds[1] - expand_y, bounds[3] + expand_y)  

    # ✅ Add Basemap AFTER modifying limits
    ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, crs=gdf_route_3857.crs.to_string(), zorder=1)

    # ✅ Formatting
    ax_map.set_title(f"{anomaly_label} - Geographic Map View", fontsize=18)
    ax_map.set_xlabel("Easting (m)")
    ax_map.set_ylabel("Northing (m)")
    ax_map.grid(True, linestyle='--', alpha=0.4)
    ax_map.legend()

    # ✅ Save the Map
    plt.savefig(os.path.join(output_folder, f"Map_{anomaly_label.replace(' ', '_')}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# -----------------------------
# FUNCTION TO PLOT PROFILE (PIPELINE ELEVATION)
# -----------------------------
def plot_anomaly_profile(df, anomaly_label, anomaly_types):
    """ Plot the pipeline elevation profile with grouped anomalies """
    # ✅ Filter anomalies that match the given types
    df_anomaly = df[df["anom. type/ident"].isin(anomaly_types)].copy()

    if df_anomaly.empty:
        print(f"⚠️ No valid anomalies found for: {anomaly_label}")
        return

    # ✅ Create a Separate Figure for the Profile
    fig, ax_profile = plt.subplots(figsize=(12, 6))  # ✅ Standard pipeline profile size

    # ✅ Plot Pipeline Profile
    ax_profile.plot(df["L"], df["H"], color='gray', linewidth=2, label='Pipeline Profile')
    ax_profile.scatter(df_anomaly["L"], df_anomaly["H"], color='red', s=80, 
                       edgecolors='black', label='Anomalies')
    
    # ✅ Fill the area below the pipeline (terrain)
    ax_profile.fill_between(df["L"], df["H"], df["H"].min() - 10, color="lightgray", alpha=0.5, label="Terrain")

    # ✅ Formatting
    ax_profile.set_title(f"{anomaly_label} - Elevation Profile", fontsize=16)
    ax_profile.set_xlabel("Pipeline Length (m)")
    ax_profile.set_ylabel("Elevation (H)")

    # ✅ Ensure Both Plots Have the Same X-Limits
    ax_profile.set_xlim(global_xlim)  # ✅ Aligns profile with the map

    # ✅ Grid and Formatting
    ax_profile.set_xticks(np.linspace(global_x_min, global_x_max, 10))  
    ax_profile.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))
    ax_profile.grid(True, linestyle="--", linewidth=1, alpha=0.7)

    ax_profile.set_facecolor("white")
    ax_profile.legend()

    # ✅ Save the Profile
    plt.savefig(os.path.join(output_folder, f"Profile_{anomaly_label.replace(' ', '_')}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# -----------------------------
# EXECUTE FUNCTIONS FOR EACH ANOMALY TYPE
# -----------------------------
for anomaly_label, anomaly_types in ANOMALY_GROUPS.items():
    plot_anomaly_map(df, gdf_route, anomaly_label, anomaly_types)  # Generates a separate map plot
    plot_anomaly_profile(df, anomaly_label, anomaly_types)         # Generates a separate profile plot




###############################################################################################################################
# Ploting The Quantiti and Position In the Circunferencial Location



# --------------------------
# Define Bin Sizes
# --------------------------
LENGTH_BIN_SIZE = 50     # in meters
DEG_BIN_SIZE = 30        # in degrees

# --------------------------
# Ensure Required Columns Exist
# --------------------------
required_columns = ["L", "Circumference Degrees", "anom. type/ident"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise KeyError(f"Missing required columns: {missing_columns}. Ensure the dataset has all necessary fields.")

# --------------------------
# Binning: Length & Circumference
# --------------------------
df["Length Bin"] = df["L"].floordiv(LENGTH_BIN_SIZE) * LENGTH_BIN_SIZE
df["Circ. Degree Bin"] = df["Circumference Degrees"].floordiv(DEG_BIN_SIZE) * DEG_BIN_SIZE

# --------------------------
# Merge Corrosion and Corrosion Cluster into One Category
# --------------------------
df["anomaly_group"] = df["anom. type/ident"].replace(
    {"Anomaly  / Corrosion": "Corrosion", "Anomaly  / Corrosion cluster": "Corrosion"}
)

# --------------------------
# Define List of Anomaly Types to Plot
# --------------------------
anomaly_types = [
    "Corrosion",
    "Anomaly  / Dent",
    "Anomaly  / Girth weld anomaly",
    "Anomaly  / Grinding",
    "Anomaly  / Lamination",
    "Anomaly  / Milling"
]

# --------------------------
# Create Folder for Saving Graphs
# --------------------------
# --------------------------
# Process and Plot Each Anomaly Type in a Separate Figure
# --------------------------
for anomaly_type in anomaly_types:
    df_anomaly = df[df["anomaly_group"] == anomaly_type].copy()

    if df_anomaly.empty:
        print(f"No data for {anomaly_type}, skipping plot.")
        continue

    # Aggregate anomalies by circumferential bins
    anomaly_agg = df_anomaly.groupby("Circ. Degree Bin").size().reset_index(name="Anomaly Count")

    # Ensure full 0-360 coverage even if some bins are empty
    full_bins = pd.DataFrame({"Circ. Degree Bin": np.arange(0, 360, DEG_BIN_SIZE)})
    anomaly_agg = pd.merge(full_bins, anomaly_agg, on="Circ. Degree Bin", how="left").fillna(0)
    anomaly_agg["Anomaly Count"] = anomaly_agg["Anomaly Count"].astype(int)

    # Convert bins to radians
    angles_rad = np.deg2rad(anomaly_agg["Circ. Degree Bin"])
    anomaly_counts = anomaly_agg["Anomaly Count"]

    # Create a separate figure for each anomaly type
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

    # Plot
    bars = ax.bar(angles_rad, anomaly_counts, width=np.deg2rad(DEG_BIN_SIZE), bottom=0,
                  color='darkorange', edgecolor='black', alpha=0.85)

    # Add Labels Inside Bars
    for bar, count in zip(bars, anomaly_counts):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height - 2, str(count),
                    ha='center', va='center', fontsize=10, color='gray', fontweight='bold')

    # Formatting
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{d}°" for d in np.arange(0, 360, 30)])
    ax.set_yticklabels([])

    # Place Title Outside the Plot
    plt.title(f"Radial Distribution of {anomaly_type}", fontsize=14, pad=20)
    plt.show()

    # # Save the figure
    # file_name = os.path.join(output_folder, f"{anomaly_type.replace(' ', '_')}.png")
    # plt.savefig(file_name, dpi=300, bbox_inches="tight")
    # plt.close(fig)

print("All graphs saved successfully in 'PowerBI_Charts' folder.")


########################################################################################################################################
import pandas as pd
import folium
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from pyproj import Transformer
import os
import webbrowser
from shapely.geometry import LineString
import xml.etree.ElementTree as ET
import numpy as np

# Load datasets
Morken = input("Informe o Caminho do Arquivo Contendo as Anomalias:")
df_morken = pd.read_excel(Morken, sheet_name=None)
df_anomalies = pd.read_excel(r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\List of Anomalies_Rosen_UTM.xlsx")
kml_filepath = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\Rota Mineroduto.kml"

# Define anomaly groups
ANOMALY_GROUPS = {
    "Corrosion": ["Anomaly  / Corrosion", "Anomaly  / Corrosion cluster"],
    "Dent": ["Anomaly  / Dent"],
    "Girth Weld Anomaly": ["Anomaly  / Girth weld anomaly"],
    "Grinding": ["Anomaly  / Grinding"],
    "Lamination": ["Anomaly  / Lamination"],
    "Milling": ["Anomaly  / Milling"],
}

# Function to load the pipeline route from KML
def load_kml_route(kml_file):
    with open(kml_file, 'r', encoding='utf-8') as f:
        kml_content = f.read()
    
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    root = ET.fromstring(kml_content)
    
    for placemark in root.findall(".//kml:Placemark", namespace):
        line_string = placemark.find(".//kml:LineString", namespace)
        if line_string is not None:
            coordinates = line_string.find(".//kml:coordinates", namespace).text.strip()
            coord_list = []
            for coord in coordinates.split():
                lon, lat, *_ = map(float, coord.split(','))
                coord_list.append((lon, lat))
            return LineString(coord_list)  # Convert to Shapely LineString
    
    raise ValueError("❌ No LineString pipeline route found in the KML file.")

# Load pipeline route
gdf_route = gpd.GeoDataFrame(geometry=[load_kml_route(kml_filepath)], crs="EPSG:4326")

# Convert "wl [%]" to numeric
df_anomalies["wl [%]"] = pd.to_numeric(df_anomalies["wl [%]"], errors="coerce")

# Filter anomalies based on type and depth
df_anomalies_filtered = df_anomalies.copy()

# Apply depth filter only for corrosion anomalies
df_anomalies_filtered = df_anomalies_filtered[
    (~df_anomalies_filtered["anom. type/ident"].isin(ANOMALY_GROUPS["Corrosion"])) | 
    (df_anomalies_filtered["anom. type/ident"].isin(ANOMALY_GROUPS["Corrosion"]) & (df_anomalies_filtered["wl [%]"] >= 35))
]

# Ensure valid coordinates
df_anomalies_filtered = df_anomalies_filtered.dropna(subset=["long", "lat"])
df_anomalies_filtered["long"] = pd.to_numeric(df_anomalies_filtered["long"], errors="coerce")
df_anomalies_filtered["lat"] = pd.to_numeric(df_anomalies_filtered["lat"], errors="coerce")

# Convert filtered DataFrame to GeoDataFrame
gdf_anomalies_filtered = gpd.GeoDataFrame(
    df_anomalies_filtered, geometry=gpd.points_from_xy(df_anomalies_filtered["long"], df_anomalies_filtered["lat"]), crs="EPSG:4326"
)

# Convert Morken anomalies into GeoDataFrame
if 'Leste' in df_morken and 'Norte' in df_morken:
    df_morken["long"] = pd.to_numeric(df_morken["Leste"], errors="coerce")
    df_morken["lat"] = pd.to_numeric(df_morken["Norte"], errors="coerce")
    df_morken = df_morken.dropna(subset=["long", "lat"])
    gdf_morken = gpd.GeoDataFrame(df_morken, geometry=gpd.points_from_xy(df_morken["long"], df_morken["lat"]), crs="EPSG:4326")
else:
    gdf_morken = None

# Convert to Web Mercator for basemap
gdf_route_3857 = gdf_route.to_crs(epsg=3857)
gdf_anomalies_3857 = gdf_anomalies_filtered.to_crs(epsg=3857)
if gdf_morken is not None:
    gdf_morken_3857 = gdf_morken.to_crs(epsg=3857)

# Create figure with subplots
fig, ax = plt.subplots(figsize=(12, 12))

# Plot Pipeline Route
gdf_route_3857.plot(ax=ax, color='cyan', linewidth=1.5, label="Pipeline Route")

# Plot anomalies
if not gdf_anomalies_3857.empty:
    gdf_anomalies_3857.plot(ax=ax, color='red', marker='o', markersize=50, alpha=0.8, edgecolor='black', label="Anomalies")

# Plot Morken anomalies if available
if gdf_morken is not None and not gdf_morken.empty:
    gdf_morken_3857.plot(ax=ax, color='magenta', marker='x', markersize=60, alpha=0.9, edgecolor='black', label="Morken Anomalies")

# Expand map area
bounds = gdf_anomalies_3857.total_bounds
expand_y = (bounds[3] - bounds[1]) * 2
ax.set_xlim(bounds[0], bounds[2])
ax.set_ylim(bounds[1] - expand_y, bounds[3] + expand_y)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs='EPSG:3857')
ax.set_title("Pipeline Route with Anomalies")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

# Save and Show
plt.savefig("Pipeline_Map_Comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()