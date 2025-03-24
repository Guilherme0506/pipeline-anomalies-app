import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import cv2

# Set the path for FFMpeg
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

# Parameters
L = 600  # Comprimento do tubo
Nx = 15  # Número de nós no tubotf
Nt = 550
dx = L / (Nx - 1)
tf = 20  # Tempo total de simulação
dt = tf / (Nt - 1)
tav = 2.1  # tempo de abertura válvula
Io = 0.0  # Parte de F, que representa a declividade do conduto.
g = 9.8  # Aceleração da gravidade
E = 200e9  # Módulo de elasticidade do material do tubo
K = 1.75e9  # Módulo de elasticidade do fluido
D = 0.50  # Diâmetro do tubo
esp = 20 / 1000  # Espessura da parede da tubulação
A = np.pi * 0.25 * D**2  # Área da seção transversal do conduto
f = 0.018  # Fator de resistência de Darcy-Weisbach
rho = 1000  # Massa específica do líquido
k = 1  # Coeficiente que depende da relação de

# Cálculo da Celeridade das Ondas de Pressão:
a = np.sqrt((K / rho) / (1 + K * D * k / (E * esp)))
dxa = dx / a
Courant = dt * a / dx

print(Courant)

# Check stability criterion
if dt > dxa:
    print('Stability criterion violated')

# Initialize arrays for pressure and flow
x = np.zeros(Nx)
Q = np.zeros((Nx, Nt))
H = np.zeros((Nx, Nt))
Cmais = np.zeros((Nx, Nt))
Cmenos = np.zeros((Nx, Nt))

B = a / (g * A)
R = f * dx / (2 * g * D * A**2)

# Initial conditions - Steady state
zr = 150  # Cota do nível d'água no reservatório em relação ao eixo do tubo
CdA0 = 0.009  # Produto entre o coeficiente de vazão e a área para t=0;
Q0=np.sqrt(2*g*CdA0*CdA0*zr/(R*L*2*g*CdA0*CdA0+1))
V0 = Q0 / A
n=0

H[0,:] = zr  # H em x=1 e para todos os instantes

# Cálculo Linha Piezométrica em Regime Permanente e plotagem
for i in range(1,Nx):
    x[i] = i * dx
    Q[i, :] = Q0
    H[i, :] = zr - ((V0**2) / (2 * g )) * (f * x[i] / D)

plt.figure(1)
plt.plot(x, H[:, 0])
plt.show()

CVP=0.5*Q0**2/(H[Nx-1,0])
print(CVP)


T=0

# Cálculo do Transiente Hidraulico
for t in np.arange(0, tf, dt):  # Adjusted to ensure n does not exceed Nt-1
    n+=1
    T += dt


    for i in range(1, Nx - 1):
        Cmenos[i, n] = H[i + 1, n-1] - B * Q[i + 1, n-1] + R * Q[i + 1, n-1] * abs(Q[i + 1, n-1])
        Cmais[i, n] = H[i - 1, n-1] + B * Q[i - 1, n-1] - R * Q[i - 1, n-1] * abs(Q[i - 1, n-1])
        H[i, n] = 0.5 * (Cmais[i, n] + Cmenos[i, n])
        Q[i, n] = (H[i, n] - Cmenos[i, n]) / B

        
    if T <= tav:
        TAU=(1 - T / tav)**1.5
        CV=CVP*TAU*TAU
        Q[Nx-1, n] = - CV * B + np.sqrt(CV*CV*B*B + 2*CV*Cmais[Nx-2, n])
        H[Nx - 1, n] = Cmais[Nx - 2, n] - B * Q[Nx - 1, n]
    else:
        TAU=0
        CV=CVP*TAU*TAU
        Q[Nx -1,  n] = - CV * B + np.sqrt(CV*CV*B*B + 2*CV*Cmais[Nx-2, n])
        H[Nx - 1, n] = Cmais[Nx - 2, n] - B * Q[Nx - 1, n]

    # Reservatório
    H[0, n] = zr
    Q[0, n] = (H[0, n] - Cmenos[1, n]) / B



# Visualização dos Resultados:
plt.show()

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
    plt.plot(x[1:Nx], H[1:Nx, n] / np.max(H), label=f'H/Hmax (t={n*dt:.2f}s)')
    plt.plot(x[1:Nx], Q[1:Nx, n] / np.max(Q), label=f'Q/Qmax (t={n*dt:.2f}s)')
    plt.xlabel('Position (x)')
    plt.ylabel('Normalized H and Q')
    plt.xlim([0, L])
    plt.ylim([-1, 1])
    plt.legend(loc='upper right')


# Envelope with maximum H values
# Envelope with maximum H values
MaxH = np.zeros(Nx-1)
MinH = np.zeros(Nx-1)

for i in range(1, Nx):
    MaxH[i-1] = np.max(H[i, :])
    MinH[i-1] = np.min(H[i, :])

plt.subplot(2, 2, 3)
plt.plot(x[1:Nx], MaxH)
plt.xlabel('Position (x)')
plt.ylabel('max(H(t))')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x[1:Nx], MinH)
plt.xlabel('Position (x)')
plt.ylabel('min(H(t))')
plt.grid(True)

plt.show()

time_steps_to_plot = list(range(0, Nt, 500))

plt.figure(figsize=(12, 6))
for t in time_steps_to_plot:
    plt.plot(x[1:], H[1:, t], label=f'Time step {t}')

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
    for n in range(0, Nt, 1):
        plt.clf()  # Clear the current figure
        
        # Plot the data
        plt.plot(x[0:Nx], H[0:Nx, n])
        plt.xlabel('x [m]')
        plt.ylabel('H [mca]')
        plt.grid(True)
        plt.xlim([0, L])
        plt.ylim([np.nanmin(H), np.nanmax(H)])  # Use nanmin and nanmax to avoid NaN/Inf issues
        
        # Capture and write frame
        writer.grab_frame()

plt.close()
