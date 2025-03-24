import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from PIL import Image
import os
import seaborn as sns
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os




Qd = int(input("Informe a Vazão Desejada [m³/j]:"))
Q=np.arange(20,Qd*10,1)
n1=int(input("informe a quantidade de diâmetros no trajeto:"))  # Quantidade de diâmetros
n2=int(len(Q))                                                  # Vazão
n3=4                                                            # Percentual de Sólidos
n4=4                                                            # Frações de partículas
cvpf=np.zeros((n4))
cw=np.zeros((n3,n2))
veh=np.zeros((n1,n2,n3))
veht=np.zeros((n1,n2,n3))
bed=np.zeros((n1,n2,n3))
ibed=np.zeros((n1,n2,n3))
iph= np.zeros((n1,n2,n3))
iw=np.zeros((n1,n2))
mif=np.zeros((n3))
vr=np.zeros((n3))
vr_n=np.zeros((n1,n2,n3))
cv=np.zeros((n3))
cw=np.zeros(n3)
rhoveh=np.zeros((n1,n2,n3))
durand=np.zeros((n1,n2,n3,n4))
fd=np.zeros((n1,n2,n3))
vtpf=np.zeros((n1,n2,n3,n4))
Rep=np.zeros((n1,n2,n3,n4))
cd=np.zeros((n1,n2,n3,n4))
CCA=np.zeros((n1,n2,n3,n4))
rhof=np.zeros((n3))
dint=np.zeros((n1))
L=np.zeros((n1))
LA=np.zeros((n1+1))
Aint=np.zeros((n1))
velf=np.zeros((n1,n2))
Rey=np.zeros((n1,n2))
Reyveh=np.zeros((n1,n2,n3))
Reyp=np.zeros((n1,n2,n3))
unp=np.zeros((n1,n2,n3))
fp=np.zeros((n1,n2,n3))
f=np.zeros((n1,n2))
Reynveh=np.zeros((n3,n2))
phi=np.zeros((n3))
phi_n=np.zeros((n1,n2,n3))
miveh=np.zeros((n1,n2,n3))
fveh=np.zeros((n1,n2,n3))
iveh=np.zeros((n1,n2,n3))
iwasp=np.zeros((n1,n2,n3))
iwaspw=np.zeros((n1,n2,n3))
iwaspn=np.zeros((n1,n3))
iwaspn_Q1=np.zeros((n1,n3))
iwaspn_Q2=np.zeros((n1,n3))
iwaspn_Q3=np.zeros((n1,n3))
iwaspn_Q4=np.zeros((n1,n3))
dH=np.zeros((n1,n3))
dH_Q1=np.zeros((n1,n3))
dH_Q2=np.zeros((n1,n3))
dH_Q3=np.zeros((n1,n3))
dH_Q4=np.zeros((n1,n3))
dHT=np.zeros((n3))
dHT_Q1=np.zeros((n3))
dHT_Q2=np.zeros((n3))
dHT_Q3=np.zeros((n3))
dHT_Q4=np.zeros((n3))
iwn=np.zeros((n1))
iwn_Q1=np.zeros((n1))
iwn_Q2=np.zeros((n1))
iwn_Q3=np.zeros((n1))
iwn_Q4=np.zeros((n1))

dHw=np.zeros((n1))
dHw_Q1=np.zeros((n1))
dHw_Q2=np.zeros((n1))
dHw_Q3=np.zeros((n1))
dHw_Q4=np.zeros((n1))
n=0
g=9.81
rho=1000                      # Densidade da água
mi=10**-3                     # Viscosidade da água
e=0.05                        # Rugosidade absoluta
ds=3200                       # Densidade dos sólidos
B=2.09                         # Parâmetro reológico
cw[0]=55
cw[1]=56
cw[2]=58
cw[3]=62

cvdp=[0.05,0.50,0.20,0.25]
d50=0.055
dpf=np.zeros(n4)
dpf[0]=d50*3
dpf[1]=d50
dpf[2]=d50*0.5
dpf[3]=d50*0.25
beta=1

for j in range (n1):
    diameter=input("Informe o diametro em [mm]:")
    dint[j]=int(float(diameter))
    Aint[j]=(np.pi*(dint[j]/1000)**2)/4
    L[j]=float(input("Informe o comprimento em [m]:"))

k=e/dint 
b=(k/3.7)**1.11

# Cálculo da perda de carga com água:

for i in range (n1):
    for j in range (n2):

       velf[i][j]=(Q[j]/3600)/Aint[i]
       Rey[i][j]=(velf[i][j]*(dint[i]/1000)*rho)/mi
       f[i][j]=(1/(-1.8*(np.log10(6.9/Rey[i][j] + b[i]))))**2
       iw[i][j]=(f[i][j]/(dint[i]/1000))*((velf[i][j])**2)/(2*9.81)

# Calculo pelo Método de Wasp:

# Passo 01: Densidade do vehicle (Tudo Formando o Vehicle)
for i in range(n1):
    for j in range (n2):
        for k in range (n3):
            rhof[k]=1000/(1-(cw[k]/100)*((ds-1000)/ds))
            cv[k]=((rhof[k]-1000)/(ds-1000))
            rhoveh[i,j,k]=rhof[k]

for i in range(n1):
    for j in range (n2):
        for k in range (n3):
            phi[k]=((rhof[k]-rho)/(ds-rho))
            vr[k]=phi[k]/(1-phi[k])
            mif[k]=(0.89*0.001*(10**(vr[k])*B))/rhof[k]
            miveh[i][j][k]=mif[k]

for i in range (n1):
    for j in range (n2):
        for k in range (n3):

            Reyp[i][j][k]=(velf[i][j]*(dint[i]/1000)*rhof[k])/mif[k]
            fp[i][j][k]=(1/(-1.8*(np.log10(6.9/Reyp[i][j][k] + b[i]))))**2
            iph[i][j][k]=(((fp[i][j][k]/(dint[i]/1000))*(velf[i][j]**2))/(2*g))
            unp[i][j][k]=velf[i][j]*np.sqrt((fp[i][j][k])/8)*(rhof[k]/rho)

# Passo 02: Correção (Iterativo)


for i in range(n1):
    for n in range(10):
        for j in range(n2):
            for k in range(n3):
                fd_temp = 0
                for m in range(n4):
                   
                    vtpf [i,j,k,m] = (((ds - rhoveh[i,j,k]) / rhoveh[i,j,k]) * ((dpf[m] / 1000)**2) * 9.81 / (18 * miveh[i][j][k]))
                    Rep  [i,j,k,m] = vtpf[i,j,k,m] * (dpf[m] / 1000) / miveh[i][j][k]
                    cd   [i,j,k,m] = (24 / Rep[i,j,k,m]) * ((1 + 0.173 * Rep[i,j,k,m])**0.657) + (0.413 / (1 + 16300 * Rep[i,j,k,m]**(-1.09)))
                    CCA  [i,j,k,m] = 10**(-1.8 * (vtpf[i,j,k,m] / (beta * 0.4 * unp[i,j,k])))
               
                    veht   [i,j,k] = (cvdp[0] * CCA[i,j,k,0] + cvdp[1] * CCA[i,j,k,1] + cvdp[2] * CCA[i,j,k,2] + cvdp[3] * CCA[i,j,k,3])*cv[k]
                    bed    [i,j,k] = 1 - veht[i,j,k]
                    rhoveh [i,j,k] = (veht[i,j,k] * ds / rho + (1 - veht[i,j,k])) * 1000
                    
                    durand [i,j,k,m] = (g * (dint[i] / 1000) * ((ds - rhoveh[i,j,k]) / rhoveh[i,j,k]) / ((velf[i,j]**2) * np.sqrt(cd[i,j,k,m])))**(3 / 2)

                    fd_temp=sum(durand[i,j,k,:])

                    fd[i,j,k]= bed[i, j, k] * 82 * cv[k] * fd_temp 

                ibed[i, j, k] = fd[i, j, k] * iw[i, j] # metros de coluna de água

                phi_n[i][j][k] = (rhoveh[i][j][k] - rho) / (ds - rho)
                vr_n[i][j][k] = phi_n[i][j][k] / (1 - phi_n[i][j][k])
                miveh[i][j][k] = (0.89 * 0.001 * (10**(vr_n[i][j][k] * B))) / (rhoveh[i][j][k])

                Reyveh[i][j][k] = (velf[i][j] * (dint[i] / 1000) * rhoveh[i][j][k]) / miveh[i][j][k]
                fveh[i][j][k] = (1 / (-1.8 * (np.log10(6.9 / Reyveh[i][j][k] + b[i]))))**2
                iveh[i][j][k] = (((fveh[i][j][k] / (dint[i] / 1000)) * (velf[i][j]**2) * (rhoveh[i][j][k] / rho)) / (2 * g))
                iwasp[i][j][k] = (iveh[i][j][k] + ibed[i][j][k]) / (rhoveh[i][j][k] / rho)                                                                         # metros de coluna de polpa
                unp[i][j][k] = velf[i][j] * np.sqrt((iwasp[i][j][k] * 9.81 * (dint[i] / 1000 / 4)))
    n=n+1
# Determinar Linh a Piezométrica

Q1=int(Qd*0.8)
Q2=int(Qd*0.9)
Q3=int(Qd*1.1)
Q4=int(Qd*1.2)

if Qd in Q:
    IQd = int(np.where(Q == Qd)[0][0])
    IQ1 = int(np.where(Q == Q1)[0][0])
    IQ2 = int(np.where(Q == Q2)[0][0])
    IQ3 = int(np.where(Q == Q3)[0][0])
    IQ4 = int(np.where(Q == Q4)[0][0])
   
    for i in range(n1):
        for j in range(n3):
            iwaspn   [i][j] = iwasp[i, IQd, j]
            iwaspn_Q1[i][j] = iwasp[i, IQ1, j]
            iwaspn_Q2[i][j] = iwasp[i, IQ2, j]
            iwaspn_Q3[i][j] = iwasp[i, IQ3, j]
            iwaspn_Q4[i][j] = iwasp[i, IQ4, j]

            iwn[i]          = iw   [i,IQd]
            iwn_Q1[i]       = iw   [i,IQ1]
            iwn_Q2[i]       = iw   [i,IQ2]
            iwn_Q3[i]       = iw   [i,IQ3]
            iwn_Q4[i]       = iw   [i,IQ4]
else:
    
    print("Vazão desejada não encontrada na matriz Q.")

#print(iwasp)
#print(iw)



for i in range(n1):
    for j in range (n3):
        dH[i][j]=iwaspn[i][j]*L[i]
        dH_Q1[i][j]=iwaspn_Q1[i][j]*L[i]
        dH_Q2[i][j]=iwaspn_Q2[i][j]*L[i]
        dH_Q3[i][j]=iwaspn_Q3[i][j]*L[i]
        dH_Q4[i][j]=iwaspn_Q4[i][j]*L[i]

        dHw[i]     =iwn   [i]*L[i]
        dHw_Q1[i]  =iwn_Q1[i]*L[i]
        dHw_Q2[i]  =iwn_Q2[i]*L[i]
        dHw_Q3[i]  =iwn_Q3[i]*L[i]
        dHw_Q4[i]  =iwn_Q4[i]*L[i]

for j in range(n3):
    dHT[j]=sum(dH[:,j])
    dHT_Q1[j]=sum(dH_Q1[:,j])
    dHT_Q2[j]=sum(dH_Q2[:,j])
    dHT_Q3[j]=sum(dH_Q3[:,j])
    dHT_Q4[j]=sum(dH_Q4[:,j])

dHTw=sum(dHw[:])
dHTw_Q1=sum(dHw_Q1[:])
dHTw_Q2=sum(dHw_Q2[:])
dHTw_Q3=sum(dHw_Q3[:])
dHTw_Q4=sum(dHw_Q4[:])

#print(dHw)
#print(dHTw)

LT=sum(L[:]) #comprimento Total

LA[0]=0

for i in range(1,(n1+1)):
    LA[i]=LA[i-1]+L[i-1]

#print(LA)
dHSm=dHT/LT
#print(dHSm)

# Importanto o Perfil do Mineroduto

RP=input("Informe o Caminho do Arquivo Contendo o Perfil do Rejeitoduto / Mineroduto:")
df=pd.read_excel(RP)

LR=df['L'].values
HR=df['H'].values

n5=int(len(LR))
n6=n5-1
LP=np.zeros((n5,n3))
PL=np.zeros((n5,n3))
LP_Q1=np.zeros((n5,n3))
PL_Q1=np.zeros((n5,n3))
LP_Q2=np.zeros((n5,n3))
PL_Q2=np.zeros((n5,n3))
LP_Q3=np.zeros((n5,n3))
PL_Q3=np.zeros((n5,n3))
LP_Q4=np.zeros((n5,n3))
PL_Q4=np.zeros((n5,n3))

LPw=np.zeros((n5,1))
PLw=np.zeros((n5,1))
LPw_Q1=np.zeros((n5,1))
PLw_Q1=np.zeros((n5,1))
LPw_Q2=np.zeros((n5,1))
PLw_Q2=np.zeros((n5,1))
LPw_Q3=np.zeros((n5,1))
PLw_Q3=np.zeros((n5,1))
LPw_Q4=np.zeros((n5,1))
PLw_Q4=np.zeros((n5,1))
HRmax=np.ones(n5)
HRrd=np.ones(n5)
PT=0
PEB=0

HRm=np.max(HR)*HRmax
HRrd=HRrd*1241

# Calcula a Linha Piezométrica
# Calcula a Linha Piezométrica Inicial

for j in range(n3):
    LP[0,j]=dHT[j]      +((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j]+HR[0])  
    LP_Q1[0,j]=dHT_Q1[j]+((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j]+HR[0])
    LP_Q2[0,j]=dHT_Q2[j]+((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j]+HR[0])
    LP_Q3[0,j]=dHT_Q3[j]+((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j]+HR[0])
    LP_Q4[0,j]=dHT_Q4[j]+((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j]+HR[0])

    LPw[0]=dHTw      +((HR[n6]-HR[0])+10*(PT-PEB)+HR[0])  
    LPw_Q1[0]=dHTw_Q1+((HR[n6]-HR[0])+10*(PT-PEB)+HR[0])
    LPw_Q2[0]=dHTw_Q2+((HR[n6]-HR[0])+10*(PT-PEB)+HR[0])
    LPw_Q3[0]=dHTw_Q3+((HR[n6]-HR[0])+10*(PT-PEB)+HR[0])
    LPw_Q4[0]=dHTw_Q4+((HR[n6]-HR[0])+10*(PT-PEB)+HR[0])

#    LPw[0]=dHTw+((1215-HR[0])+10*(PT-PEB)+HR[0])  
#    LPw_Q1[0]=dHTw_Q1+((1215-HR[0])+10*(PT-PEB)+HR[0])
#    LPw_Q2[0]=dHTw_Q2+((1215-HR[0])+10*(PT-PEB)+HR[0])
#    LPw_Q3[0]=dHTw_Q3+((1215-HR[0])+10*(PT-PEB)+HR[0])
#    LPw_Q4[0]=dHTw_Q4+((1215-HR[0])+10*(PT-PEB)+HR[0])

#print(LPw)
#print(iwn)

#CS=dHw[:]+(HR[n6]--HR[0])

#print (CS)

for k in range (n1):
    for j in range (1,n5):
        for i in range (n3):
            if LR[j] > LA[k] and LR[j]<=LA[k+1]:
                LP   [j][i]=   LP[j-1][i]-(LR[j]-LR[j-1])*   iwaspn[k][i]
                LP_Q1[j][i]=LP_Q1[j-1][i]-(LR[j]-LR[j-1])*iwaspn_Q1[k][i]
                LP_Q2[j][i]=LP_Q2[j-1][i]-(LR[j]-LR[j-1])*iwaspn_Q2[k][i]
                LP_Q3[j][i]=LP_Q3[j-1][i]-(LR[j]-LR[j-1])*iwaspn_Q3[k][i]
                LP_Q4[j][i]=LP_Q4[j-1][i]-(LR[j]-LR[j-1])*iwaspn_Q4[k][i]

                LPw   [j]=   LPw[j-1]-(LR[j]-LR[j-1])*   iwn[k]
                LPw_Q1[j]=LPw_Q1[j-1]-(LR[j]-LR[j-1])*iwn_Q1[k]
                LPw_Q2[j]=LPw_Q2[j-1]-(LR[j]-LR[j-1])*iwn_Q2[k]
                LPw_Q3[j]=LPw_Q3[j-1]-(LR[j]-LR[j-1])*iwn_Q3[k]
                LPw_Q4[j]=LPw_Q4[j-1]-(LR[j]-LR[j-1])*iwn_Q4[k]

            elif LR[j] > LA[k+1]:
                LPw   [j]=HR[j]
                LPw_Q1[j]=HR[j]
                LPw_Q2[j]=HR[j]
                LPw_Q3[j]=HR[j]
                LPw_Q4[j]=HR[j]

# Cacula a Pressão na Linha 
for j in range(n5):
    for i in range (n3):
        PL[j][i]=(LP[j][i] - HR[j])*(rhof[i]/rho)/10
        PL_Q1[j][i]=(LP_Q1[j][i] - HR[j])*(rhof[i]/rho)/10
        PL_Q2[j][i]=(LP_Q2[j][i] - HR[j])*(rhof[i]/rho)/10
        PL_Q3[j][i]=(LP_Q3[j][i] - HR[j])*(rhof[i]/rho)/10
        PL_Q4[j][i]=(LP_Q3[j][i] - HR[j])*(rhof[i]/rho)/10

        PLw   [j]=(LPw   [j] - HR[j])/10
        PLw_Q1[j]=(LPw_Q1[j] - HR[j])/10
        PLw_Q2[j]=(LPw_Q2[j] - HR[j])/10
        PLw_Q3[j]=(LPw_Q3[j] - HR[j])/10
        PLw_Q4[j]=(LPw_Q3[j] - HR[j])/10


# Create subplots: one for loglog scale and one for normal scale
static_head=np.ones(len(HR))
static_head = static_head*(max(HR))
rupture_disk=1241

# Gráficos
##########################################################################################################################################################################################
# Calculo da media ponderada das perdas de cargas na tubulação
iwaspmp=np.zeros((n2,n3))
iwp=np.zeros((n2))
iwp[:]=(iw[0,:]*L[0]+iw[1,:]*L[1]+iw[2,:]*L[2]+iw[3,:]*L[3])/(L[0]+L[1]+L[2]+L[3])


iwaspmp[:,0]=((iwasp[0,:,0]*L[0]+ iwasp[1,:,0]*L[1]+ iwasp[2,:,0]*L[2]+iwasp[3,:,0]*L[3]))*1000/(L[0]+L[1]+L[2]+L[3])   #cv1
iwaspmp[:,1]=((iwasp[0,:,1]*L[0]+ iwasp[1,:,1]*L[1]+ iwasp[2,:,1]*L[2]+iwasp[3,:,1]*L[3]))*1000/(L[0]+L[1]+L[2]+L[3])   #cv2 
iwaspmp[:,2]=((iwasp[0,:,2]*L[0]+ iwasp[1,:,2]*L[1]+ iwasp[2,:,2]*L[2]+iwasp[3,:,2]*L[3]))*1000/(L[0]+L[1]+L[2]+L[3])   #cv3
iwaspmp[:,3]=((iwasp[0,:,3]*L[0]+ iwasp[1,:,3]*L[1]+ iwasp[2,:,3]*L[2]+iwasp[3,:,3]*L[3]))*1000/(L[0]+L[1]+L[2]+L[3])   #cv4


##########################################################################################################################################################################################

# Create output directory
output_folder = "Gráficos_Regime_Permanente"
os.makedirs(output_folder, exist_ok=True)

########################################################################################################################################################################################


# Create subplots: one for loglog scale and one for normal scale
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
# LogLog plot
for j in range(n3):
    axes[0].loglog(velf[0,:], iwaspmp[:, j], linestyle='-', label=f'cw {cw[j]}')
axes[0].loglog(velf[0,:], 1000*iwp, linestyle='--', color='black', label='Água [Colebrook]')
axes[0].set_xlabel('Velociade do Fluxo [m/s]')
axes[0].set_ylabel('Perda de Carga Unitária [m/Km]')
axes[0].set_title('Gráfico da Perda de Carga (Escala Log-Log)')
axes[0].set_xlim(0.1, 10)
axes[0].legend()
axes[0].grid(True, which="both", linestyle='--')

# Normal plot
for j in range(n3):
    axes[1].plot(velf[0,:], iwaspmp[:, j], linestyle='-', label=f'cw {cw[j]}')
axes[1].plot(velf[0,:], 1000*iwp, linestyle='--', color='black', label='Água [Colebrook]')
axes[1].set_xlabel('')
axes[1].set_ylabel('Perda de Carga Unitária [m/Km]')
axes[1].set_title('Gráfico da Perda de Carga ')
axes[1].set_xlim(0.1,10)
axes[1].set_ylim(0,250)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()


# Save the figure automatically
fig.savefig(os.path.join(output_folder, "Curva J.png"), dpi=300, bbox_inches='tight')
plt.show()


#######################################################################################################################################################################################

# # Gráifico 01: Curva J e Curva J em LogLog Scale
# fig1, ax=plt.subplots(int(2),int(n1))
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
# ax=ax.flatten()
# for j in range (n1):
#     ax[j].loglog(velf[0, :], 1000 * iw[j, :], label=f'Cw:0.0 %')
#     ax[j].loglog(velf[0, :], 1000 * iwasp[j, :, 0], label=f'Cw: {cw[0]} %')
#     ax[j].loglog(velf[0, :], 1000 * iwasp[j, :, 1], label=f'Cw: {cw[1]} %')
#     ax[j].loglog(velf[0, :], 1000 * iwasp[j, :, 2], label=f'Cw: {cw[2]} %')
#     ax[j].loglog(velf[0, :], 1000 * iwasp[j, :, 3], label=f'Cw: {cw[3]} %')
#     ax[j].set_xlabel('Velocidade [m/s]')
#     ax[j].set_ylabel('Perda de Carga [m/Km]')
#     ax[j].set_title(f'Curva J Diâmetro: {dint[j]} [mm]')
#     ax[j].minorticks_on()
#     ax[j].grid(which='minor', linestyle=':', linewidth=0.5, color='grey')
#     ax[j].legend(loc='upper center', ncol=5,bbox_to_anchor=(0.5, 1.25))

# for j in range (n1,n1*2):
#     ax[j].plot(velf[0, :], 1000 * iw[0, :], label=f'Cw:0.0 %')
#     ax[j].plot(velf[0, :], 1000 * iwasp[0, :, 0], label=f'Cw: {cw[0]} %')
#     ax[j].plot(velf[0, :], 1000 * iwasp[0, :, 1], label=f'Cw: {cw[1]} %')
#     ax[j].plot(velf[0, :], 1000 * iwasp[0, :, 2], label=f'Cw: {cw[2]} %')
#     ax[j].plot(velf[0, :], 1000 * iwasp[0, :, 3], label=f'Cw: {cw[3]} %')
#     ax[j].set_xlabel('Velocidade [m/s]')
#     ax[j].set_ylabel('Perda de Carga [m/Km]')
#     ax[j].set_title(f'Curva J Diâmetro: {dint[0]} [mm]')
#     ax[j].minorticks_on()
#     ax[j].grid(which='minor', linestyle=':', linewidth=0.5, color='grey')
#     ax[j].legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.28))

# Gráifico 02: Linhas Piezométricas para Diferentes Cw[%] e Varaiando-se a Vazão:
# fig2, axs1=plt.subplots(int(n3/2),int(n3/2), figsize=(8,6))
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
# axs1=axs1.flatten()
# for j in range (0,n3,1):
#     axs1[j].plot(LR, HR, label=f'Perfil do Terreno')  
#     axs1[j].plot(LR, LP[:, j], label=f'Vazão: {Qd} [m³/h]')  
#     axs1[j].plot(LR, LP_Q1[:, j], label=f'Vazão: {Q1} [m³/h]')   
#     axs1[j].plot(LR, LP_Q2[:, j], label=f'Vazão: {Q2} [m³/h]')   
#     axs1[j].plot(LR, LP_Q3[:, j], label=f'Vazão: {Q3} [m³/h]') 
#     axs1[j].plot(LR, LP_Q4[:, j], label=f'Vazão: {Q4} [m³/h]')  
#     axs1[j].plot(LR, HRm[:], label='Linha Estática [Altura Manométrica]', linestyle='--')  
#     axs1[j].plot(LR, HRrd[:], label='Disco de Ruptura [Altura]',linestyle='--')  
#     axs1[j].set_xlabel("Comprimento [m]")
#     axs1[j].set_ylabel("Metros de Coluna de Fluído [m]")
#     axs1[j].set_title(f'Linha Piezométrica: {cw[j]} [mm]')
#     axs1[j].minorticks_on()
#     axs1[j].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#     axs1[j].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.32))

# # Plotagem das Pressõe ao Longo da Linha:
# fig3, axs2=plt.subplots(int(n3/2),int(n3/2), figsize=(8,6))
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
# axs2=axs2.flatten()
# for j in range (0,n3,1):
#     axs2[j].plot(LR, PL[:, j], label=f'Vazão: {Qd} [m³/h]')  
#     axs2[j].plot(LR, PL_Q1[:, j], label=f'Vazão: {Q1} [m³/h]')   
#     axs2[j].plot(LR, PL_Q2[:, j], label=f'Vazão: {Q2} [m³/h]')   
#     axs2[j].plot(LR, PL_Q3[:, j], label=f'Vazão: {Q3} [m³/h]') 
#     axs2[j].plot(LR, PL_Q4[:, j], label=f'Vazão: {Q4} [m³/h]')  
#     axs2[j].set_xlabel("Comprimento [m]")
#     axs2[j].set_ylabel("Metros de Coluna de Fluído [m]")
#     axs2[j].set_title(f'Linha Piezométrica: {cw[j]} [mm]')
#     axs2[j].minorticks_on()
#     axs2[j].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#     axs2[j].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.32))
 
 
# #Plotando a Curva de Perda de Carga para Água
# fig4=plt.figure()
# for j in range (n1):
#     plt.plot(velf[0,:],iw[j,:],label=f'Diâmetro: {dint[j]} [mm]')
#     plt.xlabel("Velocidade do Fluxo [m/s]")
#     plt.ylabel("Perdade de Carga Unitária [m/Km]")
#     plt.title(f'Curva de Perda de Carga')
#     plt.minorticks_on()
#     plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#     plt.legend()

# # Linha Piezométrica para Água
# fig5=plt.figure()
# plt.plot(LR, HR, label=f'Perfil do Terreno')  
# plt.plot(LR, LPw[:], label=f'Vazão: {Qd} [m³/h]')  
# plt.plot(LR, LPw_Q1[:], label=f'Vazão: {Q1} [m³/h]')   
# plt.plot(LR, LPw_Q2[:], label=f'Vazão: {Q2} [m³/h]')   
# plt.plot(LR, LPw_Q3[:], label=f'Vazão: {Q3} [m³/h]') 
# plt.plot(LR, LPw_Q4[:], label=f'Vazão: {Q4} [m³/h]')  
# plt.xlabel("Comprimentos da Tubulação [m]")
# plt.ylabel("Elevação de Coluna de Fluido [m]")
# plt.title(f'Linha Piezométrica para Diferentes Vazões')
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.legend()
# plt.tight_layout()

# fig6=plt.figure()
# plt.plot(LR, PLw[:], label=f'Vazão: {Qd} [m³/h]')  
# plt.plot(LR, PLw_Q1[:], label=f'Vazão: {Q1} [m³/h]')   
# plt.plot(LR, PLw_Q2[:], label=f'Vazão: {Q2} [m³/h]')   
# plt.plot(LR, PLw_Q3[:], label=f'Vazão: {Q3} [m³/h]') 
# plt.plot(LR, PLw_Q4[:], label=f'Vazão: {Q4} [m³/h]')  
# plt.xlabel("Comprimento [m]")
# plt.ylabel("Pressão de Operação")
# plt.title(f'Pressão de Operação para Diferentes Vazões')
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.32))
# plt.tight_layout()

# fig1.savefig(os.path.join(output_folder, "Curva J para Diametros.png"), dpi=300, bbox_inches='tight')
# fig2.savefig(os.path.join(output_folder, "Linha Piezometricas Cw.png"), dpi=300, bbox_inches='tight')
# fig3.savefig(os.path.join(output_folder, "Pressao de Operacao Q.png"), dpi=300, bbox_inches='tight')
# fig4.savefig(os.path.join(output_folder, "Perda de CargaAgua.png"), dpi=300, bbox_inches='tight')
# fig5.savefig(os.path.join(output_folder, "Linha Pizometrica Agua.png"), dpi=300, bbox_inches='tight')
# fig6.savefig(os.path.join(output_folder, "Pressao de Operacao Agua.png"), dpi=300, bbox_inches='tight')

# plt.show()


###############################################################################################################################################################
# Calculo da pressão considerando a estático e disco de Ruptura
LPm=np.maximum(LP[:,2], static_head[:])
PLm=(LPm-HR)*1.71/10

###############################################################################################################################################################

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

##############################################################################################################################################################################
 ##  Inserindo a Logo

# Load the logo using PIL
logo_path = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\Logo.jpg"
logo = Image.open(logo_path)

# Original logo size
original_width, original_height = 276, 120

# Set the scaling factor (e.g., 0.5 to reduce size by 50%)
logo_scale = 0.36

# Calculate the new size
new_size = (int(original_width * logo_scale), int(original_height * logo_scale))
print(f'Resized Logo Dimensions: {new_size}')

# Resize the logo
logo = logo.resize(new_size, Image.Resampling.LANCZOS)

#######################################################################################################################################################################

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

######################################################################################################################################################################################
##############################################################################################################################################################
# Create subplots: one for loglog scale and one for normal scale

# Offset for pipe thickness representation
offset = 20  # meters or matching HR units
HR_bottom = HR - offset / 2
HR_top = HR + offset / 2

# Create plot
fig7, ax7 = plt.subplots(figsize=(14, 7))

# Fill terrain
ax7.fill_between(LR, HR, HR.min() - 20, color='whitesmoke', alpha=1.0, label='Terreno')

# Fill the pipeline with brown (representing slurry) along the entire profile
ax7.fill_between(LR, HR_bottom, HR_top, color='saddlebrown', alpha=0.8, label='Tubo com Polpa')

# Plot elevation profile
ax7.plot(LR, HR, label='Perfil do Terreno', color='dimgray', linewidth=0.1)

# Plot piezometric line
ax7.plot(LR, LP[:, 2], label=f"Linha Piezométrica @ Vazão: {Qd} m³/h", color='royalblue', linewidth=2)

# Plot static head line
ax7.plot(LR, HRm[:], label='Altura Manométrica da Linha Estática', linestyle='--',
         color='seagreen', linewidth=1.5)

# Labels and Title
ax7.set_xlabel("Comprimento da Tubulação [m]", fontsize=12)
ax7.set_ylabel("Pressão / Altura [m]", fontsize=12)
ax7.set_title(f"Linha Piezométrica - Operação Contínua\nCw = {cw[2]}% | Vazão = {Qd} m³/h", 
              fontsize=15, fontweight='bold')

# Grid and Ticks
ax7.minorticks_on()
ax7.grid(which='major', linestyle='--', linewidth=0.6, alpha=0.6, color='gray')
ax7.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.3, color='black')

# Axes limits
ax7.set_xlim(0, max(LR))
ax7.set_ylim(min(HR) - 20, 2100)

# Legend styling
ax7.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.25), fontsize=10, frameon=True)

# Final layout and save
plt.tight_layout(rect=[0, 0.05, 1, 1])
save_path = os.path.join(output_folder, "Linha_Piezometrica_Cw62_Unificada.png")
fig7.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()


##################################################################################################################################

# Create figure
fig8, ax8 = plt.subplots(figsize=(14, 7))  # Slightly taller for consistency

# Plot operational pressure
ax8.plot(LR, PL[:, 2],
         label=f'Pressão de Operação @ Vazão: {Qd} m³/h | Cw: {cw[2]}%',
         color='royalblue',
         linewidth=2.0)

# Labels and title
ax8.set_xlabel("Comprimento da Tubulação [m]", fontsize=12)
ax8.set_ylabel("Pressão / Altura [m]", fontsize=12)
ax8.set_title(f"Pressão de Operação na Linha\nCw = {cw[2]}% | Vazão = {Qd} m³/h",
              fontsize=15, fontweight='bold')

# Grid styling
ax8.minorticks_on()
ax8.grid(which='major', linestyle='--', linewidth=0.6, alpha=0.5, color='gray')
ax8.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.3, color='black')

# Axis limits
ax8.set_xlim(0, max(LR))
ax8.set_ylim(min(PL[:, 2]) - 20, max(PL[:, 2]) + 20)  # Auto-padded Y axis

# Legend
ax8.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.25), fontsize=10, frameon=True)

# Layout and save
plt.tight_layout(rect=[0, 0.05, 1, 1])
save_path = os.path.join(output_folder, "Pressao_na_Linha_Cw62.png")
fig8.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()

















################################################################################################################################################################################


# Plot MAOPm as constant lines
fig9, ax9 = plt.subplots(figsize=(14, 7))  # Unified size with other plots

# Plot MAOP horizontal lines per range
for i in range(len(ranges)):
    x_start = 0 if i == 0 else ranges[i][0]
    x_end = ranges[i][1]
    if not np.isnan(MAOPm[i]):
        ax9.hlines(
            y=MAOPm[i], xmin=x_start, xmax=x_end,
            colors='black', linestyles='--', linewidth=2.5,
            label='MAOP [Máxima Pressão de Operação Admissível]' if i == 0 else ""
        )

# Scatter plot of failure pressure (Pfd)
for surface_class in np.unique(Surface_Location[valid_indices]):
    mask = (Surface_Location == surface_class) & valid_indices
    ax9.scatter(
        LR_valid[mask], Pfd_kg_cm2[mask],
        label=f"{surface_class} - Pressão de Falha das Anomalias",
        color=get_color(surface_class),
        s=18, edgecolors='cyan', alpha=0.6, linewidths=0.5
    )

# Line for Maximum Operating Pressure (PLm)
ax9.plot(
    LR, PLm,
    label=f'Máxima Pressão de Operação\nQ = {Qd} m³/h | Cw = {cw[2]}%',
    linewidth=2.0, color='royalblue'
)

# Labels and title
ax9.set_xlabel('Comprimento da Tubulação [m]', fontsize=12)
ax9.set_ylabel('Pressão [kg/cm²]', fontsize=12)
ax9.set_title('Máxima Pressão Admissível e Pressão de Falha\nAnomalias de Corrosão',
              fontsize=15, fontweight='bold')

# Grid styling
ax9.minorticks_on()
ax9.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.5, color='gray')
ax9.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.3, color='black')

# Axis limits
ax9.set_xlim(0, 123000)
ax9.set_ylim(0, 300)

# Legend formatting
ax9.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.25), fontsize=10, frameon=True)

# Layout and save
plt.tight_layout(rect=[0, 0.05, 1, 1])
save_path = os.path.join(output_folder, "Pressao_Operacao_MAOP_B31G.png")
fig9.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()


# # Gráifico 02: Linhas Piezométricas para Diferentes Cw[%] e Varaiando-se a Vazão:
# fig10 = plt.figure(figsize=(14, 6))
# plt.plot(LR, HR, label=f'Perfil do Terreno')  
# plt.plot(LR, LP[:, 2], label=f'Vazão: {Qd} [m³/h]')  
# plt.xlabel("Comprimento [m]")
# plt.ylabel("Pressão [Kg/cm²]")
# plt.title(f'Pressão de Operação: {cw[2]} [%] e Vazão: {Qd} [m³/h]')
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.32))


##########################################################################################################################################################
# Análise Dados Operacionais




# === 1. Load Excel File ===
file_path = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Dados_Operacionais_03-25.xlsx"  # ✅ Update with actual filename

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip().str.upper()  # Normalize column names
df['DATA'] = pd.to_datetime(df['DATA'])  # Ensure datetime format

# Define required columns
required_cols = ['DATA', 'PEB', 'QEB', 'DEB', 'P45']
optional_cols = ['PT', 'QT', 'DT']

# Convert to numeric, handling errors
for col in required_cols[1:] + [c for c in optional_cols if c in df.columns]:  # Skip 'DATA'
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === 2. Calculate Weight Concentration ===
rho_s = 3.2  # solids density [kg/m³]
rho_l = 1.0  # water density [kg/m³]
df['WEIGHT_CONCENTRATION'] = (rho_s * (df['DEB'] - rho_l)) / (df['DEB'] * (rho_s - rho_l))
df['PRODUCTION_TPH'] = df['QEB'].abs() * df['DEB'] * (df['WEIGHT_CONCENTRATION'] / 100) / 1000

# === 4. Plot Raw Variables Before Filtering ===

# Updated variables (PT and QT removed)
variables = ['PEB', 'QEB', 'DEB', 'P45', 'DT', 'WEIGHT_CONCENTRATION']
titles = [
    'Pressure @ Pump Station (PEB)', 'Flow Rate @ Pump Station (QEB)', 'Density @ Pump Station (DEB)',
    'Pressure @ Km 55 (P45)', 'Density @ Terminal (DT)', 'Weight Concentration (%)'
]

# Filter available variables in DataFrame
valid_variables = [var for var in variables if var in df.columns]
valid_titles = [titles[i] for i, var in enumerate(variables) if var in df.columns]

# Create subplots: 3 rows × 2 columns (fits 6 graphs)

# Apply clean style
sns.set_style("whitegrid")

# Set 6-hour tick locator and shorter formatter (e.g., Mar-22 06h → Mar-22 06)
locator = mdates.HourLocator(interval=6)
formatter = mdates.DateFormatter('%b-%d %Hh')  # Short format for better spacing

# Create subplots (3 rows x 2 columns)
fig, axs = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
axs = axs.flatten()

# Use a colormap for distinct colors
colors = plt.cm.tab10.colors

# Scatter plot for each variable
for i, var in enumerate(valid_variables):
    axs[i].scatter(df['DATA'], df[var], s=6, color=colors[i % len(colors)], alpha=0.8, label=var)  # Reduced marker size
    axs[i].set_title(valid_titles[i], fontsize=12, fontweight='bold')
    axs[i].grid(True, linestyle='--', alpha=0.6)
    axs[i].legend(loc='upper right', fontsize=9)
    axs[i].tick_params(axis='y', labelsize=9)
    axs[i].xaxis.set_major_locator(locator)
    axs[i].xaxis.set_major_formatter(formatter)
    axs[i].tick_params(axis='x', labelrotation=30, labelsize=8)  # Smaller rotation for compactness

# Hide unused subplots, if any
for j in range(len(valid_variables), len(axs)):
    fig.delaxes(axs[j])

# Title and layout
fig.suptitle("Operational Data Over Time (Before Filtering)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save the plot
save_path = os.path.join(output_folder, "Dados_Operacionais_6h.png")
fig.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()



import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

# === Create subplot figure ===
n_vars = len(valid_variables)
n_rows = (n_vars + 1) // 2  # layout: 2 columns per row

fig = make_subplots(
    rows=n_rows, cols=2,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=valid_titles
)

# === Add scatter traces ===
for i, var in enumerate(valid_variables):
    row = (i // 2) + 1
    col = (i % 2) + 1

    fig.add_trace(
        go.Scatter(
            x=df['DATA'],
            y=df[var],
            mode='markers',
            marker=dict(size=5, opacity=0.7),
            name=valid_titles[i],
            showlegend=False
        ),
        row=row, col=col
    )

# === Add time range slider and quick selector ===
fig.update_layout(
    height=300 * n_rows,
    width=1200,
    title_text="Operational Data Over Time (Interactive)",
    title_x=0.5,
    xaxis=dict(
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        type="date"
    ),
    margin=dict(t=60, b=40)
)

# === Save HTML output ===
html_path = os.path.join(output_folder, "Dados_Operacionais.html")
fig.write_html(html_path)

print(f"✅ Interactive plot saved to: {html_path}")




####################################################################################################
# Plotando Diagrama de Operação

# === 2. Constants ===
rho_s = 3.2  # Solids density [g/cm³]
rho_l = 1.0  # Liquid density [g/cm³]
density_threshold = 1.05  # Minimum slurry density to exclude water operation

# === 3. Filter out water operation ===
df_valid = df[df['DEB'] > density_threshold].copy()

# === 4. Compute Weight Concentration (Cw) ===
df_valid['WEIGHT_CONCENTRATION'] = ((df_valid['DEB'] - rho_l) / df_valid['DEB']) * (rho_s / (rho_s - rho_l))

# === 5. Compute Actual Total Solids Production (TSPH) ===
df_valid['TSPH'] = df_valid['QEB'] / ((1 / df_valid['WEIGHT_CONCENTRATION']) - 1 + (1 / rho_s))
df_valid['TSPH'] = df_valid['TSPH'].abs()  # Ensure all values are positive

# === 6. Theoretical Reference Curves ===
flow_range = np.linspace(200, 290, 200)
cw_55 = 0.55
cw_62 = 0.62
tsp_low = flow_range / ((1 / cw_55) - 1 + (1 / rho_s))
tsp_high = flow_range / ((1 / cw_62) - 1 + (1 / rho_s))

# === 7. Prepare Envelope Region Between 220–260 m³/h ===
mask = (flow_range >= 220) & (flow_range <= 260)
flow_fill = flow_range[mask]
tsp_low_fill = tsp_low[mask]
tsp_high_fill = tsp_high[mask]

# === 8. Plotting: Clean & Professional ===


# === Common settings ===
x_label = "Vazão [m³/h]"
y_label = "TSPH [t/h]"
title = "Diagrama Operacional"
x_lim = (210, 280)
y_lim = (180, 300)

# === Version 1: With Scatter Plot ===
plt.figure(figsize=(12, 6))

# Fill operational envelope
plt.fill_between(flow_fill, tsp_low_fill, tsp_high_fill,
                 color='mediumseagreen', alpha=0.3, label='Faixa Operacional [55% – 62%]')

# Theoretical limit lines
plt.plot(flow_range, tsp_low, 'g--', linewidth=1.5, label='Concentração de Sólidos [Cw 55%]')
plt.plot(flow_range, tsp_high, 'g--', linewidth=1.5, label='Concentração de Sólidos [Cw 62%]')

# Scatter: Actual operational data
plt.scatter(df_valid['QEB'], df_valid['TSPH'],
            color='royalblue', s=10, alpha=0.6, label='Dados Operacionais')

# Beautification
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.title(title, fontsize=14, weight='bold')
plt.xlim(*x_lim)
plt.ylim(*y_lim)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper left', fontsize=10, frameon=True)
plt.tight_layout()

# Save the plot with scatter
scatter_path = os.path.join(output_folder, "Diagrama_Operacional.png")
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.show()

# === Version 2: Without Scatter Plot ===
plt.figure(figsize=(12, 6))

# Fill operational envelope
plt.fill_between(flow_fill, tsp_low_fill, tsp_high_fill,
                 color='mediumseagreen', alpha=0.3, label='Faixa Operacional [55% – 62%]')

# Theoretical limit lines
plt.plot(flow_range, tsp_low, 'g--', linewidth=1.5, label='Concentração de Sólidos [Cw 55%]')
plt.plot(flow_range, tsp_high, 'g--', linewidth=1.5, label='Concentração de Sólidos [Cw 62%]')

# Beautification
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.title(title, fontsize=14, weight='bold')
plt.xlim(*x_lim)
plt.ylim(*y_lim)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper left', fontsize=10, frameon=True)
plt.tight_layout()

# Save the plot without scatter
limits_path = os.path.join(output_folder, "Diagrama_Operacional_limites.png")
plt.savefig(limits_path, dpi=300, bbox_inches='tight')
plt.show()


#############################################################################################################



# === 6. Determine Operational Status (Inside or Outside Range) ===
import matplotlib.pyplot as plt
import os

# Flag values
df_valid['IN_RANGE'] = (
    (df_valid['QEB'] >= 220) & (df_valid['QEB'] <= 260) &
    (df_valid['WEIGHT_CONCENTRATION'] >= 0.55) & (df_valid['WEIGHT_CONCENTRATION'] <= 0.62)
)

# Counts
total_points = len(df_valid)
inside_range = df_valid['IN_RANGE'].sum()
outside_range = total_points - inside_range

# === Styled Pie Chart ===
fig, ax = plt.subplots(figsize=(8, 6))

# Custom colors (blue + gray)
colors = ['steelblue', 'lightgray']
labels = [f'Dentro da Faixa ({inside_range})', f'Fora da Faixa ({outside_range})']
explode = (0.05, 0)

wedges, texts, autotexts = ax.pie(
    [inside_range, outside_range],
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    explode=explode,
    wedgeprops=dict(edgecolor='white'),
    textprops=dict(color='black', fontsize=11),
    pctdistance=0.82
)

# Add donut center circle
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
ax.add_artist(centre_circle)

# Title and layout
plt.title("Resumo da Conformidade Operacional", fontsize=15, fontweight='bold')
plt.tight_layout()

# Save figure
save_path = os.path.join(output_folder, "Resumo_Operacional_PieChart.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()


# === 8. Professional Histogram of Weight Concentration ===


# === Histogram Data ===
data = df_valid['WEIGHT_CONCENTRATION']
bins = np.linspace(0.45, 0.7, 40)
counts, bin_edges = np.histogram(data, bins=bins)
percentages = (counts / counts.sum()) * 100
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# === Plot Setup ===
# Optional: clean visual style
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Apply whitegrid styling
sns.set_style("whitegrid")

# === Create Plot ===
fig, ax = plt.subplots(figsize=(14, 7))  # Slightly larger for clarity
bar_color = 'steelblue'
highlight_color = 'palegreen'

# Plot bars
bars = ax.bar(
    bin_centers,
    percentages,
    width=(bin_edges[1] - bin_edges[0]) * 0.95,
    color=bar_color,
    edgecolor='black',
    alpha=0.9,
    linewidth=0.6
)

# Add text labels
for bar, percent in zip(bars, percentages):
    if percent > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{percent:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            color='black'
        )

# Highlight operational range
ax.axvspan(0.55, 0.62, color=highlight_color, alpha=0.3, label='Faixa Alvo (55%–62%)')

# Axis labels and title
ax.set_xlim(0.45, 0.7)
ax.set_ylim(0, max(percentages) * 1.25)
ax.set_xlabel('Concentração de Sólidos (Cw)', fontsize=12)
ax.set_ylabel('Tempo em Operação [%]', fontsize=12)
ax.set_title('Histograma de Concentração de Sólidos [Cw]', fontsize=15, fontweight='bold')

# Grid and ticks
ax.tick_params(axis='both', labelsize=10)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)

# Legend
ax.legend(loc='upper left', fontsize=10, frameon=True)

# Layout and save
plt.tight_layout(rect=[0, 0, 1, 1])
save_path = os.path.join(output_folder, "Histogram_de_Concentracoes.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()



##################################################################################################################

from datetime import timedelta

# === 1. Exclude Water Periods Based on 24h Transit Logic ===
df['exclude_water'] = False
transit_time = timedelta(hours=24)

i = 0
while i < len(df):
    if df.loc[i, 'DEB'] < 1.050:
        start_time = df.loc[i, 'DATA']
        end_time = start_time + transit_time
        mask = (df['DATA'] >= start_time) & (df['DATA'] <= end_time)
        df.loc[mask, 'exclude_water'] = True
        next_valid = df[df['DATA'] > end_time]
        i = next_valid.index.min() if not next_valid.empty else len(df)
    else:
        i += 1

df = df[~df['exclude_water']].copy()

# === 2. Filter Specific Periods of Interest Manually ===
mask_period = (
    ((df['DATA'] >= '2025-02-12') & (df['DATA'] <= '2025-02-15')) |
    ((df['DATA'] >= '2025-02-20') & (df['DATA'] <= '2025-02-25'))
)
df_valid = df[mask_period].copy()



# === 4. Compute Weight Concentration (Cw) and DHDL ===
rho_s = 3.2 * 1000  # solids density [kg/m³]
df_valid['WEIGHT_CONCENTRATION'] = ((df['DEB'] - 1.0) / df['DEB']) * (rho_s / (rho_s - 1000))

ELEB = 1180.376  # Elevation @ pump station
EL45 = 1120.444  # Elevation @ km 55
D45 = 55960      # Length [m]

df['DHDL'] = ((((df_valid['PEB'] - df_valid['P45']) * 10) - (ELEB - EL45)) / df_valid['DEB']) / D45 * 1000

# === 5. Plot Filtered Head Loss Over Time ===
# === Plot: Filtered Head Loss Over Time ===
plt.figure(figsize=(14, 6))

# Plot with enhanced styling
plt.plot(df_valid['DATA'], df_valid['DHDL'],
         color='firebrick', linewidth=2.0, label='Perda de Carga Unitária [m/km]')

# Title and axis labels
plt.title("Perda de Carga Unitária\n(Regime Permanente)", fontsize=16, fontweight='bold')
plt.xlabel("Data", fontsize=12)
plt.ylabel("Perda de Carga Unitária [m/km]", fontsize=12)

# Date formatting for better x-axis readability
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()

# Save the plot
save_path = os.path.join(output_folder, "Perda_de_Carga_Unitária_Medida_RP.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Show plot
plt.show()

###############################################################################################################



# === Constants ===
D_ext_inch = 9.625
wall_thickness_inch = 0.39250
D_ext_m = D_ext_inch * 0.0254
wall_thickness_m = wall_thickness_inch * 0.0254
D_int_m = D_ext_m - 2 * wall_thickness_m
A_m2 = (np.pi / 4) * D_int_m**2  # Internal area in m²

# === Compute Velocity (m/s) for Operational Data ===
df_valid['VELOCITY'] = (df_valid['QEB'] / 3600) / A_m2
vel_data = df_valid['VELOCITY'].values
dh_data = df_valid['DHDL'].values

# === Define color ranges for Cw ===
color_ranges = [
    (0.53, 0.55, 'green'),
    (0.55, 0.57, 'blue'),
    (0.57, 0.59, 'orange'),
    (0.59, 0.61, 'purple'),
    (0.61, 0.63, 'red')
]

# === Log-Log Plot with Color-Coded DHDL by Cw ===


# === Create subplots ===
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# === First subplot: Log-Log plot ===
for j in range(n3):
    axes[0].loglog(velf[0, :], iwaspmp[:, j], linestyle='-', linewidth=2, label=f'Cw {cw[j]}')

# Reference line for water
axes[0].loglog(velf[0, :], 1000 * iwp, linestyle='--', linewidth=2, color='black', label='Água [Colebrook]')

# Scatter points grouped by concentration ranges
for cmin, cmax, color in color_ranges:
    subset = df_valid[(df_valid['WEIGHT_CONCENTRATION'] >= cmin) & (df_valid['WEIGHT_CONCENTRATION'] < cmax)]
    velocities = (subset['QEB'] / 3600) / A_m2
    axes[0].scatter(velocities, subset['DHDL'], s=20, alpha=0.6, label=f'Cw {int(cmin*100)}–{int(cmax*100)}%', color=color)

# Styling
axes[0].set_xlabel('Velocidade do Fluxo [m/s]', fontsize=12)
axes[0].set_ylabel('Perda de Carga Unitária [m/km]', fontsize=12)
axes[0].set_title('Perda de Carga - Escala Log-Log', fontsize=14, fontweight='bold')
axes[0].set_xlim(0.1, 10)
axes[0].grid(True, which="both", linestyle='--', alpha=0.5)
axes[0].legend(loc='upper right', fontsize=9)

# === Second subplot: Linear plot ===
for j in range(n3):
    axes[1].plot(velf[0, :], iwaspmp[:, j], linestyle='-', linewidth=2, label=f'Cw {cw[j]}')

axes[1].plot(velf[0, :], 1000 * iwp, linestyle='--', linewidth=2, color='black', label='Água [Colebrook]')

for cmin, cmax, color in color_ranges:
    subset = df_valid[(df_valid['WEIGHT_CONCENTRATION'] >= cmin) & (df_valid['WEIGHT_CONCENTRATION'] < cmax)]
    velocities = (subset['QEB'] / 3600) / A_m2
    axes[1].scatter(velocities, subset['DHDL'], s=20, alpha=0.6, label=f'Cw {int(cmin*100)}–{int(cmax*100)}%', color=color)

# Styling
axes[1].set_xlabel('Velocidade do Fluxo [m/s]', fontsize=12)
axes[1].set_ylabel('Perda de Carga Unitária [m/km]', fontsize=12)
axes[1].set_title('Perda de Carga - Escala Linear', fontsize=14, fontweight='bold')
axes[1].set_xlim(0.1, 10)
axes[1].set_ylim(0.1, 250)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(loc='upper right', fontsize=9)

# Layout and save
plt.tight_layout(rect=[0, 0, 1, 0.98])
save_path = os.path.join(output_folder, "Calibracao_do_Modelo.png")
fig.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()




###############################################################################################################
# Calculating at the interface

li = 60000

def find_nearest(LR, li):
    LR = np.asarray(LR)
    idx = (np.abs(LR - li)).argmin()
    return LR[idx], idx  # Nearest value and its index

# Find nearest
nearest_L, idx = find_nearest(LR, li)
hj = HR[idx]
print(hj)

Hjump=130
print(Hjump)

dHTsi=iwaspn[2][0]*L[0]+ iwaspn[2][1]*L[1]+ iwaspn[2][2]*40500
print(dHTsi)
dHwai     =iwn   [2]*(13500) + iwn[3]*49500
print(dHwai)

dHti=dHTsi+dHwai-Hjump
print(dHti)

dpp=100

LPii = dHti+((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j]+HR[0]) + dpp
print(LPii)

LPi=np.zeros(n5)
PLi=np.zeros(n5)
LPi[0]=LPii

jump_applied = False

for j in range(1, n5):
    if LR[j] > 0 and LR[j] <= 7900:
        LPi[j] = LPi[j-1] - (LR[j] - LR[j-1]) * iwaspn[2][0]

    elif LR[j] > 7900 and LR[j] <= 19500:
        LPi[j] = LPi[j-1] - (LR[j] - LR[j-1]) * iwaspn[2][1]

    elif LR[j] > 19500 and LR[j] < LR[idx]:
        LPi[j] = LPi[j-1] - (LR[j] - LR[j-1]) * iwaspn[2][2]

    elif LR[j] == LR[idx]:
        LPi[j] = LPi[j-1] - (LR[j] - LR[j-1]) * iwn[2] + Hjump

    elif LR[j] > LR[idx] and LR[j] <= 73500:
        LPi[j] = LPi[j-1] - (LR[j] - LR[j-1]) * iwn[2]

    elif LR[j] > 73500 and LR[j] < 123000:
        LPi[j] = LPi[j-1] - (LR[j] - LR[j-1]) * iwn[3]

    elif LR[j] == 123000:
        LPi[j] = LPi[j-1] - (LR[j] - LR[j-1]) * iwn[3] - dpp



# Cacula a Pressão na Linha 
for j in range(n5):

    if LR[j] > 0 and LR[j] < LR[idx]:
        PLi[j]=(LPi[j] - HR[j])*(rhof[2]/rho)/10

    if LR[j] >= LR[idx] and LR[j] <= 123000:

        PLi[j]=(LPi[j] - HR[j])/10

  

# Gráifico 02: Linhas Piezométricas para Diferentes Cw[%] e Varaiando-se a Vazão:
# Example offset (adjust as needed for pipeline thickness)


# Offset for pipe wall thickness
offset = 20  # meters (or matching HR units)
HR_bottom = HR - offset / 2
HR_top = HR + offset / 2

# === Create the plot ===
fig, ax = plt.subplots(figsize=(14, 7))

# Plot terrain profile
ax.plot(LR, HR, color='dimgray', linewidth=0.1, label='Perfil do Terreno')

# Plot pressure line
ax.plot(LR, LPi, color='royalblue', linewidth=2, label=f'Linha Piezométrica (Q = {Qd} m³/h)')

# Fill below terrain as terrain
ax.fill_between(LR, HR, HR.min() - 20, color='whitesmoke', label='Terreno')

# Fill between pipe walls (slurry and water)
ax.fill_between(LR, HR_bottom, HR_top, where=(LR <= 60000), interpolate=True,
                color='saddlebrown', alpha=0.8, label='Polpa')
ax.fill_between(LR, HR_bottom, HR_top, where=(LR > 60000), interpolate=True,
                color='skyblue', alpha=0.7, label='Água')

# Labels and titles
ax.set_xlabel("Comprimento [m]", fontsize=12)
ax.set_ylabel("Pressão / Altura [m]", fontsize=12)
ax.set_title(f"Linha Piezométrica - Operação em Batch\nCw = {cw[2]}% | Vazão = {Qd} m³/h",
             fontsize=15, fontweight='bold')

# Grid and ticks
ax.minorticks_on()
ax.grid(which='major', linestyle='--', alpha=0.5)
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
ax.set_xlim(0, 123000)
# Legend
ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.25), fontsize=10, frameon=True)

# Layout and save
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Give space for legend
save_path = os.path.join(output_folder, "Linha_Piezometrica_Operacao_em_Batch.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()


########################################################################################################################



fig11, ax = plt.subplots(figsize=(14, 6))

# Plot the operational pressure
ax.plot(LR, PLi[:], color='royalblue', linewidth=2.0, label=f'Vazão: {Qd} m³/h')

# Axis labels and title
ax.set_xlabel("Comprimento [m]", fontsize=12)
ax.set_ylabel("Pressão de Operação [kg/cm²]", fontsize=12)
ax.set_title("Pressão de Operação", fontsize=15, fontweight='bold')
ax.set_xlim(0,123000)

# Grid and ticks
ax.minorticks_on()
ax.grid(which='major', linestyle='--', alpha=0.5)
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

# Legend
ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.25), fontsize=10, frameon=True)

# Layout and save
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Space for legend
save_path = os.path.join(output_folder, "Pressao_de_Operacao_para_Diferentes_Vazoes.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()




