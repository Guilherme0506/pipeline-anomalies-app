import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

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
mif=np.zeros((n3,n2))
vr=np.zeros((n3,n2))
vr_n=np.zeros((n1,n2,n3))
cv=np.zeros((n3))
cw=np.zeros((n3,n2))
rhoveh=np.zeros((n1,n2,n3))
durand=np.zeros((n1,n2,n3,n4))
fd=np.zeros((n1,n2,n3))
vtpf=np.zeros((n3,n2,n4))
Rep=np.zeros((n3,n2,n4))
cd=np.zeros((n3,n2,n4))
CCA=np.zeros((n1,n2,n3,n4))
rhof=np.zeros((n3,n2))
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
phi=np.zeros((n3,n2))
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
ds=3940                       # Densidade dos sólidos
B=3.7                         # Parâmetro reológico
cw[0][:]=65
cw[1][:]=67
cw[2][:]=69
cw[3][:]=71

cvdp=[0.1,0.35,0.45,0.1]
d50=0.05
dpf=np.zeros(n4)
dpf[0]=d50*3
dpf[1]=d50
dpf[2]=d50*0.5
dpf[3]=d50*0.25
beta=1

for j in range (n1):
    dint[j]=int(float("Informe o diametro em [mm]:"))
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

for j in range (n2):
    for i in range (n3):

        rhof[i][j]=1000/(1-(cw[i][j]/100)*((ds-1000)/ds))
        cv[i]=((rhof[i][j]-1000)/(ds-1000))

for j in range (n2):
    for k in range (n3):

        phi[k][j]=((rhof[k][j]-rho)/(ds-rho))
        vr[k][j]=phi[k][j]/(1-phi[k][j])
        mif[k][j]=(0.89*0.001*(10**(vr[k][j])*B))/rhof[k][j]

for i in range (n1):
    for j in range (n2):
        for k in range (n3):

            Reyp[i][j][k]=(velf[i][j]*(dint[i]/1000)*rhof[k][j])/mif[k][j]
            fp[i][j][k]=(1/(-1.8*(np.log10(6.9/Reyp[i][j][k] + b[i]))))**2
            iph[i][j][k]=(((fp[i][j][k]/(dint[i]/1000))*(velf[i][j]**2))/(2*g))
            unp[i][j][k]=velf[i][j]*np.sqrt((fp[i][j][k])/8)*(rhof[k][j]/rho)

# Passo 02: Correção (Iterativo)


for i in range(n1):
    for n in range(10):
        for j in range(n2):
            for k in range(n3):
                for m in range(n4):
                   
                    vtpf[k][j][m] = (((ds - rhof[k][j]) / rhof[k][j]) * ((dpf[m] / 1000)**2) * 9.81 / (18 * mif[k][j]))
                    Rep[k][j][m] = vtpf[k][j][m] * (dpf[k] / 1000) / mif[k][j]
                    cd[k][j][m] = (24 / Rep[k][j][m]) * ((1 + 0.173 * Rep[k][j][m])**0.657) + (0.413 / (1 + 16300 * Rep[k][j][m]**(-1.09)))
                    CCA[i][j][k][m] = 10**(-1.8 * (vtpf[k][j][m] / (beta * 0.4 * unp[i][j][k])))
               
                    veht[i][j][k] = (cvdp[0] * CCA[i][j][k][0] + cvdp[1] * CCA[i][j][k][1] + cvdp[2] * CCA[i][j][k][2] + cvdp[3] * CCA[i][j][k][3])*cv[k]
                    bed[i][j][k] = 1 - veht[i][j][k]
                    rhoveh[i][j][k] = (veht[i][j][k] * ds / rho + (1 - veht[i][j][k])) * 1000
                    durand[i][j][k][m] = (g * (dint[i] / 1000) * ((ds - rhoveh[i][j][k]) / rhoveh[i][j][k]) / ((velf[i][j]**2) * np.sqrt(cd[k][j][m])))**(3 / 2)
                    fd[i][j][k] = bed[i][j][k] * 82 * cv[k] * (durand[i][j][k][0] + durand[i][j][k][1] + durand[i][j][k][2] + durand[i][j][k][3])
                    
                    ibed[i][j][k] = fd[i][j][k] * iw[i][j]  # metros de coluna de água

                    phi_n[i][j][k] = (rhoveh[i][j][k] - rho) / (ds - rho)
                    vr_n[i][j][k] = phi_n[i][j][k] / (1 - phi_n[i][j][k])
                    miveh[i][j][k] = (0.89 * 0.001 * (10**(vr_n[i][j][k] * B))) / (rhoveh[i][j][k])

                Reyveh[i][j][k] = (velf[i][j] * (dint[i] / 1000) * rhoveh[i][j][k]) / miveh[i][j][k]
                fveh[i][j][k] = (1 / (-1.8 * (np.log10(6.9 / Reyveh[i][j][k] + b[i]))))**2
                iveh[i][j][k] = (((fveh[i][j][k] / (dint[i] / 1000)) * (velf[i][j]**2) * (rhoveh[i][j][k] / rho)) / (2 * g))
                iwasp[i][j][k] = (iveh[i][j][k] + ibed[i][j][k]) / (rhoveh[i][j][k] / rho)  # metros de coluna de polpa
                unp[i][j][k] = velf[i][j] * np.sqrt((iwasp[i][j][k] * 9.81 * (dint[i] / 1000 / 4)))
    n=n+1

# Determinar Linh a Piezométrica

Q1=Qd*0.8
Q2=Qd*0.9
Q3=Qd*1.1
Q4=Qd*1.2

if Qd in Q:
    IQd = int(np.where(Q == Qd)[0][0])
    IQ1 = int(np.where(Q == Q1)[0][0])
    IQ2 = int(np.where(Q == Q2)[0][0])
    IQ3 = int(np.where(Q == Q3)[0][0])
    IQ4 = int(np.where(Q == Q4)[0][0])
   
    for i in range(n1):
        for j in range(n3):
            iwaspn[i][j] = iwasp[i, IQd, j]
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
PT=0
PEB=0

HRm=np.max(HR)*HRmax

# Calcula a Linha Piezométrica
# Calcula a Linha Piezométrica Inicial

for j in range(n3):
    LP[0,j]=dHT[j]+((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j][0]+HR[0])  
    LP_Q1[0,j]=dHT_Q1[j]+((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j][0]+HR[0])
    LP_Q2[0,j]=dHT_Q2[j]+((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j][0]+HR[0])
    LP_Q3[0,j]=dHT_Q3[j]+((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j][0]+HR[0])
    LP_Q4[0,j]=dHT_Q4[j]+((HR[n6]-HR[0])+10*(PT-PEB)*rho/rhof[j][0]+HR[0])

    LPw[0]=dHTw+((HR[n6]-HR[0])+10*(PT-PEB)+HR[0])  
    LPw_Q1[0]=dHTw_Q1+((HR[n6]--HR[0])+10*(PT-PEB)+HR[0])
    LPw_Q2[0]=dHTw_Q2+((HR[n6]--HR[0])+10*(PT-PEB)+HR[0])
    LPw_Q3[0]=dHTw_Q3+((HR[n6]--HR[0])+10*(PT-PEB)+HR[0])
    LPw_Q4[0]=dHTw_Q4+((HR[n6]--HR[0])+10*(PT-PEB)+HR[0])

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
        PL[j][i]=(LP[j][i] - HR[j])*(rhof[i][0]/rho)/10
        PL_Q1[j][i]=(LP_Q1[j][i] - HR[j])*(rhof[i][0]/rho)/10
        PL_Q2[j][i]=(LP_Q2[j][i] - HR[j])*(rhof[i][0]/rho)/10
        PL_Q3[j][i]=(LP_Q3[j][i] - HR[j])*(rhof[i][0]/rho)/10
        PL_Q4[j][i]=(LP_Q3[j][i] - HR[j])*(rhof[i][0]/rho)/10

        PLw   [j]=(LPw   [j] - HR[j])/10
        PLw_Q1[j]=(LPw_Q1[j] - HR[j])/10
        PLw_Q2[j]=(LPw_Q2[j] - HR[j])/10
        PLw_Q3[j]=(LPw_Q3[j] - HR[j])/10
        PLw_Q4[j]=(LPw_Q3[j] - HR[j])/10

# Gráficos
# Gráifico 01: Curva J e Curva J em LogLog Scale
fig1, ax=plt.subplots(int(2),int(n1))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
ax=ax.flatten()
for j in range (n1):
    ax[j].loglog(velf[0, :], 1000 * iw[j, :], label=f'Cw:0.0 %')
    ax[j].loglog(velf[0, :], 1000 * iwasp[j, :, 0], label=f'Cw: {cw[0][0]} %')
    ax[j].loglog(velf[0, :], 1000 * iwasp[j, :, 1], label=f'Cw: {cw[1][0]} %')
    ax[j].loglog(velf[0, :], 1000 * iwasp[j, :, 2], label=f'Cw: {cw[2][0]} %')
    ax[j].loglog(velf[0, :], 1000 * iwasp[j, :, 3], label=f'Cw: {cw[3][0]} %')
    ax[j].set_xlabel('Velocidade [m/s]')
    ax[j].set_ylabel('Perda de Carga [m/Km]')
    ax[j].set_title(f'Curva J Diâmetro: {dint[0]} [mm]')
    ax[j].minorticks_on()
    ax[j].grid(which='minor', linestyle=':', linewidth=0.5, color='grey')
    ax[j].legend(loc='upper center', ncol=5,bbox_to_anchor=(0.5, 1.25))

for j in range (n1,n1*2):
    ax[j].plot(velf[0, :], 1000 * iw[0, :], label=f'Cw:0.0 %')
    ax[j].plot(velf[0, :], 1000 * iwasp[0, :, 0], label=f'Cw: {cw[0][0]} %')
    ax[j].plot(velf[0, :], 1000 * iwasp[0, :, 1], label=f'Cw: {cw[1][0]} %')
    ax[j].plot(velf[0, :], 1000 * iwasp[0, :, 2], label=f'Cw: {cw[2][0]} %')
    ax[j].plot(velf[0, :], 1000 * iwasp[0, :, 3], label=f'Cw: {cw[3][0]} %')
    ax[j].set_xlabel('Velocidade [m/s]')
    ax[j].set_ylabel('Perda de Carga [m/Km]')
    ax[j].set_title(f'Curva J Diâmetro: {dint[0]} [mm]')
    ax[j].minorticks_on()
    ax[j].grid(which='minor', linestyle=':', linewidth=0.5, color='grey')
    ax[j].legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.28))

# Gráifico 02: Linhas Piezométricas para Diferentes Cw[%] e Varaiando-se a Vazão:
fig2, axs1=plt.subplots(int(n3/2),int(n3/2), figsize=(8,6))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
axs1=axs1.flatten()
for j in range (0,n3,1):
    axs1[j].plot(LR, HR, label=f'Perfil do Terreno')  
    axs1[j].plot(LR, LP[:, j], label=f'Vazão: {Qd} [m³/h]')  
    axs1[j].plot(LR, LP_Q1[:, j], label=f'Vazão: {Q1} [m³/h]')   
    axs1[j].plot(LR, LP_Q2[:, j], label=f'Vazão: {Q2} [m³/h]')   
    axs1[j].plot(LR, LP_Q3[:, j], label=f'Vazão: {Q3} [m³/h]') 
    axs1[j].plot(LR, LP_Q4[:, j], label=f'Vazão: {Q4} [m³/h]')  
    axs1[j].plot(LR, HRm[:], label='Linha Estática')  
    axs1[j].set_xlabel("Comprimento [m]")
    axs1[j].set_ylabel("Metros de Coluna de Fluído [m]")
    axs1[j].set_title(f'Linha Piezométrica: {cw[j][0]} [mm]')
    axs1[j].minorticks_on()
    axs1[j].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axs1[j].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.32))

# Plotagem das Pressõe ao Longo da Linha:
fig3, axs2=plt.subplots(int(n3/2),int(n3/2), figsize=(8,6))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
axs2=axs2.flatten()
for j in range (0,n3,1):
    axs2[j].plot(LR, PL[:, j], label=f'Vazão: {Qd} [m³/h]')  
    axs2[j].plot(LR, PL_Q1[:, j], label=f'Vazão: {Q1} [m³/h]')   
    axs2[j].plot(LR, PL_Q2[:, j], label=f'Vazão: {Q2} [m³/h]')   
    axs2[j].plot(LR, PL_Q3[:, j], label=f'Vazão: {Q3} [m³/h]') 
    axs2[j].plot(LR, PL_Q4[:, j], label=f'Vazão: {Q4} [m³/h]')  
    axs2[j].set_xlabel("Comprimento [m]")
    axs2[j].set_ylabel("Metros de Coluna de Fluído [m]")
    axs2[j].set_title(f'Linha Piezométrica: {cw[j][0]} [mm]')
    axs2[j].minorticks_on()
    axs2[j].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axs2[j].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.32))
 
#Plotando a Curva de Perda de Carga para Água
fig4=plt.figure()
for j in range (n1):
    plt.plot(velf[0,:],iw[j,:],label=f'Diâmetro: {dint[j]} [mm]')
    plt.xlabel("Velocidade do Fluxo [m/s]")
    plt.ylabel("Perdade de Carga Unitária [m/Km]")
    plt.title(f'Curva de Perda de Carga')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.legend()

# Linha Piezométrica para Água
fig5=plt.figure()
plt.plot(LR, HR, label=f'Perfil do Terreno')  
plt.plot(LR, LPw[:], label=f'Vazão: {Qd} [m³/h]')  
plt.plot(LR, LPw_Q1[:], label=f'Vazão: {Q1} [m³/h]')   
plt.plot(LR, LPw_Q2[:], label=f'Vazão: {Q2} [m³/h]')   
plt.plot(LR, LPw_Q3[:], label=f'Vazão: {Q3} [m³/h]') 
plt.plot(LR, LPw_Q4[:], label=f'Vazão: {Q4} [m³/h]')  
plt.xlabel("Comprimentos da Tubulação [m]")
plt.ylabel("Elevação de Coluna de Fluido [m]")
plt.title(f'Linha Piezométrica para Diferentes Vazões')
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.legend()
plt.tight_layout()

fig6=plt.figure()
plt.plot(LR, PLw[:], label=f'Vazão: {Qd} [m³/h]')  
plt.plot(LR, PLw_Q1[:], label=f'Vazão: {Q1} [m³/h]')   
plt.plot(LR, PLw_Q2[:], label=f'Vazão: {Q2} [m³/h]')   
plt.plot(LR, PLw_Q3[:], label=f'Vazão: {Q3} [m³/h]') 
plt.plot(LR, PLw_Q4[:], label=f'Vazão: {Q4} [m³/h]')  
plt.xlabel("Comprimento [m]")
plt.ylabel("Pressão de Operação")
plt.title(f'Pressão de Operação para Diferentes Vazões')
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.32))
plt.tight_layout()

plt.show()

# Gráifico 03: Máxima Linha Piezométrica:
# 
# 
# 
# Gráfico 04: Pressões de Operação (Permanente):
#
#
# Gráfico 05: Máximas Pressões de Operação (Permanente):