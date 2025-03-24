import numpy as np
import matplotlib.pyplot as plt

# Dados
esppp = np.array([10.31, 8.74, 7.92, 8.74, 10.31])  # Espessuras de parede em mm
Lprojeto = np.array([9381, 11314, 54093, 46631, 1581.59])  # Comprimentos de cada segmento em mm
S = 360  # Tensão admissível (kgf/cm²)
dext = 9.625 * 25.4  # Diâmetro externo em mm

# Cálculo da Pressão Máxima Admissível de Operação (MAOP) em kgf/cm²
MAOP = (2 * S * esppp * 0.8 / dext) * 10.1972  # Conversão adequada

# Calculando o comprimento acumulado da tubulação
pipeline_length = np.cumsum(Lprojeto)  # Soma cumulativa dos comprimentos

# Criando os valores para o gráfico em degraus corretamente
pipeline_length_steps = np.insert(np.repeat(pipeline_length, 2), 0, 0)  # Adiciona 0 inicial
MAOP_steps = np.insert(np.repeat(MAOP, 2), 0, MAOP[0])  # Adiciona o primeiro valor de MAOP no início

# Plotando o gráfico em degraus corretamente
plt.figure(figsize=(8, 6))
plt.step(pipeline_length_steps, MAOP_steps, where='pre', linestyle='-', color='b', marker='o')

# Adicionando título e rótulos aos eixos
plt.title('Pressão Máxima por Comprimento Acumulado da Tubulação', fontsize=14)
plt.xlabel('Comprimento Acumulado da Tubulação (mm)', fontsize=12)
plt.ylabel('Pressão Máxima (kgf/cm²)', fontsize=12)

# Exibindo o gráfico
plt.grid(True)
plt.show()
plt.show()



# Dados
esppm = np.array([8.6, 6.882, 6.152, 9.123, 6.152, 9.123, 6.152, 6.86, 9.203, 
                  6.86, 9.203, 6.86, 9.203, 6.86, 7.8, 6.86, 8.043, 7.002])  # Espessuras em mm
Lm = np.array([9381, 11314, 27336, 409, 19240, 125, 7247, 2092, 176, 
               5000, 205, 7412, 211, 10196, 4468, 13898, 2290, 2000])  # Comprimentos em mm
S = 360  # Tensão admissível (kgf/cm²)
dext = 9.625 * 25.4  # Diâmetro externo em mm

# Cálculo da Pressão Máxima Admissível de Operação (MAOP) em kgf/cm²
MAOPm = (2 * S * esppm * 0.8 / dext) * 10.1972  # Conversão adequada

# Calculando o comprimento acumulado da tubulação
pipeline_length_m = np.cumsum(Lm)  # Soma cumulativa dos comprimentos
pipeline_length_steps_m = np.insert(np.repeat(pipeline_length_m, 2), 0, 0)  # Adiciona 0 inicial
MAOPm_steps = np.insert(np.repeat(MAOPm, 2), 0, MAOPm[0])  # Adiciona o primeiro valor de MAOP no início

# Plotando o gráfico em degraus corretamente
plt.figure(figsize=(8, 6))
plt.step(pipeline_length_steps_m, MAOPm_steps, where='pre', linestyle='-', color='b', marker='o')
plt.title('Pressão Máxima por Comprimento Acumulado da Tubulação', fontsize=14)
plt.xlabel('Comprimento Acumulado da Tubulação (mm)', fontsize=12)
plt.ylabel('Pressão Máxima (kgf/cm²)', fontsize=12)
plt.grid(True)
plt.show()

