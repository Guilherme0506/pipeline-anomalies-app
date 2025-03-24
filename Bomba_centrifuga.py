import numpy as np
import matplotlib.pyplot as plt

# Curva do Sistema
Qd = 240
Q = np.arange(1, Qd * 2, 1)
target = Qd
nearest_value = Q[np.abs(Q - target).argmin()]
n2 = int(len(Q))
velf = np.zeros(n2)
Rey = np.zeros(n2)
f = np.zeros(n2)
iw = np.zeros(n2)
SC = np.zeros(n2)

dint = (9.625 * 25.4 - 20)
Aint = (np.pi * (dint / 1000) ** 2) / 4
L = 123000
e = 0.05
k = e / dint
b = (k / 3.7) ** 1.11

rho = 1000
mi = 10 ** -6
index = np.where(Q == Qd)[0]

velf[:] = (Q[:] / 3600) / Aint
Rey[:] = (velf[:] * (dint / 1000)) / mi

f[:] = (1 / (-1.8 * (np.log10(6.9 / Rey[:] + b)))) ** 2
print(index)
print(f[index])
iw[:] = (f[:] / (dint / 1000)) * (velf[:] ** 2) / (2 * 9.81)
SC[:] = (530 - 1220) + iw[:] * L
H1 = 1220
H2 = 530
dHT = iw[index] * L
LPinicial = (530 - 1220) + dHT + H1
print(LPinicial)

Qp = np.array([0, 200, 300]) 
Hp = np.array([2000, 1500, 1000]) 
coefficients = np.polyfit(Qp, Hp, 2)
print("Polynomial Coefficients:", coefficients)
polynomial = np.poly1d(coefficients)
x_fit = np.linspace(Qp[0], Hp[-1], 100)
y_fit = polynomial(x_fit)

# Evaluate the difference between SC and polynomial values at discrete points
differences = np.abs(SC - polynomial(Q))

# Find the index of the minimum difference
min_diff_index = np.argmin(differences)
intersection_Q = Q[min_diff_index]
intersection_H = polynomial(intersection_Q)
print(intersection_Q)

print(f"Intersection Point: Q = {intersection_Q}, H = {intersection_H}")

plt.plot(Q, SC, label="System Curve")
plt.scatter(Qp, Hp, color='red', label="Pump Performance Data")
plt.plot(x_fit, y_fit, color='green', label="Pump Performance Curve")
plt.scatter(intersection_Q, intersection_H, color='blue', zorder=5, label='Intersection Point')
plt.xlabel("Flow Rate (Q)")
plt.ylabel("Head (H)")
plt.legend()
plt.grid(True)
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

