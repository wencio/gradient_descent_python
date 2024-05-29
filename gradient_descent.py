import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Inicialización de parámetros
m = 0
b = 0
alpha = 0.01
iterations = 1000
epsilon = 1e-6  # Umbral de convergencia
n = float(len(X))

# Para almacenar la pérdida en cada iteración
losses = []

# Gradient Descent con criterio de convergencia
for i in range(iterations):
    y_pred = m * X + b
    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)
    
    # Cálculo de los gradientes
    gradient_m = (-2/n) * sum(X * (y - y_pred))
    gradient_b = (-2/n) * sum(y - y_pred)
    
    # Actualización de los parámetros
    m = m - alpha * gradient_m
    b = b - alpha * gradient_b
    
    # Criterio de convergencia
    if i > 0 and abs(losses[i] - losses[i-1]) < epsilon:
        print(f'Convergencia alcanzada en la iteración {i}')
        break

# Visualizar los resultados
print("Pendiente (m):", m)
print("Intersección (b):", b)

plt.plot(range(len(losses)), losses)
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida')
plt.title('Minimización de la Función de Pérdida')
plt.show()
