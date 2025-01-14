import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file
loss_values = []
with open("C:/#DTU/3 ugers dec2025/Sklaering/Training loss.txt", 'r') as file:
    lines = file.readlines()
    for line in lines:
        if "loss:" in line:
            loss = line.split("loss:")[1].strip()
            loss_values.append(float(loss))

loss_values = np.array(loss_values)

# Initialize parameters
a = 0

T = 1000
step = 1e-3

cost = np.zeros(T)
param_a = np.zeros(T)
param_b = np.zeros(T)

# Perform gradient descent
a = 0
learning_rate = 1e-3
T = 1000

param_a = []
loss_history = []

# Gradient descent loop
for t in range(T):
    y_est = a * loss_values
    loss = np.sum((loss_values - y_est) ** 2)
    grad_a = -2 * np.sum((loss_values - y_est) * loss_values)
    a -= learning_rate * grad_a

    # Store values for visualization
    param_a.append(a)
    loss_history.append(loss)

# Plot Loss vs. Iterations
plt.figure(figsize=(10, 5))
plt.plot(range(T), loss_history, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs. Iterations')
plt.grid()
plt.legend()
plt.show()

# Plot Loss vs. Parameter a
plt.figure(figsize=(10, 5))
plt.plot(param_a, loss_history, label='Loss vs. Parameter a')
plt.xlabel('Parameter a')
plt.ylabel('Loss')
plt.title('Loss vs. Parameter a')
plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(range(T), param_a, label='Parameter a')
plt.xlabel('Iterations')
plt.ylabel('Parameter a')
plt.title('Parameter a vs. Iterations')
plt.grid()
plt.legend()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Prepare data for 3D plot
iterations = range(T)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(param_a, iterations, loss_history, label='Loss Surface', color='blue')
ax.set_xlabel('Parameter a')
ax.set_ylabel('Iterations')
ax.set_zlabel('Loss')
ax.set_title('3D Plot of Loss vs. Parameter and Iterations')
plt.show()

