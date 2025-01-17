import numpy as np
import matplotlib.pyplot as plt
import torch
from ChessCNN import ChessCNN

model = ChessCNN()
model.load_state_dict(torch.load('chess_model.pth'))
model.eval()


# Access the state_dict
state_dict = model.state_dict()


# Read the data from the file
loss_values = []

with open("Training loss.txt", 'r') as file:
    lines = file.readlines()
    for line in lines:
        if "loss:" in line:
            loss = line.split("loss:")[1].strip()
            loss_values.append(float(loss))

loss_values = np.array(loss_values)

# Initialize parameters

a = 0
b = 0
learning_rate = 1e-3
T = 1000


# Perform gradient descent


param_a = []
param_b = []
loss_history = []
for t in range(T):
    y_est = a * loss_values + b
    loss = np.sum((loss_values - y_est) ** 2)
    grad_a = -2 * np.sum((loss_values - y_est) * loss_values)
    grad_b = -2 * np.sum(loss_values - y_est)
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    loss = np.sum((loss_values - y_est) ** 2)
    param_b.append(b)
    loss_history.append(loss)
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
plt.xlabel('Iterations')
plt.ylabel('Parameter a')
plt.title('Parameter a vs. Iterations')
plt.grid()
plt.legend()
plt.show()

# Plot Parameter b vs. Iterations
plt.figure(figsize=(10, 5))
plt.plot(range(T), param_b, label='Parameter b')
plt.xlabel('Iterations')
plt.ylabel('Parameter b')
plt.title('Parameter b vs. Iterations')
plt.grid()
plt.legend()
plt.show()
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




# Perform gradient descent on model parameters
learning_rate = 1e-3
T = 1000

# Convert state_dict to a list of parameters
params = [param for param in model.parameters()]

# Initialize lists to store parameter values and loss history
param_history = [[] for _ in params]
lossHistory = []

# Gradient descent loop
for t in range(T):
    # Forward pass
    y_est = model(torch.tensor(loss_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    loss = torch.nn.functional.mse_loss(y_est, torch.tensor(loss_values, dtype=torch.float32).unsqueeze(1))

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Update parameters
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    # Store values for visualization
    for i, param in enumerate(model.parameters()):
        param_history[i].append(param.clone().detach().numpy())
    lossHistory.append(loss.item())

# Plot Loss vs. Iterations
plt.figure(figsize=(10, 5))
plt.plot(range(T), lossHistory, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs. Iterations')
plt.grid()
plt.legend()
plt.show()

# Plot parameter values vs. Iterations
for i, param_hist in enumerate(param_history):
    plt.figure(figsize=(10, 5))
    plt.plot(range(T), param_hist, label=f'Parameter {i}')
    plt.xlabel('Iterations')
    plt.ylabel(f'Parameter {i}')
    plt.title(f'Parameter {i} vs. Iterations')
    plt.grid()
    plt.legend()
    plt.show()

    # Generate a grid of parameter values
    a_values = np.linspace(min(param_a), max(param_a), 100)
    b_values = np.linspace(min(param_a), max(param_a), 100)
    A, B = np.meshgrid(a_values, b_values)
    Z = np.zeros_like(A)

    # Compute the cost for each pair of (a, b)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            a = A[i, j]
            b = B[i, j]
            y_est = a * loss_values + b
            Z[i, j] = np.sum((loss_values - y_est) ** 2)

    # Plot the cost contour
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(A, B, Z, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel('Parameter a')
    plt.ylabel('Parameter b')
    plt.title('Cost Contour Plot')
    plt.grid()
    plt.show()

