import numpy as np
import matplotlib.pyplot as plt
import torch
from ChessCNN import ChessCNN

model = ChessCNN()
model.load_state_dict(torch.load('chess_model.pth', weights_only=True))
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
    print(loss_values[t])
    input_tensor = torch.tensor(loss_values[t], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 1, 8, 8)
    y_est = model(input_tensor)[0]
    loss = torch.nn.functional.mse_loss(y_est, input_tensor)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Ensure the gradients are of fixed size
    for param in model.parameters():
        if param.grad is not None:
            param.grad = param.grad.clone().detach()
    # Update parameters
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
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
    param_a = param_hist[0]
    param_b = param_hist[1]
    a_values = np.linspace(min(param_a), max(param_a), 100)
    b_values = np.linspace(min(param_b), max(param_b), 100)
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
