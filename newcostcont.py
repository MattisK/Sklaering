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
print(loss_values)
# Count the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X_norm = (X - mean) / std
    return X_norm, mean, std

def add_bias_feature(X):
    return np.column_stack((np.ones(len(X)),X))

def compute_cost(X,Y,theta):
    m = len(Y)
    h = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(h-Y))
    return cost

def gradient_descent(X,Y,theta,alpha,learning_iterations):

    m = len(Y)
    cost_history = []
    for i in range(learning_iterations):
        h = X.dot(theta)
        error = h-Y  # loss
        gradient = (1/m)*X.T.dot(error)
        theta = theta - alpha*gradient
        cost = compute_cost(X,Y,theta)
        cost_history.append(cost)
    return theta,cost_history

min_length = min(len(loss_values), len([param.numel() for param in model.parameters() if param.requires_grad]))
X = loss_values[:min_length]
Y = np.array([param.numel() for param in model.parameters() if param.requires_grad][:min_length])

X_norm, mean_X, std_X = normalize(X)

X_bias = add_bias_feature(X_norm)

theta_initial = np.zeros(2)

alpha = 0.01

num_iterations = 1000

theta_final, cost_history = gradient_descent(X_bias,Y,theta_initial,alpha,num_iterations)

print(f"Final theta: {theta_final}")
print(f"Final cost: {cost_history[-1]}")

plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()

plt.plot(X, loss_values[:min_length], ".-")
a_mesh = np.linspace(theta_final[1] - 0.5, theta_final[1] + 0.5, 100)
b_mesh = np.linspace(theta_final[0] - 0.5, theta_final[0] + 0.5, 100)
B, A = np.meshgrid(b_mesh, a_mesh)
Z = np.sum((A[:, :, np.newaxis] * X_bias[:, 0][np.newaxis, np.newaxis] + B[:, :, np.newaxis] * X_bias[:, 1][np.newaxis, np.newaxis] - Y[np.newaxis, np.newaxis]) ** 2, 2)
plt.contour(B, A, np.log(Z), 20)
plt.gca().set_aspect("equal", "box")
plt.xlabel("b")
plt.ylabel("a")
plt.grid(True)
plt.title("Contour plot of cost function")
plt.show()
