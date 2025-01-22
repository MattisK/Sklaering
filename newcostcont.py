import numpy as np
import matplotlib.pyplot as plt
import torch
from ChessCNN import ChessCNN

model = ChessCNN()
model.load_state_dict(torch.load('chess_model_raw.pth', weights_only=True))
model.eval()


# Access the state_dict
state_dict = model.state_dict()

# Read the data from the file
loss_values = np.load("loss_history.npy")
if len(loss_values) == 0:
    raise ValueError("The length of loss_values is zero.")
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

