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

    # Example data for X and Y
    X = np.random.rand(100, 2)  # 100 samples, 2 features
    Y = np.random.rand(100)     # 100 target values

    # Normalize the features
    X_norm, mean, std = normalize(X)

    # Add bias feature
    X_bias = add_bias_feature(X_norm)

    # Initialize theta (parameters) with zeros
    theta = np.zeros(X_bias.shape[1])

    # Set learning rate and number of iterations
    alpha = 0.01
    learning_iterations = 1000

    # Perform gradient descent
    theta, cost_history = gradient_descent(X_bias, Y, theta, alpha, learning_iterations)

    # Print the final parameters and cost history
    print("Final parameters (theta):", theta)
    print("Cost history:", cost_history)

    # Plot the cost history
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost History over Iterations')
    plt.show()