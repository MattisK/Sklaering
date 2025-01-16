import torch
from ChessCNN import ChessCNN
from ChessModel import ChessModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# Initialize the model
model = ChessCNN()

# Access the state_dict
state_dict = model.state_dict()
for name, param in state_dict.items():
    print(f"{name}: {param}")

with open("stateDict.txt", "w") as f:
    for name, param in state_dict.items():
        f.write(f"{name}: {param}\n")

        # Plot the model parameters
for name, param in state_dict.items():
        plt.figure()
        plt.title(name)
        if param.dim() == 1:
            plt.plot(param.numpy())
        elif param.dim() == 2:
            plt.imshow(param.numpy(), aspect='auto', cmap='viridis')
            plt.colorbar()
        elif param.dim() == 4:
            # For convolutional layers, plot the first filter
            plt.imshow(param[0, 0].numpy(), aspect='auto', cmap='viridis')
            plt.colorbar()
        plt.savefig(f"{name}.png")
        plt.close()


#calculate the mean and standard deviation of the weights
mean = []
std = []
var = []
for name, param in state_dict.items():
    mean.append(np.mean(param.numpy()))
    std.append(np.std(param.numpy()))
    var.append(np.var(param.numpy()))
print(f"mean:{mean}, std:{std}, var:{var}")




