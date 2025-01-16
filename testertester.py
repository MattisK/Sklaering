import torch
from ChessCNN import ChessCNN
from ChessModel import ChessModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# Initialize the model
model = ChessCNN()
model.load_state_dict(torch.load('chess_model.pth'))
model.eval()

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

# confidence intervals for the mean, and standard deviation with alpha =0.05
data = mean
if len(data) == 0:
        raise ValueError("The data list must not be empty.")
data = np.array(data)
    # Beregn gennemsnittet
avg = np.mean(data)
    
    # Beregn standardafvigelsen
std_dev = np.std(data, ddof=1)  # ddof=1 for at få stikprøve-standardafvigelsen
    
    # Beregn standardfejlen (standardafvigelse / sqrt(n))
standard_error = std_dev / np.sqrt(len(data))
    
    # Beregn 95% konfidensinterval
t_value = stats.t.ppf(0.975, df=len(data) - 1)  # 0.975 for tosidet 95%
margin_of_error = t_value * standard_error
confidence_interval = (avg - margin_of_error, avg + margin_of_error)
    


print(avg, confidence_interval)
