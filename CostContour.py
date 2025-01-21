import numpy as np
import matplotlib.pyplot as plt
import torch
from ChessCNN import ChessCNN


# Read the data from the file
loss_values = np.load("loss_history.npy")

# Initialize parameters

# Plot Loss vs. Iterations
plt.figure(figsize=(10, 5))
plt.plot(range(len(loss_values)), loss_values, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.grid()
plt.legend()
plt.show()
