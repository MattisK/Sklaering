import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from ChessDataset import ChessDataset
from ChessModel import ChessModel
import os
import numpy as np
import time

# Use cuda cores if available.
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Load the complete list of possible moves for each board state.
moves = np.load("moves.npy", allow_pickle=True)

# Create input tensors and unique labels for each move.
train_data = ChessDataset(moves)

# Wraps an iterable around the Dataset (train_data) to enable easy access to the samples.
# 32 board states per training iteration.
# Ensures data is randomly shuffled during training.
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Initiate model and check if a saved model already exists.
model = ChessModel().to(device)
if os.path.exists("chess_model.pth"):
    model.load_state_dict(torch.load("chess_model.pth"))

# Loss function.
criterion = nn.CrossEntropyLoss()

# Optimization algorithm with learning rate 0.001.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop. Executes 10 epochs.
time_var = time.time()
for epoch in range(10):
    # Iterate over batches of data from train_loader.
    for board_states, target_moves in train_loader:
        # Reset the gradients from previous iteration.
        optimizer.zero_grad()

        # Inputs the board states to the model to get predicted moves.
        outputs = model(board_states)

        # Calulates the loss (how far the predictions are from the true labels)
        loss = criterion(outputs, target_moves)
        
        # Gradients for all models with respect to loss.
        loss.backward()

        # Adjust model parameters using the gradients.
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the model's parameters.
    torch.save(model.state_dict(), "chess_model.pth")

print(f"Done training. Took {time.time() - time_var} seconds.")