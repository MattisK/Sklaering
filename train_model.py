from ChessDataset import ChessDataset
from ChessCNN import ChessCNN
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt


def train(model: ChessCNN, dataloader: DataLoader, optimizer: torch.optim.Adam, criterion_policy: nn.CrossEntropyLoss, criterion_value: nn.MSELoss, epochs: int) -> None:
    """Trains the model."""
    # Set the model to training mode.
    model.train()

    # Training loop.
    for epoch in range(epochs):
        # Keeps track of the loss
        total_loss = 0.0

        # Fetches a sample from the dataloader which is an instance of 'DataLoader'.
        for boards, moves, results in dataloader:
            # The samples fetched.
            boards = boards.to(device)
            moves = moves.to(device).long()
            results = results.to(device)

            # Reset the gradients from previous iteration.
            optimizer.zero_grad()

            # Get the policy and value from the model for a given board state.
            policy, value = model(boards)

            # Cross entropy loss function for the policy, since this is a logsoftmax function.
            loss_policy = criterion_policy(policy, moves)

            # Mean squared error loss function for the value, since this is a tanh function.
            loss_value = criterion_value(value.squeeze(), results)

            # The loss for the model is the total loss.
            loss = loss_policy + loss_value
            
            # Gradients for all models with respect to loss.
            loss.backward()

            # Adjust model parameters using the gradients.
            optimizer.step()

            # Update the total loss.
            total_loss += loss.item()
        
        print(f"Epoch: {epoch + 1}/{epochs}, loss: {total_loss / len(dataloader)}")

        # Save the model.
        torch.save(model.state_dict(), "chess_model.pth")


if __name__ == "__main__":
    pgn_file = "lichess_db_standard_rated_2014-09.pgn"
    batch_size = 64
    learning_rate = 0.001
    epochs = 100

    # Checks if cuda cores are available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and parse it to a DataLoader instance with a given batch size and shuffling.
    dataset = ChessDataset(pgn_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model. If one already exists load that model.
    model = ChessCNN().to(device)
    if os.path.exists("chess_model.pth"):
        model.load_state_dict(torch.load("chess_model.pth"))

    # Optimizer and loss functions.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    # Train the model.
    train(model, dataloader, optimizer, criterion_policy, criterion_value, epochs)

