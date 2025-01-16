from ChessDataset import ChessDataset
from ChessCNN import ChessCNN
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import numpy as np


class EarlyStopping:
    def __init__(self, patience=5) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    
    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train(
        model: ChessCNN,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        optimizer: torch.optim.Adam,
        criterion_policy: nn.CrossEntropyLoss,
        criterion_value: nn.MSELoss,
        epochs: int
        ) -> None:
    """Trains the model."""
    # Load training loss history.
    if os.path.exists("loss_history.npy"):
        loss_history = np.load("loss_history.npy", allow_pickle=True)
    else: 
        loss_history = []
    
    # Load validation loss history.
    if os.path.exists("val_loss_history.npy"):
        val_loss_history = np.load("val_loss_history.npy", allow_pickle=True)
    else: 
        val_loss_history = []

    early_stopping = EarlyStopping()

    # Training loop.
    for epoch in range(epochs):
        # Set the model to training mode.
        model.train()

        # Keeps track of the loss
        total_loss = 0.0

        # Fetches a sample from the dataloader which is an instance of 'DataLoader'.
        for boards, moves, results in train_dataloader:
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
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch + 1}/{epochs}, loss: {avg_train_loss}")
        loss_history.append(avg_train_loss)
        loss_np = np.array(loss_history)

        # Set model to evaluation mode.
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for boards, moves, results in validation_dataloader:
                boards = boards.to(device)
                moves = moves.to(device).long()
                results = results.to(device)

                policy, value = model(boards)

                loss_policy = criterion_policy(policy, moves)
                loss_value = criterion_value(value.squeeze(), results)
                loss = loss_policy + loss_value

                val_loss += loss.item()

        avg_val_loss = val_loss / len(validation_dataloader)
        print(f"Epoch: {epoch + 1}/{epochs}, validation loss: {avg_val_loss}")
        val_loss_history.append(avg_val_loss)
        val_loss_np = np.array(val_loss_history)

        # Check for early stopping.
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Triggered early stopping on epoch {epoch + 1}.")
            break
        
        # Save the model and loss.
        print("Saving model and loss.")
        torch.save(model.state_dict(), "chess_model.pth")
        np.save("loss_history.npy", loss_np, allow_pickle=True)
        np.save("val_loss_history.npy", val_loss_np, allow_pickle=True)
        print("Done saving model and loss.")


if __name__ == "__main__":
    pgn_file = "lichess_db_standard_rated_2014-09.pgn"
    batch_size = 64
    learning_rate = 0.001
    epochs = 50

    # Checks if cuda cores are available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and parse it to a DataLoader instance with a given batch size and shuffling.
    dataset = ChessDataset(pgn_file)

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print(f"Training samples: {len(train_dataset)}, validation samples: {len(validation_dataset)}.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model. If one already exists load that model.
    model = ChessCNN().to(device)
    if os.path.exists("chess_model.pth"):
        model.load_state_dict(torch.load("chess_model.pth"))

    # Optimizer and loss functions.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    # Train the model.
    train(model, train_dataloader, validation_dataloader, optimizer, criterion_policy, criterion_value, epochs)
