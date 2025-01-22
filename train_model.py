from ChessDataset import ChessDataset
from ChessCNN import ChessCNN
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from functions import collate_fn


class EarlyStopping:
    def __init__(self, patience=10) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.rewind = False
        self.rewinded = False

    
    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.rewind = False
        else:
            self.rewind = True
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train(
        model: ChessCNN,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        optimizer: torch.optim.Adam,
        criterion: nn.CrossEntropyLoss,
        epochs: int
        ) -> None:
    """Trains the model."""
    # Initialize early stopping.
    early_stopping = EarlyStopping()

    # Lists for tracking loss
    loss_history = []
    val_loss_history = []

    # Training loop.
    for epoch in range(epochs):
        # Set the model to training mode.
        model.train()

        # Keeps track of the loss.
        total_loss = 0.0

        # Fetches a sample from the dataloader which is an instance of 'DataLoader'.
        for i, (boards_batch, moves_batch) in enumerate(train_dataloader):
            # The samples fetched.
            boards = torch.cat(boards_batch, dim=0).to(device)
            moves = torch.cat(moves_batch, dim=0).to(device)

            # Reset the gradients from previous iteration.
            optimizer.zero_grad()

            # Get the policy and value from the model for a given board state.
            policy = model(boards)

            # Cross entropy loss function for the policy, since this is a logsoftmax function.
            loss = criterion(policy, moves)
            
            # Gradients for all models with respect to loss.
            loss.backward()

            # Adjust model parameters using the gradients.
            optimizer.step()

            # Update the total loss.
            total_loss += loss.item()

            print(f"{i + 1}/{len(train_dataloader)}")
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch + 1}/{epochs}, loss: {avg_train_loss}")
        loss_history.append(avg_train_loss)
        loss_np = np.array(loss_history)

        # Set model to evaluation mode.
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i, (boards_batch, moves_batch) in enumerate(validation_dataloader):
                boards = torch.cat(boards_batch, dim=0).to(device)
                moves = torch.cat(moves_batch, dim=0).to(device)

                policy = model(boards)

                loss = criterion(policy, moves)

                val_loss += loss.item()

                print(f"{i + 1}/{len(validation_dataloader)}")

        avg_val_loss = val_loss / len(validation_dataloader)
        print(f"Epoch: {epoch + 1}/{epochs}, validation loss: {avg_val_loss}")
        val_loss_history.append(avg_val_loss)
        val_loss_np = np.array(val_loss_history)

        # Check for early stopping.
        early_stopping(avg_val_loss)
        if not early_stopping.rewind:
            print("Saving early stopping model.")
            torch.save(model.state_dict(), "chess_model_early_stopping.pth")
            print("Done saving early stopping model.")
        if early_stopping.early_stop:
            print(f"Triggered early stopping on epoch {epoch + 1}.")
            break
        
        # Save the model and loss.
        print("Saving model and loss.")
        torch.save(model.state_dict(), "chess_model_raw.pth")
        np.save("loss_history.npy", loss_np, allow_pickle=True)
        np.save("val_loss_history.npy", val_loss_np, allow_pickle=True)
        print("Done saving model and loss.")
        

if __name__ == "__main__":
    pgn_file = "lichess_db_standard_rated_2014-11.pgn"
    batch_size = 32
    learning_rate = 0.001
    epochs = 1000
    games = 100000

    # Checks if CUDA cores are available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and parse it to a DataLoader instance later with a given batch size and shuffling.
    dataset = ChessDataset(pgn_file, games)

    # Split the data in 80% training and 20% validation.
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print(f"Training samples: {len(train_dataset)}, validation samples: {len(validation_dataset)}.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize the model with CUDA.
    model = ChessCNN().to(device)

    # Optimizer and loss functions.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model.
    train(model, train_dataloader, validation_dataloader, optimizer, criterion, epochs)
