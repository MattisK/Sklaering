import torch
from torch.utils.data import Dataset
from functions import board_to_tensor, move_to_idx, encode_move


class ChessDataset(Dataset):
    """
    Inherits from PyTorch's Dataset class in order to hold a dataset consisting of different chess board states and the corresponding moves,
    and make sure we retrieve data points in a format that is suitable for training.
    """
    def __init__(self, data: list) -> None:
        """Takes data as a parameter, which is a tuple of a chess board object and a move object."""
        self.data = data

    
    def __len__(self) -> int:
        """Used for the PyTorch DataLoader class to determine the amount of samples available for iteration."""
        return len(self.data)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves and processes a single data point based on the index parameter idx."""
        # Extract the board and move from the data for a given index.
        board, move = self.data[idx]
        
        # For the board belonging to the current index gets a dictionary of possible moves with an index.
        move_idx = move_to_idx(board)

        # 12x8x8 one-hot encoded tensor.
        input_tensor = board_to_tensor(board)

        # Creates a tensor of target move indices.
        target_label = torch.tensor(encode_move(move, move_idx))

        return input_tensor, target_label
