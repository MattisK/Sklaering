import torch
import torch.nn as nn


class ChessModel(nn.Module):
    """
    Convolution Neural Network (CNN) for the chess model.
    """
    def __init__(self) -> None:
        super(ChessModel, self).__init__()
        # Convolutional layers.
        self.conv = nn.Sequential(
            # 12 channels in the input tensor from board_to_tensor, and 8x8=64 convolutional filters (one for each square in the chess board).
            nn.Conv2d(12, 64, kernel_size=3),
            # Our non-linear activation funtion is ReLU.
            nn.ReLU(),
            # 2x64=128 convolutional filters.
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU()
        )
        # Fully connected (fc) layers.
        self.fc = nn.Sequential(
            # 128 for each of the squares i the chess grid.
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            # 4672 is the number of possible moves (legal moves, including promotions).
            nn.Linear(1024, 4672)
        )
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the flow through the network. Required for the neural network. Returns a tensor of shape (batch_size, 4672),
        where each row is a board in the batch, and each column is a numerical value for a specific move."""
        x = self.conv(x)

        # Reshape tensor
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
