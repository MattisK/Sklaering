import torch
import torch.nn as nn


class ChessModel(nn.Module):
    """
    Convolution Neural Network (CNN) for the chess model.
    """
    def __init__(self) -> None:
        super(ChessModel, self).__init__()
        # Convolutional layers.
        self.conv = nn.Sequential( # TODO: Add more convolutional layers, potentially
            
            # Block 1
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 2x2 -> 1x1   
        )
        
        """# 12 channels in the input tensor from board_to_tensor, and 8x8=64 convolutional filters (one for each square in the chess board).
            nn.Conv2d(12, 12*64, kernel_size=3),
            # Our non-linear activation funtion is ReLU.
            nn.ReLU(),
            # 2x64=128 convolutional filters.
            nn.Conv2d(12*64, 12*64*9),
            nn.ReLU(),
            # 2x128 = 256 convolutional filters.
            nn.Conv2d(12*64*9, 12*64, kernel_size=2),
            nn.ReLU()"""
        
        self.fc = nn.Sequential(
            nn.Linear(256, 1024),  # Fra 256 (output fra pooling) til 1024 neuroner
            nn.ReLU(),
            nn.Dropout(p=0.1),  # Dropout for at undgå overfitting
            nn.Linear(1024, 4672)  # Output-dimensionen svarer til antallet af mulige træk
        )
        """
        # Fully connected (fc) layers.
        self.fc = nn.Sequential(
            # 128 for each of the squares in the chess grid.
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            # 4672 is the number of possible moves (legal moves, including promotions).
            nn.Linear(1024, 4672)
        )
        """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # TODO: Maybe make it so, that it only returns an evaluation of the board state.
        """Defines the flow through the network. Required for the neural network. Returns a tensor of shape (batch_size, 4672),
        where each row is a board in the batch, and each column is a numerical value for a specific move."""
        x = self.conv(x)

        # Reshape tensor
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
