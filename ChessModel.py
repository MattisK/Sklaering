import torch
import torch.nn as nn
import numpy as np


# Use cuda cores if available.
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class ChessModel(nn.Module):
    """
    Convolution Neural Network (CNN) for the chess model.
    """
    def __init__(self) -> None:
        super(ChessModel, self).__init__()
        
        """
        A simplified neural network to evaluate a chess board.
        Input: 12x8x8 board state
        Output
        """
        
        self.net = nn.Sequential( # TODO: Add more convolutional layers, potentially
            # Convolutional layers.
            # Block 1
            nn.Conv2d(12, 64, kernel_size=3, padding=1), # input 12x8x8 -> output 64x8x8
            nn.ReLU(),
            nn.Dropout(p=0.05),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), #input 64x8x8 -> output 128x8x8
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.AvgPool2d(kernel_size=2, stride=1),  # input 128x8x8 -> output 128x7x7

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # input 128x7x7 -> output 256x7x7
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.MaxPool2d(kernel_size=2, stride=1),  # input 256x7x7 -> output 256x6x6

            # Flatten
            nn.Flatten(), # input 256x6x6 -> output 9216
            
            nn.Linear(9216, 1),  # Fra 256 (output fra pooling) til 1024 neuroner
            nn.LeakyReLU(1), # TODO: change this to allow negative values
            nn.Dropout(p=0.05),  # Dropout for at undgå overfitting
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # TODO: Maybe make it so, that it only returns an evaluation of the board state.
        """Defines the flow through the network. Required for the neural network. Returns a tensor of shape (batch_size, 4672),
        where each row is a board in the batch, and each column is a numerical value for a specific move."""
        x = self.net(x)

        return x
    
    def select_action(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(len(moves))
        else:
            with torch.no_grad():
                q_values = ChessModel(state)
            return torch.argmax(q_values).item()

    # Function to update the target Q-network
    def update_target_network():
        ChessModel.load_state_dict(ChessModel.state_dict())
