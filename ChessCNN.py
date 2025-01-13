import torch.nn as nn
import torch


class ChessCNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for the chess model.
    """
    def __init__(self) -> None:
        """
        Initialize the layers of the chess model.
        """
        super(ChessCNN, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Convolutional layer 2, will be called multiple times.
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Policy network. Convolutional and fully connected layers.
        self.net1 = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4672),
            nn.LogSoftmax()
        )

        # Value network. Convolutional and fully connected layers.
        self.net2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        From documentation: Define the computation performed at every call.
        """
        x = self.conv1(x)
        
        for _ in range(6):
            x = self.conv2(x)
        
        policy = self.net1(x)
        value = self.net2(x)

        return policy, value
