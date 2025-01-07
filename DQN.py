import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self) -> None:
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 64 * 64)

    
    def forward(self, x: torch.Tensor) -> nn.Linear:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        return self.fc2(x)
    