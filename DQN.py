import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions): # on init we define the layers of the network
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128) # currently we have 3 layers maximum number of moves # 4672
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    
    def forward(self, x): # take input x and pass it through the layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    