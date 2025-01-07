import random
import math
import torch
from DQN import DQN


device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

n_actions = None
state, info = None, None
n_observations = len(state)


EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000

policy_net = DQN()

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY)
    steps_done += 1

    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_net(state)