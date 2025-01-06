from collections import deque, namedtuple
import random


class ReplayMemory(object):
    Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    
    def push(self, *args):
        self.memory.append(self.Transition(*args))

    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    
    def __len__(self):
        return len(self.memory)
