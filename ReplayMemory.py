from collections import deque, namedtuple
import random


class ReplayMemory:
    Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


    def __init__(self, capacity): # on initialzation create memory with a size of capacity
        self.memory = deque([], maxlen=capacity)
    
    
    def push(self, *args):
        self.memory.append(self.Transition(*args)) #add a named tuple to memory

    
    def sample(self, batch_size): # Takes a random sample of size batch_size from the memory
        return random.sample(self.memory, batch_size)
    
    
    def __len__(self): # returns the length(size) of the memory
        return len(self.memory)
