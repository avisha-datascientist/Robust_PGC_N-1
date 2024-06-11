import os
import random
import numpy as np
import csv
import pickle
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.memory = deque(maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(list(self.memory), n)

    def extend_memory(self, path):
        data = pickle.load(open(os.path.join(path, f'memory.pkl'), 'rb'))
        self.memory.extend(data)
        print(self.memory.__len__())
    
    def store_memory(self,path):
        pickle.dump(self.memory, open(os.path.join(path, f'memory.pkl'), 'wb'))
