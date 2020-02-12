import torch
import collections
import random

from settings import device

"""
store previous sequence-action pairs to decorrelate temporal locality
transitions are composed of state, action, reward, next_state, and done_mask
"""
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
        
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, next_state_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, next_state, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
            
        return  torch.cat(s_lst).to(device), \
                torch.tensor(a_lst).to(device), \
                torch.tensor(r_lst).to(device), \
                torch.cat(next_state_lst).to(device), \
                torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)

class ExperiencedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_limit):
        super().__init__(self, buffer_limit)