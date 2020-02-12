#import gym
#import argparse
#import math
#import os
#import tensorwatch as tw

#import torch
#import torch.optim as optim
import torch.nn.functional as F

#from tqdm import tqdm
# from apex import amp # playing around with mixed-precision training

# Local Imports
#from models import Qnet
#from wrappers import make_env
#from memory import ReplayBuffer
#from helpers import saveTrainedGameplay, get_state
#from settings import device

from DQN import DQN

'''
    Optimizes our training policy by computing the Huber Loss between our minibatch of samples and the maximum possible reward for the next state(s)
    Huber Loss here is defined as:
    loss(x,y) = \frac{1}{n}\sum{z_i}, where z_i = 0.5(x_i-y_i)^2; if |x_i - y_i| < 1 or 
                                                = |x_i - y_i| - 0.5; otherwise
    Double DQN Update Policy is:
    Y_t = R_{t+1} + \gamma * Q(S_{t+1},argmax Q(S_{t+1},a;theta_t);\theta_t^-)
'''
class DDQN(DQN):
    def __init__(self, env, save_location, start_episode = 1, saved_model = None):
        super().__init__(env, save_location, start_episode, saved_model)
        self.buffer_limit = 10 ** 5 * 2

    def train(self):
        s,a,r,s_prime,done_mask = self.memory.sample(self.batch_size)
        # Q_out is the observed transitions given the current network
        q_out = self.q(s)
        # collect output from the chosen action dimension
        q_a = q_out.gather(1,a)

        # DDQN Update
        argmax_q = self.q(s_prime).argmax(1).unsqueeze(1)
        q_prime = self.q_target(s_prime).gather(1,argmax_q)
        target = r + self.gamma * q_prime * done_mask

        # how much is our policy different from the true target 
        loss = F.smooth_l1_loss(q_a, target)
        self.optimizer.zero_grad()

        #with amp.scale_loss(loss, optimizer) as scaled_loss: # playing around with mixed-precision training
        	#scaled_loss.backward()
        loss.backward()
        self.optimizer.step()

