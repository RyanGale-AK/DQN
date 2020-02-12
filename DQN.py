import gym
import argparse
import math
import os
import tensorwatch as tw

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
#from apex import amp # playing around with mixed-precision training

# Local Imports
from models import Qnet
from wrappers import make_env
from memory import ReplayBuffer
from helpers import saveTrainedGameplay, get_state
from settings import device

'''
    Optimizes our training policy by computing the Huber Loss between our minibatch of samples and the maximum possible reward for the next state(s)
    Huber Loss here is defined as:
    loss(x,y) = \frac{1}{n}\sum{z_i}, where z_i = 0.5(x_i-y_i)^2; if |x_i - y_i| < 1 or 
                                                = |x_i - y_i| - 0.5; otherwise
'''
class DQN():
    def __init__(self, env, save_location, start_episode = 1, saved_model = None):
        self.env = env
        self.start_episode = start_episode
        self.save_location = save_location

        self.learning_rate = 1e-4
        self.gamma = 0.98
        self.buffer_limit = 10 ** 5
        self.training_frame_start = 10000
        self.batch_size = 32

        self.eps_start = 1
        self.eps_end = 0.01
        self.decay_factor = 10 ** 5
        
        self.q = Qnet(84,84, in_channels = 4, n_actions = 4).to(device)
        self.q_target = Qnet(84,84, in_channels = 4, n_actions = 4).to(device)
        self.memory = ReplayBuffer(buffer_limit = self.buffer_limit)

        if saved_model:
            self.epsilon_decay = lambda x: self.eps_end
            self.q.load_state_dict(torch.load(saved_model)) # Load pretrained model
        else:
            self.epsilon_decay = lambda x: self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * x / self.decay_factor)
        
        self.q_target.load_state_dict(self.q.state_dict()) # Load policy weights into target network
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        self.save_interval = 100000
        self.update_target_interval = 10000

        #[self.q, self.q_target], self.optimizer = amp.initialize([self.q, self.q_target], self.optimizer, opt_level="O1") #playing around with mixed-precision training


    def train(self):
        s,a,r,s_prime,done_mask = self.memory.sample(self.batch_size)

        q_out = self.q(s)
        # collect output from the chosen action dimension
        q_a = q_out.gather(1,a)

        # most reward we get in next state s_prime
        max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + self.gamma * max_q_prime * done_mask

        # how much is our policy different from the true target 
        loss = F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()

        #with amp.scale_loss(loss, self.optimizer) as scaled_loss: # playing around with mixed-precision training
        #	scaled_loss.backward()
        loss.backward()
        self.optimizer.step()

    def run(self, num_episodes):
        self.beginLogging()
        #watcher = tw.Watcher()
        env = self.env
        best_episode_score = float('-Inf')
        score = 0.0
        total_frames = 0
        state = get_state(env.reset()) # Start first game
        for episode in tqdm(range(self.start_episode,self.start_episode + num_episodes)):
            # anneal 100% to 1% over training
            epsilon = self.epsilon_decay(total_frames)
            episode_score = 0
            done = False
            while not done:
                action = self.q.sample_action(state.to(device), epsilon)

                obs, reward, done, info = env.step(action)

                next_state = get_state(obs)

                done_mask = 0.0 if done else 1.0
                self.memory.put((state,action,reward,next_state,done_mask))

                state = next_state

                score += reward
                episode_score += reward

                if total_frames > self.training_frame_start:
                    self.train()

                # Copy policy weights to target
                if total_frames%self.update_target_interval == 0:
                    self.q_target.load_state_dict(self.q.state_dict())
                # Save policy weights
                if total_frames%self.save_interval==0:
                    torch.save(self.q.state_dict(), os.path.join(self.save_location, 'policy_%s.pt' % episode))
                # Reset environment for the next game
                if done:
                    state = get_state(env.reset())
                total_frames += 1

            best_episode_score = max(best_episode_score, episode_score)
            # Print updates every episode
            out = "n_episode : {}, Total Frames : {}, Average Score : {:.1f}, Episode Score : {:.1f}, Best Score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                episode, total_frames, score/episode, episode_score, best_episode_score, self.memory.size(), epsilon*100)
            print(out)
            self.log(out)
            

            # Microsoft Tensorwatch Watcher for Visualizing Training
            #watcher.observe(
            #    episode = episode,
            #    episode_score = episode_score,
            #    total_score = score,
            #    buffer_size = self.memory.size(),
            #    epsilon = epsilon,
            #    frames = total_frames,
            #)

        # save final model weights
        torch.save(self.q.state_dict(), os.path.join(self.save_location, 'policy_final.pt'))

    def beginLogging(self):
        with open(os.path.join(self.save_location, 'log.out'), 'w') as f:
            f.write('')
    
    def log(self, out):
        with open(os.path.join(self.save_location, 'log.out'), 'a') as f:
                        f.write('%s\n'%out)