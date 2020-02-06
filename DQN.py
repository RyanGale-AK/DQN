import gym
import math
import random
import cv2
import numpy as np
import collections
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(env):
    # convert to channel,h,w dimensions
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    
    # erase background
    screen[screen == 72] = 0 
    screen[screen == 74] = 0 
    screen[screen == 144] = 0 
    screen[screen != 0] = 213
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    
    screen = torch.from_numpy(screen)
    
    # convert to batch,channel,h,w dimensions
    return resize(screen).unsqueeze(0).to(device)


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
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            
        return  torch.cat(s_lst).to(device), \
                torch.tensor(a_lst).to(device), \
                torch.tensor(r_lst).to(device), \
                torch.cat(s_prime_lst).to(device), \
                torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, h, w):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
    # action is either random or max probability estimated by Qnet
    def sample_action(self, obs, epsilon):
        out = self.forward(obs) # don't need if random action 
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer, batch_size, gamma):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        
        q_out = q(s)
        # collect output from the chosen action dimension
        q_a = q_out.gather(1,a) 
        
        # most reward we get in next state s_prime
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        # how much is our policy different from the true target 
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def main(num_episodes):
    learning_rate = 0.0005
    gamma = 0.98
    buffer_limit = 50000
    batch_size = 32

    env = gym.make('Pong-v0')
    _, _, h, w = get_screen(env).shape
    q = Qnet(h,w).to(device)
    q_target = Qnet(h,w).to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(buffer_limit = buffer_limit)
    
    save_interval = 250
    print_interval = 50
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    for episode in tqdm(range(1,num_episodes+1)):
        # anneal 8% to 1% over training
        epsilon = max(0.01, 0.08 - 0.01*(episode/200))
        env.reset()
        current_s = get_screen(env)
        done = False
        last_s = get_screen(env)
        current_s = get_screen(env)
        s = last_s - current_s
        episode_score = 0
        while not done:
            a = q.sample_action(s, epsilon)
            # first variable would be s_prime but we have get_screen
            _, r, done, info = env.step(a + 2)
            last_s = current_s
            current_s = get_screen(env)
            s_prime = last_s - current_s
            
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.,s_prime,done_mask))
            s = s_prime
            
            score += r
            episode_score += r
            if done:
                break
        if memory.size()>2000:
            train(q, q_target, memory, optimizer, batch_size, gamma)
        
        if episode%print_interval==0 and episode!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode : {}, Average Score : {:.1f}, Episode Score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                episode, score/episode, episode_score, memory.size(), epsilon*100))
            
        if episode%save_interval==0 and episode!=0:
            # save model weights 
            torch.save(q_target.state_dict(), 'checkpoints/target_bot_%s.pt' % episode)
            torch.save(q.state_dict(), 'checkpoints/policy_bot_%s.pt' % episode)
    # save final model weights 
    torch.save(q_target.state_dict(), 'checkpoints/target_bot_final.pt')
    torch.save(q.state_dict(), 'checkpoints/policy_bot_final.pt')

# record trained agent gameplay
def getTrainedGameplay(target_bot):
    env = gym.make('Pong-v0')
    _, _, h, w = get_screen(env).shape
    q = Qnet(h,w).to(device)
    q.load_state_dict(torch.load('checkpoints/%s.pt' % target_bot))
    q.eval()
    frames = []

    env.reset()
    current_s = get_screen(env)
    done = False
    last_s = get_screen(env)
    current_s = get_screen(env)
    s = last_s - current_s
    epsilon = 0.0
    while not done:
        a = q.sample_action(s, epsilon) + 2
        
        # use environment's frame instead of preprocessed get_screen(env)
        next_frame, _, done, info = env.step(a)
        frames.append(next_frame)
        last_s = current_s
        current_s = get_screen(env)
        s_prime = last_s - current_s

        done_mask = 0.0 if done else 1.0
        s = s_prime
        if done:
            return frames


def saveGameVideo(frames):
    # save game to video 
    height, width = frames[0].shape[:2] 

    writer = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 30
    video_file = 'playback.avi'
    out = cv2.VideoWriter(video_file, writer, fps, (width,height))
    for frame in frames:
        out.write(frame)

    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN Model on Pong-v0')
    parser.add_argument('--episodes', '-e', type=int, default=1000)

    args = parser.parse_args()
    
    main(args.episodes)
    frames = getTrainedGameplay('target_bot_final')
    saveGameVideo(frames)



